//! Fixed FP32 text-only EmbeddingGemma 2 contract. No automatic downloads.

use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::SystemTime,
};

use ndarray::Array2;
use ort::{
    session::{OutputSelector, RunOptions, Session},
    value::{TensorElementType, Value, ValueType},
};
use sha2::{Digest, Sha256};
use tokenizers::{Tokenizer, TruncationParams};

pub use crate::ep::OnnxExecution;
use crate::{
    ep::{build_text_session, text_execution},
    InferError,
};

pub const MODEL: &str = "embeddinggemma2_text_256";
pub const REVISION: &str = "daa72c51243991dfcaf9f9137d2c573d8f7790c0";
pub const PREPROCESSING: &str = "eg2-search-v1";
pub const DIM: usize = 256;
pub const MAX_TEXT_BYTES: usize = 16 * 1024;

const ARTIFACTS: [(&str, u64, &str); 3] = [
    (
        "onnx/model.onnx",
        421421,
        "bc47de15f81208a5c99e5ab10f746d5e33b51ea228b7dc0bef9c133a94f1c1c3",
    ),
    (
        "onnx/model.onnx_data",
        1084170240,
        "9fd452bfc6916e5a92f080f8dada35598ca4512b21c65897d4c5974d5f9630a2",
    ),
    (
        "tokenizer.json",
        32170510,
        "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    ),
];

#[derive(Clone, Copy, Debug, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TextTask {
    #[default]
    Query,
    Document,
}

/// The caller supplies raw text; only this function adds search prefixes.
pub fn prepare_text(
    text: &str,
    title: Option<&str>,
    task: TextTask,
) -> Result<String, &'static str> {
    let title = title.unwrap_or("");
    if text.len().saturating_add(title.len()) > MAX_TEXT_BYTES {
        return Err("text and title exceed 16384 bytes");
    }
    match task {
        TextTask::Query if text.trim().is_empty() => Err("query text is empty"),
        TextTask::Query => Ok(format!("task: search result | query: {text}")),
        TextTask::Document if text.trim().is_empty() && title.trim().is_empty() => {
            Err("document text and title are empty")
        }
        TextTask::Document => Ok(format!(
            "title: {} | text: {text}",
            if title.is_empty() { "none" } else { title }
        )),
    }
}

#[derive(Debug, PartialEq, Eq)]
struct FileStamp {
    path: PathBuf,
    len: u64,
    modified: SystemTime,
    #[cfg(unix)]
    identity: (u64, u64, i64, i64),
}

/// Source metadata invalidates caches; only the verified snapshot binds loaded bytes.
#[derive(Debug, PartialEq, Eq)]
struct Identity {
    revision: &'static str,
    preprocessing: &'static str,
    files: Vec<FileStamp>,
}

impl Identity {
    fn read(dir: &Path) -> Result<Self, InferError> {
        let mut files = Vec::with_capacity(ARTIFACTS.len());
        for (name, len, _) in ARTIFACTS {
            let path = dir.join(name);
            let meta = fs::metadata(&path).map_err(|error| {
                if error.kind() == std::io::ErrorKind::NotFound {
                    InferError::ModelNotDownloaded(dir.to_owned())
                } else {
                    error.into()
                }
            })?;
            if !meta.is_file() || meta.len() != len {
                return Err(invalid(format!("invalid artifact size/type: {name}")));
            }
            files.push(FileStamp {
                path: path.canonicalize()?,
                len,
                modified: meta.modified()?,
                #[cfg(unix)]
                identity: {
                    use std::os::unix::fs::MetadataExt;
                    (meta.dev(), meta.ino(), meta.ctime(), meta.ctime_nsec())
                },
            });
        }
        Ok(Self {
            revision: REVISION,
            preprocessing: PREPROCESSING,
            files,
        })
    }
}

fn copy_verified_artifact(
    source: &Path,
    target: &Path,
    size: u64,
    expected: &str,
) -> Result<(), InferError> {
    let mut source = fs::File::open(source)?.take(size + 1);
    let mut target = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut total = 0;
    loop {
        let n = source.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        // Hash the very bytes written to the private files ORT/tokenizers open.
        target.write_all(&buffer[..n])?;
        hash.update(&buffer[..n]);
        total += n as u64;
    }
    if total != size || format!("{:x}", hash.finalize()) != expected {
        return Err(invalid("artifact SHA-256 or size mismatch"));
    }
    Ok(())
}

fn snapshot_artifacts(dir: &Path) -> Result<tempfile::TempDir, InferError> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("yu-eg2-");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        builder.permissions(fs::Permissions::from_mode(0o700));
    }
    let snapshot = builder.tempdir()?;
    fs::create_dir(snapshot.path().join("onnx"))?;
    for (name, size, expected) in ARTIFACTS {
        copy_verified_artifact(&dir.join(name), &snapshot.path().join(name), size, expected)?;
    }
    Ok(snapshot)
}

fn invalid(message: impl Into<String>) -> InferError {
    InferError::InvalidModelOutput(message.into())
}

/// One resident model bounds RAM even when callers alternate model directories.
#[derive(Default)]
pub struct EmbeddingGemma2Cache {
    cached: Option<(Identity, Encoder)>,
}

impl EmbeddingGemma2Cache {
    pub fn execution(&self) -> OnnxExecution {
        self.cached
            .as_ref()
            .map_or_else(text_execution, |(_, encoder)| encoder.execution.clone())
    }

    pub fn busy_execution() -> OnnxExecution {
        let mut execution = text_execution();
        execution.loaded = None;
        execution
    }

    pub fn encode(
        &mut self,
        dir: &Path,
        text: &str,
        title: Option<&str>,
        task: TextTask,
    ) -> Result<Vec<f32>, InferError> {
        let prepared = prepare_text(text, title, task).map_err(invalid)?;
        let identity = Identity::read(dir)?;
        if self.cached.as_ref().map(|(key, _)| key) != Some(&identity) {
            self.cached = None;
            let encoder = Encoder::load(snapshot_artifacts(dir)?)?;
            if Identity::read(dir)? != identity {
                return Err(invalid("artifacts changed during model load"));
            }
            self.cached = Some((identity, encoder));
        }
        let (identity, encoder) = self.cached.as_mut().expect("model just loaded");
        let vector = encoder.encode(&prepared)?;
        if Identity::read(dir)? != *identity {
            self.cached = None;
            return Err(invalid("artifacts changed during inference"));
        }
        Ok(vector)
    }
}

struct Encoder {
    session: Session,
    execution: OnnxExecution,
    tokenizer: Tokenizer,
    // Drop after Session: external tensor data may remain mapped for its lifetime.
    _snapshot: tempfile::TempDir,
}

impl Encoder {
    fn load(snapshot: tempfile::TempDir) -> Result<Self, InferError> {
        let dir = snapshot.path();
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| InferError::Tokenizer(e.to_string()))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 8192,
                ..Default::default()
            }))
            .map_err(|e| InferError::Tokenizer(e.to_string()))?;
        // Match the four CPU threads used by the host adoption benchmark.
        let (session, execution) = build_text_session(&dir.join("onnx/model.onnx"))?;
        validate_contract(&session)?;
        Ok(Self {
            session,
            execution,
            tokenizer,
            _snapshot: snapshot,
        })
    }

    fn encode(&mut self, text: &str) -> Result<Vec<f32>, InferError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| InferError::Tokenizer(e.to_string()))?;
        let input_ids = Array2::from_shape_vec(
            (1, encoding.len()),
            encoding.get_ids().iter().map(|&v| i64::from(v)).collect(),
        )
        .map_err(|e| invalid(e.to_string()))?;
        let attention_mask = Array2::from_shape_vec(
            (1, encoding.len()),
            encoding
                .get_attention_mask()
                .iter()
                .map(|&v| i64::from(v))
                .collect(),
        )
        .map_err(|e| invalid(e.to_string()))?;
        let options = RunOptions::new()?
            .with_outputs(OutputSelector::no_default().with("sentence_embedding"));
        let outputs = self.session.run_with_options(
            ort::inputs![
                "input_ids" => Value::from_array(input_ids)?,
                "attention_mask" => Value::from_array(attention_mask)?,
                "image_features" => Value::from_array(Array2::<f32>::zeros((0, 512)))?,
                "video_features" => Value::from_array(Array2::<f32>::zeros((0, 512)))?,
                "audio_features" => Value::from_array(Array2::<f32>::zeros((0, 512)))?,
            ],
            &options,
        )?;
        #[cfg(test)]
        assert_eq!(
            outputs.len(),
            1,
            "only sentence_embedding should be fetched"
        );
        let output = outputs
            .get("sentence_embedding")
            .ok_or_else(|| invalid("missing sentence_embedding"))?;
        let (shape, values) = output.try_extract_tensor::<f32>()?;
        normalize(shape, values)
    }
}

fn tensor_matches(dtype: &ValueType, expected: TensorElementType, dims: &[i64]) -> bool {
    matches!(dtype, ValueType::Tensor { ty, shape, .. } if *ty == expected && shape.len() == dims.len() && shape.iter().zip(dims).all(|(&actual, &wanted)| actual == -1 || actual == wanted))
}

fn validate_contract(session: &Session) -> Result<(), InferError> {
    if session.inputs().len() != 5 {
        return Err(invalid("expected five EG2 inputs"));
    }
    for (name, ty, dims) in [
        ("input_ids", TensorElementType::Int64, [1, -1]),
        ("attention_mask", TensorElementType::Int64, [1, -1]),
        ("image_features", TensorElementType::Float32, [0, 512]),
        ("video_features", TensorElementType::Float32, [0, 512]),
        ("audio_features", TensorElementType::Float32, [0, 512]),
    ] {
        if !session
            .inputs()
            .iter()
            .any(|v| v.name() == name && tensor_matches(v.dtype(), ty, &dims))
        {
            return Err(invalid(format!("invalid EG2 input: {name}")));
        }
    }
    if !session.outputs().iter().any(|v| {
        v.name() == "sentence_embedding"
            && tensor_matches(v.dtype(), TensorElementType::Float32, &[1, 768])
    }) {
        return Err(invalid("invalid sentence_embedding output"));
    }
    Ok(())
}

fn normalize(shape: &[i64], values: &[f32]) -> Result<Vec<f32>, InferError> {
    if shape != [1, 768] || values.len() != 768 || values.iter().any(|x| !x.is_finite()) {
        return Err(invalid(
            "expected finite sentence_embedding with shape (1,768)",
        ));
    }
    let mut vector = values[..DIM].to_vec();
    let norm = vector
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return Err(invalid("zero/nonfinite embedding norm"));
    }
    for x in &mut vector {
        *x = (f64::from(*x) / norm) as f32;
    }
    Ok(vector)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_fetches_only_sentence_and_matches_all_outputs() {
        let mut session = Session::builder()
            .unwrap()
            .commit_from_memory(include_bytes!("../tests/fixtures/eg2_two_outputs.onnx"))
            .unwrap();
        validate_contract(&session).unwrap();
        let baseline = session
            .run(ort::inputs![
                "input_ids" => Value::from_array(Array2::<i64>::zeros((1, 1))).unwrap(),
                "attention_mask" => Value::from_array(Array2::<i64>::ones((1, 1))).unwrap(),
                "image_features" => Value::from_array(Array2::<f32>::zeros((0, 512))).unwrap(),
                "video_features" => Value::from_array(Array2::<f32>::zeros((0, 512))).unwrap(),
                "audio_features" => Value::from_array(Array2::<f32>::zeros((0, 512))).unwrap(),
            ])
            .unwrap();
        assert_eq!(baseline.len(), 2);
        assert!(baseline.get("last_hidden_state").is_some());
        let (shape, values) = baseline["sentence_embedding"]
            .try_extract_tensor::<f32>()
            .unwrap();
        let expected = normalize(shape, values).unwrap();
        drop(baseline);
        let tokenizer = Tokenizer::new(
            tokenizers::models::wordlevel::WordLevel::builder()
                .vocab([("fixture".to_owned(), 0)].into_iter().collect())
                .unk_token("fixture".to_owned())
                .build()
                .unwrap(),
        );
        let mut encoder = Encoder {
            session,
            execution: text_execution(),
            tokenizer,
            _snapshot: tempfile::tempdir().unwrap(),
        };
        let actual = encoder.encode("fixture").unwrap();
        assert_eq!(actual.len(), DIM);
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let norm = actual
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-7);
        assert!(actual[DIM - 1] > actual[0]);
    }

    #[test]
    fn execution_is_unknown_before_load_and_while_busy() {
        let unloaded = EmbeddingGemma2Cache::default().execution();
        assert_eq!(unloaded.loaded, Some(false));
        assert!(unloaded.registered_providers.is_none());
        let busy = EmbeddingGemma2Cache::busy_execution();
        assert_eq!(busy.loaded, None);
        assert!(busy.registered_providers.is_none());
        assert_eq!(busy.requested_providers, unloaded.requested_providers);
    }

    const PARITY: &str = include_str!("../tests/fixtures/embeddinggemma2_fp32_parity.json");

    fn parity_cosine(actual: &[f32], expected: &[f32]) -> Result<f64, &'static str> {
        if actual.len() != DIM
            || expected.len() != DIM
            || actual.iter().chain(expected).any(|v| !v.is_finite())
        {
            return Err("parity vectors must have 256 finite values");
        }
        let norm = |v: &[f32]| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
        let (actual_norm, expected_norm) = (norm(actual), norm(expected));
        if (expected_norm - 1.0).abs() > 1e-5 || (actual_norm - 1.0).abs() > 1e-5 {
            return Err("parity vectors must have unit norm");
        }
        Ok(actual
            .iter()
            .zip(expected)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>()
            / (actual_norm * expected_norm))
    }

    #[test]
    fn parity_rejects_scaled_nonfinite_or_wrong_dimension_references() {
        let expected = vec![0.0625; DIM];
        assert!((parity_cosine(&expected, &expected).unwrap() - 1.0).abs() < 1e-12);
        let nearly_unit: Vec<f32> = expected.iter().map(|v| v * (1.0 - 1e-6)).collect();
        assert!((parity_cosine(&expected, &nearly_unit).unwrap() - 1.0).abs() < 1e-12);
        assert!(parity_cosine(&expected, &[0.125; DIM]).is_err());
        assert!(parity_cosine(&expected, &[f32::NAN; DIM]).is_err());
        assert!(parity_cosine(&expected, &[f32::INFINITY; DIM]).is_err());
        assert!(parity_cosine(&expected, &[0.0; DIM]).is_err());
        assert!(parity_cosine(&expected, &expected[..255]).is_err());
    }

    #[test]
    fn checked_in_parity_reference_has_valid_provenance_and_vectors() {
        let provenance: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/embeddinggemma2_fp32_parity.provenance.json"
        ))
        .unwrap();
        assert_eq!(provenance["revision"], REVISION);
        assert_eq!(provenance["preprocessing"], PREPROCESSING);
        assert_eq!(provenance["precision"], "fp32");
        assert_eq!(
            provenance["fixture_sha256"],
            format!("{:x}", Sha256::digest(PARITY.as_bytes()))
        );
        let rows: Vec<serde_json::Value> = serde_json::from_str(PARITY).unwrap();
        assert!(!rows.is_empty());
        for row in rows {
            let expected: Vec<f32> = serde_json::from_value(row["vector"].clone()).unwrap();
            assert!(parity_cosine(&expected, &expected).is_ok());
        }
    }

    #[test]
    fn copied_artifact_is_bound_to_verified_bytes_after_source_changes() {
        let source_dir = tempfile::tempdir().unwrap();
        let snapshot = tempfile::tempdir().unwrap();
        let source = source_dir.path().join("weights");
        let target = snapshot.path().join("weights");
        fs::write(&source, b"verified").unwrap();
        let hash = format!("{:x}", Sha256::digest(b"verified"));
        copy_verified_artifact(&source, &target, 8, &hash).unwrap();
        fs::write(&source, b"modified").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"verified");
        assert!(
            copy_verified_artifact(&source, &snapshot.path().join("bad-hash"), 8, &hash).is_err()
        );
        fs::write(&source, b"verified-extra").unwrap();
        assert!(
            copy_verified_artifact(&source, &snapshot.path().join("too-long"), 8, &hash).is_err()
        );
        fs::write(&source, b"short").unwrap();
        assert!(
            copy_verified_artifact(&source, &snapshot.path().join("too-short"), 8, &hash).is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn original_symlink_swap_cannot_change_snapshot_load_bytes() {
        use std::os::unix::fs::symlink;
        let source = tempfile::tempdir().unwrap();
        let snapshot = tempfile::tempdir().unwrap();
        let a = source.path().join("a");
        let b = source.path().join("b");
        let link = source.path().join("current");
        fs::write(&a, b"verified").unwrap();
        fs::write(&b, b"modified").unwrap();
        symlink(&a, &link).unwrap();
        let target = snapshot.path().join("model");
        copy_verified_artifact(
            &link,
            &target,
            8,
            &format!("{:x}", Sha256::digest(b"verified")),
        )
        .unwrap();
        fs::remove_file(&link).unwrap();
        symlink(&b, &link).unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"verified");
        fs::remove_file(&link).unwrap();
        symlink(&a, &link).unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"verified");
    }

    #[test]
    fn prefixes_and_input_bounds() {
        assert_eq!(
            prepare_text("猫", None, TextTask::Query).unwrap(),
            "task: search result | query: 猫"
        );
        assert_eq!(
            prepare_text("猫", None, TextTask::Document).unwrap(),
            "title: none | text: 猫"
        );
        assert_eq!(
            prepare_text("", Some("猫.jpg"), TextTask::Document).unwrap(),
            "title: 猫.jpg | text: "
        );
        assert!(prepare_text(" ", None, TextTask::Query).is_err());
        assert!(prepare_text("", None, TextTask::Document).is_err());
        assert!(prepare_text(&"a".repeat(MAX_TEXT_BYTES), Some("x"), TextTask::Document).is_err());
    }

    #[test]
    fn truncates_before_normalizing_and_rejects_bad_outputs() {
        let mut values = vec![2.0; 768];
        values[256..].fill(100.0);
        let vector = normalize(&[1, 768], &values).unwrap();
        assert_eq!(vector, vec![0.0625; 256]);
        assert!(normalize(&[768], &values).is_err());
        assert!(normalize(&[1, 768], &[0.0; 768]).is_err());
        values[767] = f32::NAN;
        assert!(normalize(&[1, 768], &values).is_err());
        values[767] = f32::INFINITY;
        assert!(normalize(&[1, 768], &values).is_err());
    }

    #[test]
    fn rejects_missing_wrong_size_and_wrong_hash_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("onnx")).unwrap();
        assert!(matches!(
            Identity::read(dir.path()),
            Err(InferError::ModelNotDownloaded(_))
        ));
        fs::write(dir.path().join("onnx/model.onnx"), b"bad").unwrap();
        assert!(Identity::read(dir.path()).is_err());
        for (name, size, _) in ARTIFACTS {
            fs::File::create(dir.path().join(name))
                .unwrap()
                .set_len(size)
                .unwrap();
        }
        let identity = Identity::read(dir.path()).unwrap();
        assert!(snapshot_artifacts(dir.path()).is_err());
        fs::remove_file(dir.path().join("onnx/model.onnx")).unwrap();
        fs::File::create(dir.path().join("onnx/model.onnx"))
            .unwrap()
            .set_len(ARTIFACTS[0].1)
            .unwrap();
        assert_ne!(identity, Identity::read(dir.path()).unwrap());
    }

    #[test]
    #[ignore = "requires fixed FP32 artifacts in EG2_MODEL_DIR"]
    fn real_model_parity() {
        let dir = PathBuf::from(std::env::var("EG2_MODEL_DIR").expect("EG2_MODEL_DIR"));
        let fixture: serde_json::Value = serde_json::from_str(PARITY).unwrap();
        let mut cache = EmbeddingGemma2Cache::default();
        let rows = fixture.as_array().unwrap();
        assert!(!rows.is_empty(), "parity fixture must contain examples");
        let mut minimum_cosine = 1.0_f64;
        for row in rows {
            let task = serde_json::from_value(row["task"].clone()).unwrap();
            let actual = cache
                .encode(
                    &dir,
                    row["text"].as_str().unwrap(),
                    row.get("title").and_then(|v| v.as_str()),
                    task,
                )
                .unwrap();
            let expected: Vec<f32> = serde_json::from_value(row["vector"].clone()).unwrap();
            let cosine = parity_cosine(&actual, &expected).unwrap();
            assert!(cosine >= 0.9999, "parity cosine {cosine}");
            minimum_cosine = minimum_cosine.min(cosine);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let snapshot = &cache.cached.as_ref().unwrap().1._snapshot;
            assert_eq!(
                fs::metadata(snapshot.path()).unwrap().permissions().mode() & 0o777,
                0o700
            );
        }
        let execution = cache.execution();
        assert_eq!(execution.loaded, Some(true));
        assert!(execution
            .registered_providers
            .as_ref()
            .unwrap()
            .contains(&"CPUExecutionProvider"));
        println!(
            "onnx_execution={}",
            serde_json::to_string(&execution).unwrap()
        );
        println!(
            "{} parity cases, minimum cosine {minimum_cosine}",
            rows.len()
        );
        let snapshot_path = cache.cached.as_ref().unwrap().1._snapshot.path().to_owned();
        assert!(snapshot_path.join("onnx/model.onnx_data").is_file());
        drop(cache);
        assert!(
            !snapshot_path.exists(),
            "cache drop must remove its snapshot"
        );
    }
}
