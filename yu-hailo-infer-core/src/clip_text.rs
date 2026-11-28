use std::{path::Path, sync::Mutex};

use ndarray::Array2;
use ort::{session::Session, value::Value};
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams,
    TruncationStrategy,
};

use crate::{clip::validate_and_normalize, ep::build_session, InferError};

const MAX_SEQUENCE_LENGTH: usize = 77;

/// ONNX Runtime encoder for Xenova's CLIP ViT-B/16 text model.
pub struct ClipTextEncoder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl ClipTextEncoder {
    pub fn new(model_dir: &Path) -> Result<Self, InferError> {
        if !is_clip_text_model_downloaded(model_dir) {
            return Err(InferError::ModelNotDownloaded(model_dir.to_owned()));
        }

        let mut tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|error| InferError::Tokenizer(error.to_string()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_SEQUENCE_LENGTH,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: tokenizers::TruncationDirection::Right,
            }))
            .map_err(|error| InferError::Tokenizer(error.to_string()))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(MAX_SEQUENCE_LENGTH),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "<|endoftext|>".to_string(),
        }));

        Ok(Self {
            session: Mutex::new(build_session(&model_dir.join("text_model.onnx"))?),
            tokenizer,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<f32>, InferError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| InferError::Tokenizer(error.to_string()))?;
        let mut session = self.session.lock().unwrap();
        let mut inputs = Vec::with_capacity(session.inputs().len());
        for input in session.inputs() {
            let name = input.name();
            let tokens = match name {
                "input_ids" => encoding.get_ids(),
                "attention_mask" => encoding.get_attention_mask(),
                _ => {
                    return Err(InferError::InvalidModelOutput(format!(
                        "unsupported CLIP text input: {name}"
                    )));
                }
            };
            let tensor = Array2::from_shape_vec(
                (1, MAX_SEQUENCE_LENGTH),
                tokens.iter().map(|&token| i64::from(token)).collect(),
            )
            .map_err(|error| InferError::InvalidModelOutput(error.to_string()))?;
            inputs.push((name.to_owned(), Value::from_array(tensor)?));
        }
        if !inputs.iter().any(|(name, _)| name == "input_ids") {
            return Err(InferError::InvalidModelOutput(
                "missing CLIP text input: input_ids".into(),
            ));
        }
        let outputs = session.run(inputs)?;
        let (_, output) = outputs[0].try_extract_tensor::<f32>()?;
        let mut vector = output.to_vec();
        validate_and_normalize(&mut vector)?;
        Ok(vector)
    }
}

/// Returns whether `model_dir` contains both files required by the text encoder.
pub fn is_clip_text_model_downloaded(model_dir: &Path) -> bool {
    model_dir.join("text_model.onnx").is_file() && model_dir.join("tokenizer.json").is_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn fixture_encoder(model: &[u8]) -> ClipTextEncoder {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("text_model.onnx"), model).unwrap();
        let tokenizer = Tokenizer::new(
            tokenizers::models::wordlevel::WordLevel::builder()
                .vocab([("fixture".to_owned(), 3)].into_iter().collect())
                .unk_token("fixture".to_owned())
                .build()
                .unwrap(),
        );
        tokenizer
            .save(dir.path().join("tokenizer.json"), false)
            .unwrap();
        ClipTextEncoder::new(dir.path()).unwrap()
    }

    #[test]
    fn encodes_with_input_ids_only() {
        let encoder = fixture_encoder(include_bytes!("../tests/fixtures/clip_text_ids.onnx"));
        let mut expected = vec![0.0; 512];
        expected[0] = 1.0;
        assert_eq!(encoder.encode("fixture").unwrap(), expected);
    }

    #[test]
    fn encodes_with_attention_mask() {
        let encoder = fixture_encoder(include_bytes!("../tests/fixtures/clip_text_mask.onnx"));
        let vector = encoder.encode("fixture").unwrap();
        let norm = 10.0_f32.sqrt();
        assert!((vector[0] - 3.0 / norm).abs() < 1e-6);
        assert!((vector[MAX_SEQUENCE_LENGTH] - 1.0 / norm).abs() < 1e-6);
    }

    #[test]
    fn rejects_unknown_input() {
        let encoder = fixture_encoder(include_bytes!("../tests/fixtures/clip_text_unknown.onnx"));
        assert!(matches!(
            encoder.encode("fixture").unwrap_err(),
            InferError::InvalidModelOutput(message) if message.contains("extra_input")
        ));
    }

    #[test]
    fn rejects_missing_input_ids() {
        let encoder = fixture_encoder(include_bytes!(
            "../tests/fixtures/clip_text_missing_ids.onnx"
        ));
        assert!(matches!(
            encoder.encode("fixture").unwrap_err(),
            InferError::InvalidModelOutput(message) if message.contains("input_ids")
        ));
    }

    #[test]
    fn model_download_check_requires_both_files() {
        let dir = TempDir::new().unwrap();
        assert!(!is_clip_text_model_downloaded(dir.path()));
        std::fs::write(dir.path().join("text_model.onnx"), b"model").unwrap();
        assert!(!is_clip_text_model_downloaded(dir.path()));
        std::fs::write(dir.path().join("tokenizer.json"), b"tokenizer").unwrap();
        assert!(is_clip_text_model_downloaded(dir.path()));
    }
}
