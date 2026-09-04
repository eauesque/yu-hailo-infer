mod clip;
pub mod clip_image;
pub mod clip_text;
pub mod engine;
pub mod error;
pub mod profile;
pub mod tags;
pub mod yolo_labels;
pub mod yolo_postprocess;

pub use clip_image::{is_clip_image_model_downloaded, ClipImageEncoder};
pub use clip_text::{is_clip_text_model_downloaded, ClipTextEncoder};
pub use engine::{builtin_wd_profile, TagPrediction, TagResult, WdInferEngine};
pub use error::InferError;
pub use profile::WdProfileSpec;

/// Whether the default WD layout is present: the two files every v1 WD
/// profile declares.
pub fn is_model_downloaded(cache_dir: &std::path::Path, model_id: &str) -> bool {
    let base = cache_dir.join(model_id);
    base.join("model.onnx").exists() && base.join("selected_tags.csv").exists()
}

/// Whether the files a profile actually names are present in `model_dir`.
///
/// The WD-shaped check above would reject camie, whose weights are
/// `camie-tagger-v2.onnx` and whose vocabulary is a JSON — and would accept a
/// directory holding WD's files while the request asks for something else.
pub fn is_profile_model_ready(model_dir: &std::path::Path, spec: &WdProfileSpec) -> bool {
    let tag_file = match &spec.tag_source {
        profile::TagSourceSpec::Csv { file, .. }
        | profile::TagSourceSpec::JsonDict { file, .. } => file,
    };
    model_dir.join(&spec.model_file).exists() && model_dir.join(tag_file).exists()
}

/// Resolve a model directory, honouring an optional variant subdirectory.
///
/// Mirrors `validate_hf_subdir` on the Python side: up to four segments of
/// `[A-Za-z0-9._-]`, with `.` and `..` rejected. Returns `None` for anything
/// else rather than joining it, because the result is a filesystem path built
/// from a value that arrived over the wire.
pub fn resolve_model_dir(
    cache_dir: &std::path::Path,
    model_id: &str,
    subdir: Option<&str>,
) -> Option<std::path::PathBuf> {
    let mut dir = cache_dir.join(model_id);
    let Some(subdir) = subdir.filter(|s| !s.is_empty()) else {
        return Some(dir);
    };
    let segments: Vec<&str> = subdir.split('/').collect();
    if segments.len() > 4 {
        return None;
    }
    for segment in segments {
        if segment.is_empty()
            || segment == "."
            || segment == ".."
            || !segment
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
        {
            return None;
        }
        dir = dir.join(segment);
    }
    Some(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn is_model_downloaded_returns_false_when_missing() {
        let dir = TempDir::new().unwrap();
        assert!(!is_model_downloaded(dir.path(), "wd_vit_v3"));
    }

    #[test]
    fn is_model_downloaded_returns_true_when_both_files_present() {
        let dir = TempDir::new().unwrap();
        let model_dir = dir.path().join("wd_vit_v3");
        fs::create_dir(&model_dir).unwrap();
        fs::write(model_dir.join("model.onnx"), b"dummy").unwrap();
        fs::write(
            model_dir.join("selected_tags.csv"),
            b"tag_id,name,category\n0,a,0",
        )
        .unwrap();
        assert!(is_model_downloaded(dir.path(), "wd_vit_v3"));
    }

    fn camie_spec() -> WdProfileSpec {
        serde_json::from_value(serde_json::json!({
            "model_file": "camie-tagger-v2.onnx",
            "preprocess_spec": {"input_size": 512, "layout": "NCHW"},
            "tag_source": {
                "type": "json_dict",
                "file": "camie-tagger-v2-metadata.json",
                "idx_to_tag_key": "idx_to_tag",
                "tag_to_category_key": "tag_to_category"
            }
        }))
        .unwrap()
    }

    #[test]
    fn profile_readiness_checks_the_files_the_profile_names() {
        let dir = TempDir::new().unwrap();
        let model_dir = dir.path().join("camie");
        fs::create_dir(&model_dir).unwrap();
        let spec = camie_spec();

        assert!(!is_profile_model_ready(&model_dir, &spec));
        fs::write(model_dir.join("camie-tagger-v2.onnx"), b"dummy").unwrap();
        assert!(!is_profile_model_ready(&model_dir, &spec));
        fs::write(model_dir.join("camie-tagger-v2-metadata.json"), b"{}").unwrap();
        assert!(is_profile_model_ready(&model_dir, &spec));
    }

    #[test]
    fn the_wd_shaped_check_would_have_rejected_a_ready_camie_directory() {
        // Guards the reason is_profile_model_ready exists at all.
        let dir = TempDir::new().unwrap();
        let model_dir = dir.path().join("camie");
        fs::create_dir(&model_dir).unwrap();
        fs::write(model_dir.join("camie-tagger-v2.onnx"), b"dummy").unwrap();
        fs::write(model_dir.join("camie-tagger-v2-metadata.json"), b"{}").unwrap();
        assert!(is_profile_model_ready(&model_dir, &camie_spec()));
        assert!(!is_model_downloaded(dir.path(), "camie"));
    }

    #[test]
    fn a_variant_subdirectory_is_joined_when_it_is_a_plain_relative_path() {
        let root = std::path::Path::new("/cache");
        assert_eq!(
            resolve_model_dir(root, "repo", None),
            Some(root.join("repo"))
        );
        assert_eq!(
            resolve_model_dir(root, "repo", Some("")),
            Some(root.join("repo")),
            "an empty subdir means no subdir, as on the Python side"
        );
        assert_eq!(
            resolve_model_dir(root, "repo", Some("v1.1/fp16")),
            Some(root.join("repo").join("v1.1").join("fp16"))
        );
    }

    #[test]
    fn a_subdirectory_that_escapes_the_cache_is_refused() {
        let root = std::path::Path::new("/cache");
        for bad in [
            "..",
            "../other",
            "v1/../../etc",
            "/abs",
            "a/b/c/d/e",
            "a b",
            "a\\b",
            ".",
        ] {
            assert_eq!(
                resolve_model_dir(root, "repo", Some(bad)),
                None,
                "subdir {bad:?} must be refused, not joined"
            );
        }
    }
}
