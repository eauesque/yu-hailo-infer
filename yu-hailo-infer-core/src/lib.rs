pub mod clip_text;
pub mod engine;
pub mod error;
pub mod profile;
pub mod tags;
pub mod yolo_labels;
pub mod yolo_postprocess;

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

/// Whether the files a profile actually names are present.
///
/// The WD-shaped check above would reject camie, whose weights are
/// `camie-tagger-v2.onnx` and whose vocabulary is a JSON — and would accept a
/// directory holding WD's files while the request asks for something else.
pub fn is_profile_model_downloaded(
    cache_dir: &std::path::Path,
    model_id: &str,
    spec: &WdProfileSpec,
) -> bool {
    let base = cache_dir.join(model_id);
    let tag_file = match &spec.tag_source {
        profile::TagSourceSpec::Csv { file, .. }
        | profile::TagSourceSpec::JsonDict { file, .. } => file,
    };
    base.join(&spec.model_file).exists() && base.join(tag_file).exists()
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

        assert!(!is_profile_model_downloaded(dir.path(), "camie", &spec));
        fs::write(model_dir.join("camie-tagger-v2.onnx"), b"dummy").unwrap();
        assert!(!is_profile_model_downloaded(dir.path(), "camie", &spec));
        fs::write(model_dir.join("camie-tagger-v2-metadata.json"), b"{}").unwrap();
        assert!(is_profile_model_downloaded(dir.path(), "camie", &spec));
    }

    #[test]
    fn the_wd_shaped_check_would_have_rejected_a_ready_camie_directory() {
        // Guards the reason is_profile_model_downloaded exists at all.
        let dir = TempDir::new().unwrap();
        let model_dir = dir.path().join("camie");
        fs::create_dir(&model_dir).unwrap();
        fs::write(model_dir.join("camie-tagger-v2.onnx"), b"dummy").unwrap();
        fs::write(model_dir.join("camie-tagger-v2-metadata.json"), b"{}").unwrap();
        assert!(is_profile_model_downloaded(
            dir.path(),
            "camie",
            &camie_spec()
        ));
        assert!(!is_model_downloaded(dir.path(), "camie"));
    }
}
