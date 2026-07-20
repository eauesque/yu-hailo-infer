pub mod clip_text;
pub mod engine;
pub mod error;
pub mod tags;
pub mod yolo_labels;
pub mod yolo_postprocess;

pub use clip_text::{is_clip_text_model_downloaded, ClipTextEncoder};
pub use engine::{TagPrediction, TagResult, WdInferEngine};
pub use error::InferError;

pub fn is_model_downloaded(cache_dir: &std::path::Path, model_id: &str) -> bool {
    let base = cache_dir.join(model_id);
    base.join("model.onnx").exists() && base.join("selected_tags.csv").exists()
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
}
