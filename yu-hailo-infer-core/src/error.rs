#[derive(Debug, thiserror::Error)]
pub enum InferError {
    #[error("model not downloaded: {0}")]
    ModelNotDownloaded(std::path::PathBuf),
    #[error("ort error: {0}")]
    Ort(#[from] ort::Error),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("invalid model output: {0}")]
    InvalidModelOutput(String),
}
