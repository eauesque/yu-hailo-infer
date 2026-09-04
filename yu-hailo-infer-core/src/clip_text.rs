use std::{path::Path, sync::Mutex};

use ndarray::Array2;
use ort::{session::Session, value::Value};
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams,
    TruncationStrategy,
};

use crate::{clip::validate_and_normalize, InferError};

const MAX_SEQUENCE_LENGTH: usize = 77;

/// CPU ONNX Runtime encoder for Xenova's CLIP ViT-B/16 text model.
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
            session: Mutex::new(Self::build_session(&model_dir.join("text_model.onnx"))?),
            tokenizer,
        })
    }

    fn build_session(model_path: &Path) -> Result<Session, InferError> {
        // ORT uses CPUExecutionProvider by default when no provider list is supplied.
        Ok(Session::builder()?.commit_from_file(model_path)?)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<f32>, InferError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| InferError::Tokenizer(error.to_string()))?;
        let input_ids = Array2::from_shape_vec(
            (1, MAX_SEQUENCE_LENGTH),
            encoding.get_ids().iter().map(|&id| id as i64).collect(),
        )
        .map_err(|error| InferError::InvalidModelOutput(error.to_string()))?;
        let attention_mask = Array2::from_shape_vec(
            (1, MAX_SEQUENCE_LENGTH),
            encoding
                .get_attention_mask()
                .iter()
                .map(|&mask| mask as i64)
                .collect(),
        )
        .map_err(|error| InferError::InvalidModelOutput(error.to_string()))?;

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "input_ids" => Value::from_array(input_ids)?,
            "attention_mask" => Value::from_array(attention_mask)?,
        ])?;
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
