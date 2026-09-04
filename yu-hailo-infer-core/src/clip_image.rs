use std::{path::Path, sync::Mutex};

use ndarray::Array4;
use ort::{session::Session, value::Value};

use crate::{clip::validate_and_normalize, InferError};

const IMAGE_WIDTH: u32 = 224;
const IMAGE_HEIGHT: u32 = 224;
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
#[allow(clippy::excessive_precision)] // CLIP's published preprocessing constants.
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// CPU ONNX Runtime encoder for Xenova's CLIP ViT-B/16 vision model.
pub struct ClipImageEncoder {
    session: Mutex<Session>,
}

impl ClipImageEncoder {
    pub fn new(model_dir: &Path) -> Result<Self, InferError> {
        if !is_clip_image_model_downloaded(model_dir) {
            return Err(InferError::ModelNotDownloaded(model_dir.to_owned()));
        }
        Ok(Self {
            session: Mutex::new(
                Session::builder()?.commit_from_file(model_dir.join("vision_model.onnx"))?,
            ),
        })
    }

    pub fn encode_rgb(&self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<f32>, InferError> {
        let input = preprocess_rgb(rgb, width, height)?;
        let mut session = self.session.lock().unwrap();
        let input_name = session.inputs()[0].name().to_owned();
        let outputs = session.run(ort::inputs![input_name => Value::from_array(input)?])?;
        let (_, output) = outputs[0].try_extract_tensor::<f32>()?;
        let mut vector = output.to_vec();
        validate_and_normalize(&mut vector)?;
        Ok(vector)
    }
}

/// Converts HWC RGB8 pixels to normalized NCHW CLIP input.
fn preprocess_rgb(rgb: &[u8], width: u32, height: u32) -> Result<Array4<f32>, InferError> {
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| {
            InferError::InvalidModelOutput("CLIP image dimensions overflow".to_string())
        })?;
    if width != IMAGE_WIDTH || height != IMAGE_HEIGHT || rgb.len() != expected_len {
        return Err(InferError::InvalidModelOutput(format!(
            "CLIP image input must be {IMAGE_WIDTH}x{IMAGE_HEIGHT} RGB8 ({}) bytes, got {width}x{height} RGB8 ({} bytes)",
            IMAGE_WIDTH as usize * IMAGE_HEIGHT as usize * 3,
            rgb.len()
        )));
    }
    let mut input = Array4::zeros((1, 3, height as usize, width as usize));
    for y in 0..height as usize {
        for x in 0..width as usize {
            let offset = (y * width as usize + x) * 3;
            for channel in 0..3 {
                input[[0, channel, y, x]] =
                    (rgb[offset + channel] as f32 / 255.0 - MEAN[channel]) / STD[channel];
            }
        }
    }
    Ok(input)
}

/// Returns whether `model_dir` contains the vision model required by the image encoder.
pub fn is_clip_image_model_downloaded(model_dir: &Path) -> bool {
    model_dir.join("vision_model.onnx").is_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn normalized(value: u8, channel: usize) -> f32 {
        (value as f32 / 255.0 - MEAN[channel]) / STD[channel]
    }

    #[test]
    fn model_download_check_requires_vision_model() {
        let dir = TempDir::new().unwrap();
        assert!(!is_clip_image_model_downloaded(dir.path()));
        std::fs::write(dir.path().join("vision_model.onnx"), b"model").unwrap();
        assert!(is_clip_image_model_downloaded(dir.path()));
    }

    #[test]
    fn preprocess_rgb_normalizes_and_transposes_distinct_pixels_and_channels() {
        let mut rgb = vec![0; IMAGE_WIDTH as usize * IMAGE_HEIGHT as usize * 3];
        for y in 0..IMAGE_HEIGHT as usize {
            for x in 0..IMAGE_WIDTH as usize {
                let offset = (y * IMAGE_WIDTH as usize + x) * 3;
                rgb[offset] = ((x + 3 * y + 11) % 251) as u8;
                rgb[offset + 1] = ((5 * x + 7 * y + 37) % 251) as u8;
                rgb[offset + 2] = ((11 * x + 13 * y + 71) % 251) as u8;
            }
        }
        let input = preprocess_rgb(&rgb, IMAGE_WIDTH, IMAGE_HEIGHT).unwrap();
        for (channel, y, x) in [
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (2, 17, 23),
        ] {
            let value = rgb[(y * IMAGE_WIDTH as usize + x) * 3 + channel];
            assert!((input[[0, channel, y, x]] - normalized(value, channel)).abs() < 1e-6);
        }
    }

    #[test]
    fn preprocess_rgb_uses_clip_published_normalization_constants() {
        let mut rgb = vec![0; IMAGE_WIDTH as usize * IMAGE_HEIGHT as usize * 3];
        rgb[..3].copy_from_slice(&[255, 255, 255]);
        let input = preprocess_rgb(&rgb, IMAGE_WIDTH, IMAGE_HEIGHT).unwrap();
        assert!((input[[0, 0, 0, 0]] - 1.9303361).abs() < 1e-6);
        assert!((input[[0, 1, 0, 0]] - 2.0748837).abs() < 1e-6);
        assert!((input[[0, 2, 0, 0]] - 2.145897).abs() < 1e-6);
    }

    #[test]
    fn preprocess_rgb_rejects_wrong_length() {
        let rgb = vec![0; IMAGE_WIDTH as usize * IMAGE_HEIGHT as usize * 3 - 1];
        assert!(preprocess_rgb(&rgb, IMAGE_WIDTH, IMAGE_HEIGHT).is_err());
    }
}
