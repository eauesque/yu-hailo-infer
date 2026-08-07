use std::{path::Path, sync::Mutex};

use image::imageops::FilterType;
use ndarray::Array4;
use ort::{session::Session, value::Value};
use serde::{Deserialize, Serialize};

use crate::{
    tags::{load_tags, TagMeta},
    InferError,
};

/// WD inputs are reduced to the model's fixed-size canvas, so decoding images
/// larger than this consumes memory without improving inference quality.
const MAX_WD_DECODED_IMAGE_DIMENSION: u32 = 4096;
const MAX_WD_DECODED_IMAGE_BYTES: u64 = 64 * 1024 * 1024;
const WD_DECODED_BYTES_PER_PIXEL: u64 = 4;

fn validate_wd_image_dimensions(width: u32, height: u32) -> Result<(), InferError> {
    if width == 0
        || height == 0
        || width > MAX_WD_DECODED_IMAGE_DIMENSION
        || height > MAX_WD_DECODED_IMAGE_DIMENSION
    {
        return Err(InferError::InvalidModelOutput(format!(
            "WD image dimensions {width}x{height} exceed the {MAX_WD_DECODED_IMAGE_DIMENSION}px limit"
        )));
    }
    let decoded_bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(WD_DECODED_BYTES_PER_PIXEL))
        .ok_or_else(|| {
            InferError::InvalidModelOutput(
                "WD image dimensions overflow allocation accounting".to_string(),
            )
        })?;
    if decoded_bytes > MAX_WD_DECODED_IMAGE_BYTES {
        return Err(InferError::InvalidModelOutput(format!(
            "WD image requires {decoded_bytes} decoded bytes, exceeding the {MAX_WD_DECODED_IMAGE_BYTES}-byte limit"
        )));
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TagPrediction {
    pub tag: String,
    pub confidence: f32,
    pub category: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TagResult {
    pub tags: Vec<TagPrediction>,
    pub rating: String,
    pub path: String,
    pub model_id: String,
}

pub struct WdInferEngine {
    session: Mutex<Session>,
    tag_meta: TagMeta,
    model_id: String,
    input_size: u32,
}

impl WdInferEngine {
    pub fn new(model_dir: &Path) -> Result<Self, InferError> {
        if !model_dir.join("model.onnx").exists() {
            return Err(InferError::ModelNotDownloaded(model_dir.to_owned()));
        }
        let session = Self::build_session(&model_dir.join("model.onnx"))?;
        let tag_meta = load_tags(model_dir)?;
        let input_size = load_input_size(model_dir).unwrap_or(448);
        let model_id = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        Ok(Self {
            session: Mutex::new(session),
            tag_meta,
            model_id,
            input_size,
        })
    }

    /// Build an ORT session, trying GPU EPs in priority order then falling back to CPU.
    ///
    /// Compile with `--features cuda` or `--features rocm` to enable GPU paths.
    /// GPU EPs fail gracefully at runtime if the hardware/driver is absent.
    fn build_session(model_path: &Path) -> Result<Session, InferError> {
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        {
            let mut eps: Vec<ort::execution_providers::ExecutionProviderDispatch> = Vec::new();
            #[cfg(feature = "cuda")]
            {
                tracing::info!("Requesting CUDA execution provider");
                eps.push(ort::ep::CUDA::default().build());
            }
            #[cfg(feature = "rocm")]
            {
                tracing::info!("Requesting ROCm execution provider");
                eps.push(ort::ep::ROCm::default().build());
            }
            // CPU must be explicit — with_execution_providers replaces the default EP list
            // and does not automatically append CPU. ORT tries EPs in order; CPU is the fallback.
            eps.push(ort::ep::CPU::default().build());
            return Ok(Session::builder()?
                .with_execution_providers(eps)?
                .commit_from_file(model_path)?);
        }

        #[allow(unreachable_code)]
        Ok(Session::builder()?.commit_from_file(model_path)?)
    }

    pub fn run(
        &self,
        image_path: &Path,
        general_thr: f32,
        character_thr: f32,
    ) -> Result<TagResult, InferError> {
        let tensor = self.preprocess(image_path)?;
        let scores: Vec<f32> = {
            let mut guard = self.session.lock().unwrap();
            let input_name = guard.inputs()[0].name().to_string();
            let session_inputs = ort::inputs![
                input_name => Value::from_array(tensor)?,
            ];
            let outputs = guard.run(session_inputs)?;
            let (_, data) = outputs[0].try_extract_tensor::<f32>()?;
            data.to_vec()
        };

        let rating = {
            let mut best: Option<(f32, String)> = None;
            for &idx in &self.tag_meta.rating_indices {
                if let (Some(&s), Some((name, _))) = (scores.get(idx), self.tag_meta.tags.get(idx))
                {
                    if best.as_ref().is_none_or(|(b, _)| s > *b) {
                        best = Some((s, name.clone()));
                    }
                }
            }
            best.map(|(_, name)| name)
                .unwrap_or_else(|| "general".to_string())
        };

        let mut preds: Vec<TagPrediction> = self
            .tag_meta
            .tags
            .iter()
            .enumerate()
            .filter_map(|(i, (name, cat))| {
                let conf = *scores.get(i)?;
                if cat == "rating" {
                    return None;
                }
                let thr = if cat == "character" {
                    character_thr
                } else {
                    general_thr
                };
                if conf < thr {
                    return None;
                }
                Some(TagPrediction {
                    tag: name.clone(),
                    confidence: conf,
                    category: cat.clone(),
                })
            })
            .collect();
        preds.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(TagResult {
            tags: preds,
            rating,
            path: image_path.to_string_lossy().into_owned(),
            model_id: self.model_id.clone(),
        })
    }

    fn preprocess(&self, path: &Path) -> Result<Array4<f32>, InferError> {
        let mut reader = image::ImageReader::open(path)?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(MAX_WD_DECODED_IMAGE_DIMENSION);
        limits.max_image_height = Some(MAX_WD_DECODED_IMAGE_DIMENSION);
        limits.max_alloc = Some(MAX_WD_DECODED_IMAGE_BYTES);
        reader.limits(limits);
        let img = reader.decode()?;
        validate_wd_image_dimensions(img.width(), img.height())?;

        // Resize first, then letterbox directly into the fixed model canvas.
        // This avoids allocating a square based on attacker-controlled source
        // dimensions before reducing the image to the model input size.
        let resized = img
            .resize(self.input_size, self.input_size, FilterType::Lanczos3)
            .to_rgb8();
        let mut canvas = image::RgbImage::from_pixel(
            self.input_size,
            self.input_size,
            image::Rgb([255, 255, 255]),
        );
        let x = (self.input_size - resized.width()) / 2;
        let y = (self.input_size - resized.height()) / 2;
        image::imageops::overlay(&mut canvas, &resized, x as i64, y as i64);

        let size = self.input_size as usize;
        let mut arr = Array4::<f32>::zeros((1, size, size, 3));
        for y in 0..size {
            for x in 0..size {
                let p = canvas.get_pixel(x as u32, y as u32);
                arr[[0, y, x, 0]] = p[2] as f32; // B
                arr[[0, y, x, 1]] = p[1] as f32; // G
                arr[[0, y, x, 2]] = p[0] as f32; // R
            }
        }
        Ok(arr)
    }
}

fn load_input_size(model_dir: &Path) -> Option<u32> {
    let json_path = std::fs::read_dir(model_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))?
        .path();
    let json: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(json_path).ok()?).ok()?;
    json.get("input_size")?.as_u64().map(|n| n as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wd_image_dimension_limits_reject_oversized_and_overflowing_inputs() {
        assert!(validate_wd_image_dimensions(4096, 4096).is_ok());
        assert!(validate_wd_image_dimensions(4097, 1).is_err());
        assert!(validate_wd_image_dimensions(u32::MAX, u32::MAX).is_err());
    }
}
