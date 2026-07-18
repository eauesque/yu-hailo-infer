use std::{path::Path, sync::Mutex};

use image::{imageops::FilterType, GenericImageView};
use ndarray::Array4;
use ort::{session::Session, value::Value};
use serde::{Deserialize, Serialize};

use crate::{
    tags::{load_tags, TagMeta},
    InferError,
};

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
        let img = image::open(path)?;
        let (w, h) = img.dimensions();
        let side = w.max(h);
        let mut canvas = image::RgbImage::from_pixel(side, side, image::Rgb([255, 255, 255]));
        let x = (side - w) / 2;
        let y = (side - h) / 2;
        image::imageops::overlay(&mut canvas, &img.to_rgb8(), x as i64, y as i64);
        let resized = image::imageops::resize(
            &canvas,
            self.input_size,
            self.input_size,
            FilterType::Lanczos3,
        );

        let size = self.input_size as usize;
        let mut arr = Array4::<f32>::zeros((1, size, size, 3));
        for y in 0..size {
            for x in 0..size {
                let p = resized.get_pixel(x as u32, y as u32);
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
