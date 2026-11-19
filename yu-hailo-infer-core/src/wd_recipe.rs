//! Shared WD pixels and tag selection; neither helper opens an ONNX session.
use std::{collections::HashMap, path::Path};

use crate::{
    engine::{
        letterbox, validate_wd_image_dimensions, MAX_WD_DECODED_IMAGE_BYTES,
        MAX_WD_DECODED_IMAGE_DIMENSION,
    },
    profile::PreprocessSpec,
    tags::TagMeta,
    InferError, TagPrediction, TagResult, WdProfileSpec,
};

pub(crate) fn canvas(path: &Path, p: &PreprocessSpec) -> Result<image::RgbImage, InferError> {
    let mut reader = image::ImageReader::open(path)?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_WD_DECODED_IMAGE_DIMENSION);
    limits.max_image_height = Some(MAX_WD_DECODED_IMAGE_DIMENSION);
    limits.max_alloc = Some(MAX_WD_DECODED_IMAGE_BYTES);
    reader.limits(limits);
    let img = reader.decode()?;
    validate_wd_image_dimensions(img.width(), img.height())?;
    letterbox(&img, p.input_size, p.pad_color)
}

/// Calibration uses the built-in resize-first Lanczos3 recipe, followed by BGR NHWC.
pub fn preprocess_u8(path: &Path) -> Result<Vec<u8>, InferError> {
    let spec = crate::builtin_wd_profile("model.onnx", 0.35, 0.85);
    let mut pixels = canvas(path, &spec.preprocess_spec)?.into_raw();
    for pixel in pixels.as_chunks_mut::<3>().0 {
        pixel.swap(0, 2);
    }
    Ok(pixels)
}

pub fn build_result(
    image_path: &Path,
    scores: &[f32],
    tag_meta: &TagMeta,
    model_id: &str,
    spec: &WdProfileSpec,
) -> TagResult {
    let mut thresholds: HashMap<&str, f32> = HashMap::new();
    let rating = {
        let mut best: Option<(f32, &str)> = None;
        for &idx in &tag_meta.rating_indices {
            if let (Some(&s), Some((name, _))) = (scores.get(idx), tag_meta.tags.get(idx)) {
                // Python seeds rating_max at 0.0; zero leaves the default label.
                if s > best.map_or(0.0, |(b, _)| b) {
                    best = Some((s, name.as_str()));
                }
            }
        }
        best.map(|(_, name)| name.to_string())
            .unwrap_or_else(|| "general".to_string())
    };
    let mut preds = Vec::new();
    for (i, (name, cat)) in tag_meta.tags.iter().enumerate() {
        let conf = scores[i];
        if cat == "rating" {
            continue;
        }
        let thr = *thresholds
            .entry(cat.as_str())
            .or_insert_with(|| spec.threshold_for(cat));
        if conf < thr {
            continue;
        }
        preds.push(TagPrediction {
            tag: name.clone(),
            confidence: conf,
            category: cat.clone(),
        });
    }
    preds.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    TagResult {
        tags: preds,
        rating,
        path: image_path.to_string_lossy().into_owned(),
        model_id: model_id.to_string(),
    }
}
