use std::{collections::HashMap, path::Path, sync::Mutex};

use image::imageops::FilterType;
use ndarray::{Array, ArrayD};
use ort::{session::Session, value::Value};
use serde::{Deserialize, Serialize};

use crate::{
    profile::WdProfileSpec,
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

/// The recipe every v1 WD profile spells out. Kept here so a request that
/// carries no profile takes the same code path as one that does, instead of a
/// parallel WD-only branch that drifts.
pub fn builtin_wd_profile(model_file: &str, general_thr: f32, character_thr: f32) -> WdProfileSpec {
    let json = serde_json::json!({
        "model_file": model_file,
        "preprocess_spec": {
            "input_size": 448,
            "resize_strategy": "longest_side_pad",
            "pad_color": [255, 255, 255],
            "channel_order": "BGR",
            "scale": 1.0,
            "mean": null,
            "std": null,
            "layout": "NHWC"
        },
        "tag_source": {
            "type": "csv",
            "file": "selected_tags.csv",
            "delimiter": ",",
            "name_col": "name",
            "category_col": "category",
            "category_map": {"0": "general", "4": "character", "9": "rating"}
        },
        "default_thresholds": {
            "general": general_thr,
            "character": character_thr,
            "rating": 0.0
        },
        "supports_categories": ["general", "character", "rating"]
    });
    serde_json::from_value(json).expect("builtin WD profile is well formed")
}

pub struct WdInferEngine {
    session: Mutex<Session>,
    tag_meta: TagMeta,
    model_id: String,
    spec: WdProfileSpec,
}

impl WdInferEngine {
    pub fn new(model_dir: &Path, spec: WdProfileSpec) -> Result<Self, InferError> {
        spec.validate()?;
        let model_path = model_dir.join(&spec.model_file);
        if !model_path.exists() {
            return Err(InferError::ModelNotDownloaded(model_dir.to_owned()));
        }
        let session = Self::build_session(&model_path)?;
        let tag_meta = load_tags(
            model_dir,
            &spec.tag_source,
            &spec.categories_mode,
            &spec.supports_categories,
        )?;
        let model_id = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        Ok(Self {
            session: Mutex::new(session),
            tag_meta,
            model_id,
            spec,
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

    pub fn run(&self, image_path: &Path) -> Result<TagResult, InferError> {
        let tensor = self.preprocess(image_path)?;
        let scores = self.infer(tensor)?;
        Ok(self.build_result(image_path, &scores))
    }

    fn infer(&self, tensor: ArrayD<f32>) -> Result<Vec<f32>, InferError> {
        let raw: Vec<f32> = {
            let mut guard = self.session.lock().unwrap();
            let input_name = guard.inputs()[0].name().to_string();
            let wanted = self.spec.output_spec.output_key.clone();
            // Resolve the head index before running so an unknown name is an
            // error rather than a silent fall back to the first output.
            let index = match &wanted {
                None => 0,
                Some(name) => guard
                    .outputs()
                    .iter()
                    .position(|o| o.name() == name)
                    .ok_or_else(|| {
                        let available: Vec<String> = guard
                            .outputs()
                            .iter()
                            .map(|o| o.name().to_string())
                            .collect();
                        InferError::UnsupportedProfile(format!(
                            "output_key={name:?} not among model outputs {available:?}"
                        ))
                    })?,
            };
            let session_inputs = ort::inputs![
                input_name => Value::from_array(tensor)?,
            ];
            let outputs = guard.run(session_inputs)?;
            let (_, data) = outputs[index].try_extract_tensor::<f32>()?;
            data.to_vec()
        };

        if !self.spec.output_spec.wants_sigmoid() {
            return Ok(raw);
        }
        // Computed in f64 to match the Python side, which casts before the
        // exponential.
        Ok(raw
            .into_iter()
            .map(|v| (1.0_f64 / (1.0 + (-(v as f64)).exp())) as f32)
            .collect())
    }

    fn build_result(&self, image_path: &Path, scores: &[f32]) -> TagResult {
        // Cache per-category thresholds so a 70k-tag vocabulary does not do a
        // map lookup plus two fallbacks per tag.
        let mut thresholds: HashMap<&str, f32> = HashMap::new();

        let rating = {
            let mut best: Option<(f32, &str)> = None;
            for &idx in &self.tag_meta.rating_indices {
                if let (Some(&s), Some((name, _))) = (scores.get(idx), self.tag_meta.tags.get(idx))
                {
                    // Python seeds rating_max at 0.0, so a rating tag scoring
                    // exactly 0.0 leaves the label at "general".
                    if s > best.map_or(0.0, |(b, _)| b) {
                        best = Some((s, name.as_str()));
                    }
                }
            }
            best.map(|(_, name)| name.to_string())
                .unwrap_or_else(|| "general".to_string())
        };

        let mut preds: Vec<TagPrediction> = Vec::new();
        for (i, (name, cat)) in self.tag_meta.tags.iter().enumerate() {
            let Some(&conf) = scores.get(i) else { break };
            if cat == "rating" {
                continue;
            }
            let thr = *thresholds
                .entry(cat.as_str())
                .or_insert_with(|| self.spec.threshold_for(cat));
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
            model_id: self.model_id.clone(),
        }
    }

    fn preprocess(&self, path: &Path) -> Result<ArrayD<f32>, InferError> {
        let p = &self.spec.preprocess_spec;
        let size = p.input_size;
        let mut reader = image::ImageReader::open(path)?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(MAX_WD_DECODED_IMAGE_DIMENSION);
        limits.max_image_height = Some(MAX_WD_DECODED_IMAGE_DIMENSION);
        limits.max_alloc = Some(MAX_WD_DECODED_IMAGE_BYTES);
        reader.limits(limits);
        let img = reader.decode()?;
        validate_wd_image_dimensions(img.width(), img.height())?;

        let canvas = letterbox(&img, size, p.pad_color)?;
        Ok(build_tensor(&canvas, p))
    }
}

/// Turn the letterboxed canvas into the model's input tensor.
///
/// Split out from the engine so channel order, scaling, normalisation and
/// layout can be checked without a 752 MB ONNX file — the fixtures that
/// exercise the whole pipeline happen to use RGB and NCHW, so they cannot
/// discriminate a build that ignores those fields.
fn build_tensor(canvas: &image::RgbImage, p: &crate::profile::PreprocessSpec) -> ArrayD<f32> {
    let side = p.input_size as usize;
    let bgr = p.channel_order == "BGR";
    let scale = p.scale;
    let mean = p.mean;
    let std = p.std;

    let mut hwc = Array::<f32, _>::zeros((side, side, 3));
    for y in 0..side {
        for x in 0..side {
            let px = canvas.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                // Channel reversal happens before scaling and normalization,
                // matching preprocess_image_from_spec: mean/std index the
                // post-reversal channel order.
                let source = if bgr { 2 - c } else { c };
                let mut v = f32::from(px[source]);
                if scale != 1.0 {
                    v *= scale;
                }
                if let Some(m) = mean {
                    v -= m[c];
                }
                if let Some(s) = std {
                    v /= s[c];
                }
                hwc[[y, x, c]] = v;
            }
        }
    }

    // ORT needs a contiguous buffer; the NCHW permutation is a strided view,
    // so materialise it in standard layout before handing it over.
    if p.layout == "NCHW" {
        hwc.permuted_axes([2, 0, 1])
            .insert_axis(ndarray::Axis(0))
            .as_standard_layout()
            .to_owned()
            .into_dyn()
    } else {
        hwc.insert_axis(ndarray::Axis(0)).into_dyn()
    }
}

/// Composite onto `pad_color`, scale the longest side to `size`, and centre
/// the result on a `size`×`size` canvas.
///
/// This is a transcription of `_load_and_pad` in
/// `adapters/preprocess.py`, including two details that look incidental and
/// are not:
///   * the source is composited over the pad colour **before** resizing, so a
///     transparent PNG resolves to white rather than to whatever the encoder
///     left in the RGB channels;
///   * the scaled dimensions truncate (`int(w * scale)`), they do not round.
///     Rounding shifts the paste offset by a pixel on many inputs.
fn letterbox(
    img: &image::DynamicImage,
    size: u32,
    pad_color: [u8; 3],
) -> Result<image::RgbImage, InferError> {
    let rgba = img.to_rgba8();
    let (old_w, old_h) = (rgba.width(), rgba.height());
    let mut composited = image::RgbImage::from_pixel(old_w, old_h, image::Rgb(pad_color));
    for (x, y, px) in rgba.enumerate_pixels() {
        let alpha = f32::from(px[3]) / 255.0;
        let base = composited.get_pixel_mut(x, y);
        for c in 0..3 {
            let over = f32::from(px[c]) * alpha + f32::from(base[c]) * (1.0 - alpha);
            base[c] = over.round().clamp(0.0, 255.0) as u8;
        }
    }

    let scale = f64::from(size) / f64::from(old_w.max(old_h));
    let new_w = (f64::from(old_w) * scale) as u32;
    let new_h = (f64::from(old_h) * scale) as u32;
    if new_w == 0 || new_h == 0 {
        return Err(InferError::InvalidModelOutput(format!(
            "image {old_w}x{old_h} degenerates to {new_w}x{new_h} at input_size {size}"
        )));
    }
    let resized = image::imageops::resize(&composited, new_w, new_h, FilterType::Lanczos3);

    let mut canvas = image::RgbImage::from_pixel(size, size, image::Rgb(pad_color));
    let x = (size - new_w) / 2;
    let y = (size - new_h) / 2;
    image::imageops::overlay(&mut canvas, &resized, i64::from(x), i64::from(y));
    Ok(canvas)
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

    fn solid(w: u32, h: u32, px: image::Rgba<u8>) -> image::DynamicImage {
        image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(w, h, px))
    }

    #[test]
    fn letterbox_truncates_the_scaled_side_it_does_not_round() {
        // 448 * (3/7) = 192.0 exactly for the short side of a 7x3 image
        // scaled to 448; pick a ratio whose product has a fractional part
        // above .5 so truncation and rounding disagree.
        // 100x67 -> scale 4.48 -> 67 * 4.48 = 300.16 (both agree),
        // 100x63 -> 282.24 (agree), 100x61 -> 273.28 (agree);
        // 100x59 -> 264.32; use 100x37 -> 165.76: trunc 165, round 166.
        let img = solid(100, 37, image::Rgba([10, 20, 30, 255]));
        let canvas = letterbox(&img, 448, [255, 255, 255]).unwrap();
        assert_eq!(canvas.dimensions(), (448, 448));
        // The pasted block starts at (448 - 165) / 2 = 141 with truncation,
        // and at 141 with rounding too — so assert on the block height by
        // scanning the centre column for non-pad pixels.
        let centre_x = 224;
        let painted = (0..448)
            .filter(|&y| *canvas.get_pixel(centre_x, y) != image::Rgb([255, 255, 255]))
            .count();
        assert_eq!(painted, 165, "expected trunc(37 * 4.48) = 165 painted rows");
    }

    #[test]
    fn letterbox_composites_transparency_over_the_pad_colour() {
        // A fully transparent source must resolve to the pad colour, not to
        // the RGB bytes hiding behind alpha=0. Dropping the alpha channel
        // instead would yield pure red here.
        let img = solid(64, 64, image::Rgba([255, 0, 0, 0]));
        let canvas = letterbox(&img, 64, [255, 255, 255]).unwrap();
        assert_eq!(*canvas.get_pixel(32, 32), image::Rgb([255, 255, 255]));

        let half = solid(64, 64, image::Rgba([0, 0, 0, 128]));
        let blended = letterbox(&half, 64, [255, 255, 255]).unwrap();
        let px = *blended.get_pixel(32, 32);
        // 0 * (128/255) + 255 * (1 - 128/255) = 127.0
        assert!(
            (i32::from(px[0]) - 127).abs() <= 1,
            "expected ~127, got {px:?}"
        );
    }

    #[test]
    fn letterbox_honours_a_non_white_pad_colour() {
        let img = solid(100, 10, image::Rgba([0, 0, 0, 255]));
        let canvas = letterbox(&img, 64, [7, 8, 9]).unwrap();
        assert_eq!(*canvas.get_pixel(0, 0), image::Rgb([7, 8, 9]));
    }

    #[test]
    fn letterbox_rejects_an_aspect_ratio_that_collapses_a_side() {
        let img = solid(4096, 1, image::Rgba([0, 0, 0, 255]));
        let err = letterbox(&img, 64, [255, 255, 255]).unwrap_err();
        assert!(matches!(err, InferError::InvalidModelOutput(_)), "{err:?}");
    }

    fn preprocess_spec(json: serde_json::Value) -> crate::profile::PreprocessSpec {
        serde_json::from_value(json).expect("preprocess spec deserializes")
    }

    /// A 2x2 canvas whose channels are all distinct, so any mix-up shows.
    fn probe_canvas() -> image::RgbImage {
        image::RgbImage::from_fn(2, 2, |_, _| image::Rgb([10, 20, 30]))
    }

    #[test]
    fn rgb_and_bgr_produce_different_tensors() {
        // camie is RGB and WD is BGR; a build that hardcodes either one still
        // passes every camie fixture, so this has to be checked directly.
        let rgb = build_tensor(
            &probe_canvas(),
            &preprocess_spec(serde_json::json!({"input_size": 2, "channel_order": "RGB"})),
        );
        let bgr = build_tensor(
            &probe_canvas(),
            &preprocess_spec(serde_json::json!({"input_size": 2, "channel_order": "BGR"})),
        );
        assert_eq!(
            rgb.as_slice().unwrap()[..3],
            [10.0, 20.0, 30.0],
            "RGB must keep the source channel order"
        );
        assert_eq!(
            bgr.as_slice().unwrap()[..3],
            [30.0, 20.0, 10.0],
            "BGR must reverse it"
        );
    }

    #[test]
    fn nchw_and_nhwc_produce_different_shapes_and_orders() {
        let spec =
            |layout: &str| preprocess_spec(serde_json::json!({"input_size": 2, "layout": layout}));
        let nhwc = build_tensor(&probe_canvas(), &spec("NHWC"));
        let nchw = build_tensor(&probe_canvas(), &spec("NCHW"));
        assert_eq!(nhwc.shape(), &[1, 2, 2, 3]);
        assert_eq!(nchw.shape(), &[1, 3, 2, 2]);
        // NCHW groups by channel: the first four values are all the red plane.
        assert_eq!(nchw.as_slice().unwrap()[..4], [10.0, 10.0, 10.0, 10.0]);
        assert_eq!(nhwc.as_slice().unwrap()[..4], [10.0, 20.0, 30.0, 10.0]);
    }

    #[test]
    fn scale_then_mean_then_std_are_applied_in_that_order() {
        // (10/255 - 0.485) / 0.229 for the red channel. Applying mean before
        // the scale, or std before the mean, lands somewhere else entirely.
        let tensor = build_tensor(
            &probe_canvas(),
            &preprocess_spec(serde_json::json!({
                "input_size": 2,
                "channel_order": "RGB",
                "scale": 0.00392156862745098,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "layout": "NHWC"
            })),
        );
        let expected = (10.0_f32 / 255.0 - 0.485) / 0.229;
        assert!(
            (tensor.as_slice().unwrap()[0] - expected).abs() < 1e-6,
            "got {}, expected {expected}",
            tensor.as_slice().unwrap()[0]
        );
    }

    #[test]
    fn an_absent_mean_and_std_leave_the_raw_byte_values() {
        let tensor = build_tensor(
            &probe_canvas(),
            &preprocess_spec(serde_json::json!({"input_size": 2, "channel_order": "BGR"})),
        );
        // This is the WD recipe: raw 0..255 in BGR, no normalisation.
        assert_eq!(tensor.as_slice().unwrap()[..3], [30.0, 20.0, 10.0]);
    }

    #[test]
    fn the_builtin_wd_profile_matches_the_shipped_v1_profiles() {
        let spec = builtin_wd_profile("model.onnx", 0.35, 0.85);
        spec.validate().unwrap();
        assert_eq!(spec.preprocess_spec.input_size, 448);
        assert_eq!(spec.preprocess_spec.channel_order, "BGR");
        assert_eq!(spec.preprocess_spec.layout, "NHWC");
        assert_eq!(spec.preprocess_spec.scale, 1.0);
        assert_eq!(spec.preprocess_spec.mean, None);
        assert_eq!(spec.preprocess_spec.std, None);
        assert!(!spec.output_spec.wants_sigmoid());
        assert_eq!(spec.threshold_for("general"), 0.35);
        assert_eq!(spec.threshold_for("character"), 0.85);
        assert_eq!(spec.threshold_for("rating"), 0.0);
    }
}
