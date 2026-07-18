use super::{
    ffi::HAILO_FORMAT_TYPE_UINT8, load_yolo_metadata, run_yolo_once, HailoRtError, HailoRtResult,
    VStreamInfo,
};

const CLIP_EMBEDDING_DIMENSION: usize = 512;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ClipImageMetadata {
    pub(crate) input: VStreamInfo,
    pub(crate) output: VStreamInfo,
}

/// Loads and validates the single-input/single-output CLIP image HEF metadata.
///
/// The underlying C++ shim is intentionally shared with YOLO: it performs
/// generic HEF synchronous inference and contains no YOLO-specific behavior.
pub(crate) fn load_clip_image_metadata(hef_path: &str) -> HailoRtResult<ClipImageMetadata> {
    let metadata = load_yolo_metadata(hef_path)?;
    let input = metadata
        .inputs
        .into_iter()
        .next()
        .ok_or(HailoRtError::InvalidMetadata("missing CLIP image input"))?;
    if metadata.outputs.len() != 1 {
        return Err(HailoRtError::InvalidMetadata(
            "CLIP image HEF must expose exactly one output vstream",
        ));
    }
    let output = metadata
        .outputs
        .into_iter()
        .next()
        .expect("checked output length");
    if output.format_type != HAILO_FORMAT_TYPE_UINT8 {
        return Err(HailoRtError::InvalidMetadata(
            "CLIP image output must use uint8 quantization",
        ));
    }
    if output.frame_size != CLIP_EMBEDDING_DIMENSION {
        return Err(HailoRtError::InvalidMetadata(
            "CLIP image output must contain exactly 512 values",
        ));
    }
    Ok(ClipImageMetadata { input, output })
}

/// Runs a CLIP image HEF and converts its uint8 embedding to a normalized f32 vector.
pub(crate) fn run_clip_image_once(hef_path: &str, input: &[u8]) -> HailoRtResult<Vec<f32>> {
    let metadata = load_clip_image_metadata(hef_path)?;
    if input.len() != metadata.input.frame_size {
        return Err(HailoRtError::InvalidMetadata("input buffer size mismatch"));
    }
    let result = run_yolo_once(hef_path, input)?;
    let output = result
        .outputs
        .into_iter()
        .next()
        .ok_or(HailoRtError::InvalidMetadata("missing CLIP image output"))?;
    if output.data.len() != CLIP_EMBEDDING_DIMENSION {
        return Err(HailoRtError::InvalidMetadata(
            "CLIP image output must contain exactly 512 values",
        ));
    }
    Ok(dequantize_and_normalize(
        &output.data,
        output.info.quant.scale,
        output.info.quant.zero_point,
    ))
}

fn dequantize_and_normalize(input: &[u8], scale: f32, zero_point: f32) -> Vec<f32> {
    let mut vector = input
        .iter()
        .map(|&value| (value as f32 - zero_point) * scale)
        .collect::<Vec<_>>();
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm >= 1e-12 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantize_and_normalize_matches_python_clip_logic() {
        let vector = dequantize_and_normalize(&[10, 20], 0.5, 10.0);
        assert!((vector[0] - 0.0).abs() < 1e-6);
        assert!((vector[1] - 1.0).abs() < 1e-6);

        let vector = dequantize_and_normalize(&[10, 10], 0.5, 10.0);
        assert_eq!(vector, vec![0.0, 0.0]);
    }

    #[test]
    #[ignore = "requires /dev/hailo0 and HAILO_CLIP_HEF"]
    fn smoke_clip_image_one_zero_frame_runs() {
        let hef = std::env::var("HAILO_CLIP_HEF").expect("HAILO_CLIP_HEF");
        let metadata = load_clip_image_metadata(&hef).expect("metadata");
        let vector = run_clip_image_once(&hef, &vec![0u8; metadata.input.frame_size])
            .expect("one-frame inference");
        assert_eq!(vector.len(), CLIP_EMBEDDING_DIMENSION);
    }
}
