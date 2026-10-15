use std::{path::Path, sync::Mutex};

use ndarray::Array4;
use ort::{session::Session, value::Value};

use crate::{
    yolo_postprocess::{decode_ultralytics_output, finalize_detections, Detection, ScaleInfo},
    InferError,
};

pub struct YoloOnnxDetector {
    session: Mutex<Session>,
    input_size: u32,
}

impl YoloOnnxDetector {
    pub fn new(model_path: &Path) -> Result<Self, InferError> {
        if !model_path.is_file() {
            return Err(InferError::ModelNotDownloaded(model_path.to_owned()));
        }
        let session = crate::ep::build_session(model_path)?;
        let shape = session
            .inputs()
            .first()
            .and_then(|input| input.dtype().tensor_shape())
            .ok_or_else(|| InferError::InvalidModelOutput("missing YOLO tensor input".into()))?;
        if shape.len() != 4 || shape[1] != 3 || !matches!(shape[0], 1 | -1) {
            return Err(InferError::InvalidModelOutput(
                "YOLO input must be [1,3,H,W]".into(),
            ));
        }
        let h = if shape[2] == -1 { 640 } else { shape[2] };
        let w = if shape[3] == -1 { 640 } else { shape[3] };
        if h <= 0 || h != w || h > u32::MAX as i64 {
            return Err(InferError::InvalidModelOutput(
                "YOLO input must be square".into(),
            ));
        }
        tracing::info!(
            ?model_path,
            cuda = cfg!(feature = "cuda"),
            rocm = cfg!(feature = "rocm"),
            directml = cfg!(feature = "directml"),
            coreml = cfg!(feature = "coreml"),
            openvino = cfg!(feature = "openvino"),
            cpu = true,
            "YOLO ONNX requested execution providers"
        );
        Ok(Self {
            session: Mutex::new(session),
            input_size: h as u32,
        })
    }

    pub fn input_size(&self) -> u32 {
        self.input_size
    }

    pub fn detect(
        &self,
        rgb_hwc: &[u8],
        conf: f64,
        iou: f64,
        num_classes: usize,
        scale_info: &ScaleInfo,
    ) -> Result<Vec<Detection>, InferError> {
        let size = self.input_size as usize;
        if size.checked_mul(size).and_then(|n| n.checked_mul(3)) != Some(rgb_hwc.len()) {
            return Err(InferError::InvalidModelOutput(
                "YOLO input size mismatch".into(),
            ));
        }
        let input = preprocess_rgb(rgb_hwc, size);
        let mut session = self
            .session
            .lock()
            .map_err(|_| InferError::InvalidModelOutput("YOLO session lock poisoned".into()))?;
        let name = session.inputs()[0].name().to_owned();
        let outputs = session.run(ort::inputs![name => Value::from_array(input)?])?;
        if outputs.len() == 0 {
            return Err(InferError::InvalidModelOutput("missing YOLO output".into()));
        }
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        if shape.len() != 3 || shape[0] != 1 || shape[1] < 0 || shape[2] < 0 {
            return Err(InferError::InvalidModelOutput(
                "YOLO output must be [1,4+nc,N]".into(),
            ));
        }
        let decoded =
            decode_ultralytics_output(data, shape[1] as usize, shape[2] as usize, num_classes)
                .map_err(InferError::InvalidModelOutput)?;
        Ok(finalize_detections(
            decoded,
            conf,
            iou,
            false,
            self.input_size,
            scale_info,
        ))
    }
}

fn preprocess_rgb(rgb: &[u8], size: usize) -> Array4<f32> {
    Array4::from_shape_fn((1, 3, size, size), |(_, c, y, x)| {
        rgb[(y * size + x) * 3 + c] as f32 / 255.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocessing_normalizes_and_transposes_rgb() {
        let input = preprocess_rgb(&[0, 127, 255, 255, 64, 0, 32, 16, 8, 4, 2, 1], 2);
        assert_eq!(input.shape(), &[1, 3, 2, 2]);
        assert_eq!(input[[0, 0, 0, 0]], 0.0);
        assert_eq!(input[[0, 2, 0, 0]], 1.0);
        assert_eq!(input[[0, 0, 0, 1]], 1.0);
        assert_eq!(input[[0, 1, 0, 0]], 127.0 / 255.0);
        assert_eq!(input[[0, 2, 1, 1]], 1.0 / 255.0);
    }
}
