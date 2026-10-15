use std::path::Path;
use yu_hailo_infer_core::{
    yolo_onnx::YoloOnnxDetector,
    yolo_postprocess::{Detection, ScaleInfo},
    InferError,
};

#[test]
fn real_onnx_fixture_matches_letterbox_reference_and_nms() {
    let detector = YoloOnnxDetector::new(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/yolo_constant.onnx"
    )))
    .unwrap();
    assert_eq!(detector.input_size(), 640);
    let scale = ScaleInfo {
        orig_w: 1280,
        orig_h: 640,
        scale: 0.5,
        pad_x: 0.,
        pad_y: 160.,
    };
    let detections = detector
        .detect(&vec![114; 640 * 640 * 3], 0.25, 0.45, 80, &scale)
        .unwrap();
    assert_eq!(
        detections,
        vec![Detection {
            class_id: 0,
            class_name: "person".into(),
            confidence: 0.9,
            bbox: [0.25, 0.25, 0.75, 0.75]
        }]
    );
    assert!(detector.detect(&[0; 3], 0.25, 0.45, 80, &scale).is_err());
    assert!(detector
        .detect(&vec![0; 640 * 640 * 3], 0.25, 0.45, 2, &scale)
        .is_err());
    assert!(detector
        .detect(&vec![0; 640 * 640 * 3], 0.95, 0.45, 80, &scale)
        .unwrap()
        .is_empty());
}

#[test]
fn missing_model_is_not_downloaded() {
    assert!(matches!(
        YoloOnnxDetector::new(Path::new("/nonexistent/yolo.onnx")),
        Err(InferError::ModelNotDownloaded(_))
    ));
}
