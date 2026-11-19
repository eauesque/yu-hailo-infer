use super::*;
use core::mem::{offset_of, size_of};

extern "C" {
    fn yu_hailort_stub_yolo_enable_metadata(ctx: *mut YuHailortYolo);
}

#[test]
fn stub_yolo_metadata_reports_nhwc() {
    let model = ShimYolo::create("model").unwrap();
    assert!(model.metadata().is_err());
    // SAFETY: model owns the live stub context; this only enables its metadata fixture.
    unsafe { yu_hailort_stub_yolo_enable_metadata(model.raw.as_ptr()) };
    let metadata = super::super::yolo_metadata(&model).unwrap();
    assert_eq!(metadata.inputs.len(), 1);
    assert_eq!(metadata.outputs.len(), 1);
    for info in metadata.inputs.iter().chain(&metadata.outputs) {
        assert_eq!(info.format_order, ffi::HAILO_FORMAT_ORDER_NHWC);
        assert_eq!(info.format_type, ffi::HAILO_FORMAT_TYPE_UINT8);
        assert_eq!(
            info.quant,
            QuantInfo {
                zero_point: 0.0,
                scale: 1.0
            }
        );
    }
    assert_eq!(metadata.inputs[0].name, "input");
    assert_eq!(metadata.inputs[0].direction, VStreamDirection::Input);
    assert_eq!(metadata.inputs[0].shape.dimensions(), [1, 1, 3]);
    assert_eq!(metadata.inputs[0].frame_size, 3);
    assert_eq!(metadata.outputs[0].name, "output");
    assert_eq!(metadata.outputs[0].direction, VStreamDirection::Output);
    assert_eq!(metadata.outputs[0].shape.dimensions(), [1, 1, 512]);
    assert_eq!(metadata.outputs[0].frame_size, 512);
    super::super::clip_image_metadata(&model).unwrap();
}

#[test]
fn tensor_metadata_layout_matches_lp64_header() {
    assert_eq!(size_of::<YuHailortTensorInfo>(), 168);
    assert_eq!(offset_of!(YuHailortTensorInfo, name), 0);
    assert_eq!(offset_of!(YuHailortTensorInfo, height), 128);
    assert_eq!(offset_of!(YuHailortTensorInfo, width), 132);
    assert_eq!(offset_of!(YuHailortTensorInfo, features), 136);
    assert_eq!(offset_of!(YuHailortTensorInfo, format_type), 140);
    assert_eq!(offset_of!(YuHailortTensorInfo, qp_zp), 144);
    assert_eq!(offset_of!(YuHailortTensorInfo, qp_scale), 148);
    assert_eq!(offset_of!(YuHailortTensorInfo, frame_size), 152);
    assert_eq!(offset_of!(YuHailortTensorInfo, format_order), 160);
    assert_eq!(size_of::<YuHailortYoloMetadata>(), 13456);
    assert_eq!(offset_of!(YuHailortYoloMetadata, inputs_count), 0);
    assert_eq!(offset_of!(YuHailortYoloMetadata, outputs_count), 8);
    assert_eq!(offset_of!(YuHailortYoloMetadata, inputs), 16);
    assert_eq!(offset_of!(YuHailortYoloMetadata, outputs), 6736);
}
