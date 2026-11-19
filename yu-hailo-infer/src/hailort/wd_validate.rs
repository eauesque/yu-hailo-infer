//! Validate calibration metadata before allocating/configuring a WD network.
use super::{ffi, wd_hef::WdHefError, VStreamInfo};

pub(crate) fn metadata(
    inputs: &[VStreamInfo],
    outputs: &[VStreamInfo],
    tag_count: usize,
) -> Result<(), WdHefError> {
    let ([input], [output]) = (inputs, outputs) else {
        return Err(WdHefError::Invalid);
    };
    let count = output
        .shape
        .dimensions()
        .into_iter()
        .try_fold(1usize, usize::checked_mul);
    if input.shape.dimensions() != [448, 448, 3]
        || input.format_type != ffi::HAILO_FORMAT_TYPE_UINT8
        || input.format_order != ffi::HAILO_FORMAT_ORDER_NHWC
        || input.frame_size != 602_112
        || output.format_type != ffi::HAILO_FORMAT_TYPE_UINT8
        || tag_count == 0
        || count != Some(tag_count)
        || output.frame_size != tag_count
        || [input, output].iter().any(|info| {
            !info.quant.scale.is_finite()
                || info.quant.scale <= 0.0
                || !info.quant.zero_point.is_finite()
                || !((0.0 - info.quant.zero_point) * info.quant.scale).is_finite()
                || !((255.0 - info.quant.zero_point) * info.quant.scale).is_finite()
        })
    {
        return Err(WdHefError::Invalid);
    }
    Ok(())
}
