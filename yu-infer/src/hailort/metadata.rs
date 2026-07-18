use std::ffi::CStr;
use std::os::raw::c_char;

use super::{check_status, ffi, HailoRtResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VStreamDirection {
    Input,
    Output,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TensorShape {
    dimensions: [usize; 3],
}

impl TensorShape {
    pub(crate) fn new(dimensions: [usize; 3]) -> Self {
        Self { dimensions }
    }

    pub(crate) fn from_hailo(shape: ffi::Hailo3dImageShape) -> Self {
        Self::new([
            shape.height as usize,
            shape.width as usize,
            shape.features as usize,
        ])
    }

    pub(crate) fn element_count(&self) -> usize {
        self.dimensions.iter().product()
    }

    pub(crate) fn dimensions(&self) -> [usize; 3] {
        self.dimensions
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct QuantInfo {
    pub(crate) zero_point: f32,
    pub(crate) scale: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VStreamInfo {
    pub(crate) name: String,
    pub(crate) direction: VStreamDirection,
    pub(crate) shape: TensorShape,
    pub(crate) quant: QuantInfo,
    pub(crate) format_type: ffi::HailoFormatType,
    pub(crate) frame_size: usize,
}

impl VStreamInfo {
    pub(crate) fn from_raw(mut raw: ffi::HailoVStreamInfo) -> HailoRtResult<Self> {
        let direction = match raw.direction {
            ffi::HAILO_H2D_STREAM => VStreamDirection::Input,
            ffi::HAILO_D2H_STREAM => VStreamDirection::Output,
            _ => {
                return Err(super::HailoRtError::InvalidMetadata(
                    "unknown vstream direction",
                ))
            }
        };
        let mut format = raw.format;
        let mut frame_size = 0usize;
        // SAFETY: raw and format are stack-owned C-compatible structs copied from HailoRT metadata.
        let status =
            unsafe { ffi::hailo_get_vstream_frame_size(&mut raw, &mut format, &mut frame_size) };
        check_status("get_vstream_frame_size", status)?;

        Ok(Self {
            name: c_char_array_to_string(&raw.name),
            direction,
            // SAFETY: For the initial YOLO path we only support tensor-shaped vstreams.
            shape: TensorShape::from_hailo(unsafe { raw.shape.shape }),
            quant: QuantInfo {
                zero_point: raw.quant_info.qp_zp,
                scale: raw.quant_info.qp_scale,
            },
            format_type: raw.format.type_,
            frame_size,
        })
    }
}

pub(crate) fn c_char_array_to_string<const N: usize>(value: &[c_char; N]) -> String {
    let nul = value.iter().position(|&ch| ch == 0).unwrap_or(N);
    let bytes: Vec<u8> = value[..nul].iter().map(|&ch| ch as u8).collect();
    CStr::from_bytes_until_nul(&[bytes.as_slice(), &[0]].concat())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|_| String::from_utf8_lossy(&bytes).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_shape_size_multiplies_dimensions() {
        let shape = TensorShape::new([1, 640, 640 * 3]);
        assert_eq!(shape.element_count(), 1_228_800);
    }
}
