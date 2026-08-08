use std::ffi::{c_char, c_void, CString};
use std::ptr::NonNull;

use super::{
    check_status, ffi, HailoRtError, HailoRtResult, QuantInfo, TensorShape, VStreamDirection,
    VStreamInfo,
};

#[repr(C)]
#[derive(Clone, Copy)]
struct YuHailortTensorInfo {
    name: [c_char; ffi::HAILO_MAX_STREAM_NAME_SIZE],
    height: u32,
    width: u32,
    features: u32,
    format_type: u32,
    qp_zp: f32,
    qp_scale: f32,
    frame_size: usize,
}

#[repr(C)]
struct YuHailortYoloMetadata {
    inputs_count: usize,
    outputs_count: usize,
    inputs: [YuHailortTensorInfo; ffi::HAILO_MAX_STREAMS_COUNT],
    outputs: [YuHailortTensorInfo; ffi::HAILO_MAX_STREAMS_COUNT],
}

impl Default for YuHailortYoloMetadata {
    fn default() -> Self {
        // SAFETY: The C shim fills this POD struct before Rust reads count-bounded entries.
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
struct YuHailortBuffer {
    name: *const c_char,
    data: *mut c_void,
    size: usize,
}

enum YuHailortYolo {}

extern "C" {
    fn yu_hailort_set_vdevice_group_id(group_id: *const c_char) -> ffi::HailoStatus;
    fn yu_hailort_yolo_create(
        hef_path: *const c_char,
        out: *mut *mut YuHailortYolo,
    ) -> ffi::HailoStatus;
    fn yu_hailort_yolo_release(ctx: *mut YuHailortYolo);
    fn yu_hailort_yolo_metadata(
        ctx: *const YuHailortYolo,
        metadata: *mut YuHailortYoloMetadata,
    ) -> ffi::HailoStatus;
    fn yu_hailort_yolo_run(
        ctx: *mut YuHailortYolo,
        input: *const u8,
        input_size: usize,
        outputs: *mut YuHailortBuffer,
        outputs_count: usize,
        timeout_ms: u32,
    ) -> ffi::HailoStatus;
    #[cfg(all(test, hailo_stub))]
    fn yu_hailort_stub_vdevice_create_count() -> usize;
    #[cfg(all(test, hailo_stub))]
    fn yu_hailort_stub_yolo_create_count() -> usize;
    #[cfg(all(test, hailo_stub))]
    fn yu_hailort_stub_yolo_release_count() -> usize;
    #[cfg(all(test, hailo_stub))]
    fn yu_hailort_stub_s2t_create_count() -> usize;
    #[cfg(all(test, hailo_stub))]
    fn yu_hailort_stub_s2t_release_count() -> usize;
    #[cfg(all(test, hailo_stub))]
    fn yu_hailort_stub_llm_create_count() -> usize;
    #[cfg(all(test, hailo_stub))]
    fn yu_hailort_stub_llm_release_count() -> usize;
    #[cfg(all(test, hailo_stub))]
    fn yu_hailort_stub_llm_clear_context_count() -> usize;
    #[cfg(all(test, hailo_stub))]
    fn yu_hailort_stub_vlm_create_count() -> usize;
    #[cfg(all(test, hailo_stub))]
    fn yu_hailort_stub_vlm_release_count() -> usize;
}

#[cfg(all(test, hailo_stub))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct StubCounts {
    pub(crate) vdevice_create: usize,
    pub(crate) yolo_create: usize,
    pub(crate) yolo_release: usize,
    pub(crate) s2t_create: usize,
    pub(crate) s2t_release: usize,
    pub(crate) llm_create: usize,
    pub(crate) llm_release: usize,
    pub(crate) llm_clear_context: usize,
    pub(crate) vlm_create: usize,
    pub(crate) vlm_release: usize,
}

#[cfg(all(test, hailo_stub))]
pub(crate) fn stub_counts() -> StubCounts {
    // SAFETY: these test-only stub functions take no pointers and return process counters.
    unsafe {
        StubCounts {
            vdevice_create: yu_hailort_stub_vdevice_create_count(),
            yolo_create: yu_hailort_stub_yolo_create_count(),
            yolo_release: yu_hailort_stub_yolo_release_count(),
            s2t_create: yu_hailort_stub_s2t_create_count(),
            s2t_release: yu_hailort_stub_s2t_release_count(),
            llm_create: yu_hailort_stub_llm_create_count(),
            llm_release: yu_hailort_stub_llm_release_count(),
            llm_clear_context: yu_hailort_stub_llm_clear_context_count(),
            vlm_create: yu_hailort_stub_vlm_create_count(),
            vlm_release: yu_hailort_stub_vlm_release_count(),
        }
    }
}

pub(crate) fn set_vdevice_group_id(group_id: &str) -> HailoRtResult<()> {
    let group_id = CString::new(group_id)?;
    // SAFETY: group_id is a valid C string and remains alive for the call; the shim copies it.
    let status = unsafe { yu_hailort_set_vdevice_group_id(group_id.as_ptr()) };
    check_status("set_vdevice_group_id", status)
}

pub(crate) struct ShimYolo {
    raw: NonNull<YuHailortYolo>,
}

impl ShimYolo {
    pub(crate) fn create(hef_path: &str) -> HailoRtResult<Self> {
        let hef_path = CString::new(hef_path)?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: hef_path is a valid C string; raw out pointer is valid.
        let status = unsafe { yu_hailort_yolo_create(hef_path.as_ptr(), &mut raw) };
        check_status("shim_yolo_create", status)?;
        let raw = NonNull::new(raw).ok_or(HailoRtError::InvalidMetadata(
            "shim_yolo_create returned null",
        ))?;
        Ok(Self { raw })
    }

    pub(crate) fn metadata(&self) -> HailoRtResult<(Vec<VStreamInfo>, Vec<VStreamInfo>)> {
        let mut metadata = YuHailortYoloMetadata::default();
        // SAFETY: raw context is alive for self lifetime; metadata out pointer is valid.
        let status = unsafe { yu_hailort_yolo_metadata(self.raw.as_ptr(), &mut metadata) };
        check_status("shim_yolo_metadata", status)?;
        if metadata.inputs_count > metadata.inputs.len()
            || metadata.outputs_count > metadata.outputs.len()
        {
            return Err(HailoRtError::InvalidMetadata(
                "shim returned too many vstreams",
            ));
        }
        let inputs = convert_infos(
            &metadata.inputs[..metadata.inputs_count],
            VStreamDirection::Input,
        );
        let outputs = convert_infos(
            &metadata.outputs[..metadata.outputs_count],
            VStreamDirection::Output,
        );
        Ok((inputs, outputs))
    }

    pub(crate) fn run(
        &mut self,
        input: &[u8],
        outputs: &mut [(&str, &mut [u8])],
        timeout_ms: u32,
    ) -> HailoRtResult<()> {
        let names = outputs
            .iter()
            .map(|(name, _)| CString::new(*name))
            .collect::<Result<Vec<_>, _>>()?;
        let mut buffers = outputs
            .iter_mut()
            .zip(names.iter())
            .map(|((_, data), name)| YuHailortBuffer {
                name: name.as_ptr(),
                data: data.as_mut_ptr().cast(),
                size: data.len(),
            })
            .collect::<Vec<_>>();
        // SAFETY: input and output slices live until call returns; raw context is owned by self.
        let status = unsafe {
            yu_hailort_yolo_run(
                self.raw.as_ptr(),
                input.as_ptr(),
                input.len(),
                buffers.as_mut_ptr(),
                buffers.len(),
                timeout_ms,
            )
        };
        check_status("shim_yolo_run", status)
    }
}

impl Drop for ShimYolo {
    fn drop(&mut self) {
        // SAFETY: raw is owned by this RAII wrapper and released once here.
        unsafe { yu_hailort_yolo_release(self.raw.as_ptr()) };
    }
}

fn convert_infos(raw: &[YuHailortTensorInfo], direction: VStreamDirection) -> Vec<VStreamInfo> {
    raw.iter()
        .map(|info| VStreamInfo {
            name: super::metadata::c_char_array_to_string(&info.name),
            direction,
            shape: TensorShape::new([
                info.height as usize,
                info.width as usize,
                info.features as usize,
            ]),
            quant: QuantInfo {
                zero_point: info.qp_zp,
                scale: info.qp_scale,
            },
            format_type: info.format_type as ffi::HailoFormatType,
            frame_size: info.frame_size,
        })
        .collect()
}

#[cfg(all(test, hailo_stub))]
mod tests {
    use super::*;
    use crate::hailort::{Llm, Speech2Text, Vlm};

    #[test]
    fn all_create_paths_create_at_most_one_vdevice() {
        let _yolo = ShimYolo::create("model").unwrap();
        let _speech2text = Speech2Text::create("model").unwrap();
        let _llm = Llm::create("model", None, false).unwrap();
        let _vlm = Vlm::create("model", false).unwrap();
        let count = stub_counts().vdevice_create;
        assert!(count > 0 && count <= 1, "VDevice create count was {count}");
    }
}
