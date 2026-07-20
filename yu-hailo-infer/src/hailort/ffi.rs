#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::{c_char, c_int, c_void};

pub(crate) type HailoStatus = c_int;
pub(crate) type HailoFormatType = c_int;
pub(crate) type HailoFormatOrder = c_int;
pub(crate) type HailoFormatFlags = c_int;
pub(crate) type HailoStreamDirection = c_int;
pub(crate) type HailoVStreamStatsFlags = c_int;
pub(crate) type HailoPipelineElemStatsFlags = c_int;

pub(crate) type HailoVDevice = *mut c_void;
pub(crate) type HailoHef = *mut c_void;
pub(crate) type HailoConfiguredNetworkGroup = *mut c_void;
pub(crate) type HailoActivatedNetworkGroup = *mut c_void;

pub(crate) const HAILO_SUCCESS: HailoStatus = 0;
pub(crate) const HAILO_INVALID_ARGUMENT: HailoStatus = 2;
pub(crate) const HAILO_INSUFFICIENT_BUFFER: HailoStatus = 5;

pub(crate) const HAILO_MAX_STREAM_NAME_SIZE: usize = 128;
pub(crate) const HAILO_MAX_NETWORK_NAME_SIZE: usize = 257;
pub(crate) const HAILO_MAX_STREAMS_COUNT: usize = 40;
pub(crate) const HAILO_MAX_NETWORK_GROUPS: usize = 8;

pub(crate) const HAILO_FORMAT_TYPE_UINT8: HailoFormatType = 1;
pub(crate) const HAILO_FORMAT_TYPE_UINT16: HailoFormatType = 2;
pub(crate) const HAILO_FORMAT_TYPE_FLOAT32: HailoFormatType = 3;

pub(crate) const HAILO_H2D_STREAM: HailoStreamDirection = 0;
pub(crate) const HAILO_D2H_STREAM: HailoStreamDirection = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct HailoFormat {
    pub(crate) type_: HailoFormatType,
    pub(crate) order: HailoFormatOrder,
    pub(crate) flags: HailoFormatFlags,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct HailoQuantInfo {
    pub(crate) qp_zp: f32,
    pub(crate) qp_scale: f32,
    pub(crate) limvals_min: f32,
    pub(crate) limvals_max: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct HailoVStreamParams {
    pub(crate) user_buffer_format: HailoFormat,
    pub(crate) timeout_ms: u32,
    pub(crate) queue_size: u32,
    pub(crate) vstream_stats_flags: HailoVStreamStatsFlags,
    pub(crate) pipeline_elements_stats_flags: HailoPipelineElemStatsFlags,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct HailoInputVStreamParamsByName {
    pub(crate) name: [c_char; HAILO_MAX_STREAM_NAME_SIZE],
    pub(crate) params: HailoVStreamParams,
}

impl Default for HailoInputVStreamParamsByName {
    fn default() -> Self {
        // SAFETY: This C POD struct is valid when zero-initialized before HailoRT fills it.
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct HailoOutputVStreamParamsByName {
    pub(crate) name: [c_char; HAILO_MAX_STREAM_NAME_SIZE],
    pub(crate) params: HailoVStreamParams,
}

impl Default for HailoOutputVStreamParamsByName {
    fn default() -> Self {
        // SAFETY: This C POD struct is valid when zero-initialized before HailoRT fills it.
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Hailo3dImageShape {
    pub(crate) height: u32,
    pub(crate) width: u32,
    pub(crate) features: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct HailoNmsShape {
    pub(crate) number_of_classes: u32,
    pub(crate) max_bboxes_per_class: u32,
    pub(crate) max_bboxes_total: u32,
    pub(crate) max_accumulated_mask_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) union HailoVStreamShape {
    pub(crate) shape: Hailo3dImageShape,
    pub(crate) nms_shape: HailoNmsShape,
}

impl Default for HailoVStreamShape {
    fn default() -> Self {
        Self {
            shape: Hailo3dImageShape::default(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct HailoVStreamInfo {
    pub(crate) name: [c_char; HAILO_MAX_STREAM_NAME_SIZE],
    pub(crate) network_name: [c_char; HAILO_MAX_NETWORK_NAME_SIZE],
    pub(crate) direction: HailoStreamDirection,
    pub(crate) format: HailoFormat,
    pub(crate) shape: HailoVStreamShape,
    pub(crate) quant_info: HailoQuantInfo,
}

impl Default for HailoVStreamInfo {
    fn default() -> Self {
        // SAFETY: This C POD struct is valid when zero-initialized before HailoRT fills it.
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct HailoStreamRawBuffer {
    pub(crate) buffer: *mut c_void,
    pub(crate) size: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct HailoStreamRawBufferByName {
    pub(crate) name: [c_char; HAILO_MAX_STREAM_NAME_SIZE],
    pub(crate) raw_buffer: HailoStreamRawBuffer,
}

impl Default for HailoStreamRawBufferByName {
    fn default() -> Self {
        // SAFETY: This C POD struct is valid when zero-initialized before fields are assigned.
        unsafe { std::mem::zeroed() }
    }
}

#[cfg_attr(not(hailo_stub), link(name = "hailort"))]
extern "C" {
    pub(crate) fn hailo_create_vdevice(
        params: *mut c_void,
        vdevice: *mut HailoVDevice,
    ) -> HailoStatus;
    pub(crate) fn hailo_release_vdevice(vdevice: HailoVDevice) -> HailoStatus;

    pub(crate) fn hailo_create_hef_file(
        hef: *mut HailoHef,
        file_name: *const c_char,
    ) -> HailoStatus;
    pub(crate) fn hailo_release_hef(hef: HailoHef) -> HailoStatus;

    pub(crate) fn hailo_hef_get_all_vstream_infos(
        hef: HailoHef,
        name: *const c_char,
        vstream_infos: *mut HailoVStreamInfo,
        vstream_infos_count: *mut usize,
    ) -> HailoStatus;

    pub(crate) fn hailo_configure_vdevice(
        vdevice: HailoVDevice,
        hef: HailoHef,
        params: *mut c_void,
        network_groups: *mut HailoConfiguredNetworkGroup,
        number_of_network_groups: *mut usize,
    ) -> HailoStatus;

    pub(crate) fn hailo_shutdown_network_group(
        network_group: HailoConfiguredNetworkGroup,
    ) -> HailoStatus;

    pub(crate) fn hailo_activate_network_group(
        network_group: HailoConfiguredNetworkGroup,
        activation_params: *mut c_void,
        activated_network_group_out: *mut HailoActivatedNetworkGroup,
    ) -> HailoStatus;
    pub(crate) fn hailo_deactivate_network_group(
        activated_network_group: HailoActivatedNetworkGroup,
    ) -> HailoStatus;

    pub(crate) fn hailo_hef_make_input_vstream_params(
        hef: HailoHef,
        name: *const c_char,
        unused: bool,
        format_type: HailoFormatType,
        input_params: *mut HailoInputVStreamParamsByName,
        input_params_count: *mut usize,
    ) -> HailoStatus;

    pub(crate) fn hailo_hef_make_output_vstream_params(
        hef: HailoHef,
        name: *const c_char,
        unused: bool,
        format_type: HailoFormatType,
        output_params: *mut HailoOutputVStreamParamsByName,
        output_params_count: *mut usize,
    ) -> HailoStatus;

    pub(crate) fn hailo_get_vstream_frame_size(
        vstream_info: *mut HailoVStreamInfo,
        user_buffer_format: *mut HailoFormat,
        frame_size: *mut usize,
    ) -> HailoStatus;

    pub(crate) fn hailo_infer(
        configured_network_group: HailoConfiguredNetworkGroup,
        inputs_params: *mut HailoInputVStreamParamsByName,
        input_buffers: *mut HailoStreamRawBufferByName,
        inputs_count: usize,
        outputs_params: *mut HailoOutputVStreamParamsByName,
        output_buffers: *mut HailoStreamRawBufferByName,
        outputs_count: usize,
        frames_count: usize,
    ) -> HailoStatus;
}
