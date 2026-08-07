#![allow(dead_code)]

mod clip;
mod error;
mod ffi;
mod llm;
mod metadata;
mod safe;
mod shim;
mod speech2text;
mod vlm;
mod yolo;

#[allow(unused_imports)]
pub(crate) use clip::{load_clip_image_metadata, run_clip_image_once, ClipImageMetadata};
#[allow(unused_imports)]
pub(crate) use error::{check_status, HailoRtError, HailoRtResult};
#[allow(unused_imports)]
pub(crate) use llm::{Llm, LlmChatMessage, LlmCompletionStatus, LlmGenerationParams, LlmStream};
#[allow(unused_imports)]
pub(crate) use metadata::{QuantInfo, TensorShape, VStreamDirection, VStreamInfo};
#[allow(unused_imports)]
pub(crate) use safe::{ConfiguredNetworkGroup, Hef, VDevice};
#[allow(unused_imports)]
pub(crate) use speech2text::{Speech2Text, Speech2TextTask};
#[allow(unused_imports)]
pub(crate) use vlm::{Vlm, VlmCompletionStatus, VlmGenerationParams, VlmStream};
#[allow(unused_imports)]
pub(crate) use yolo::{
    load_yolo_metadata, run_yolo_once, validate_input_len, YoloInferenceResult, YoloModelMetadata,
    YoloOutputBuffer,
};

#[cfg(test)]
mod tests {
    // Gated to match its only consumer below. Without HailoRT headers build.rs
    // sets cfg(hailo_stub), the test disappears, and an ungated import here
    // becomes dead — which fails `clippy -D warnings` on any host lacking the
    // SDK, including CI. An `#[allow(unused_imports)]` would hide that, and
    // would also hide a genuinely dead import added later.
    #[cfg(not(hailo_stub))]
    use super::*;

    #[cfg(not(hailo_stub))]
    #[test]
    fn hailo_success_constant_matches_header() {
        assert_eq!(ffi::HAILO_SUCCESS, 0);
    }
}
