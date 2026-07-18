use std::ffi::{c_char, c_int, CString};
use std::ptr::NonNull;

use super::{check_status, ffi, HailoRtError, HailoRtResult};

enum YuHailortVlm {}
enum YuHailortVlmStream {}

extern "C" {
    fn yu_hailort_vlm_create(
        model_path: *const c_char,
        optimize_memory_on_device: bool,
        out: *mut *mut YuHailortVlm,
    ) -> ffi::HailoStatus;
    fn yu_hailort_vlm_release(ctx: *mut YuHailortVlm);
    fn yu_hailort_vlm_generate_text(
        ctx: *mut YuHailortVlm,
        prompt: *const c_char,
        frames: *const *const u8,
        frame_sizes: *const usize,
        frame_count: usize,
        timeout_ms: u32,
        out_text: *mut *mut c_char,
    ) -> ffi::HailoStatus;
    fn yu_hailort_vlm_tokenize(
        ctx: *mut YuHailortVlm,
        text: *const c_char,
        tokens: *mut c_int,
        tokens_count: *mut usize,
    ) -> ffi::HailoStatus;
    fn yu_hailort_vlm_context_usage(ctx: *mut YuHailortVlm, out: *mut usize) -> ffi::HailoStatus;
    fn yu_hailort_vlm_max_context_capacity(
        ctx: *mut YuHailortVlm,
        out: *mut usize,
    ) -> ffi::HailoStatus;
    fn yu_hailort_vlm_clear_context(ctx: *mut YuHailortVlm) -> ffi::HailoStatus;
    fn yu_hailort_vlm_input_frame_info(
        ctx: *mut YuHailortVlm,
        frame_size: *mut u32,
        height: *mut u32,
        width: *mut u32,
        features: *mut u32,
        format_type: *mut u32,
        format_order: *mut u32,
    ) -> ffi::HailoStatus;
    fn yu_hailort_vlm_generate_stream_start(
        ctx: *mut YuHailortVlm,
        prompt: *const c_char,
        system_prompt: *const c_char,
        frames: *const *const u8,
        frame_sizes: *const usize,
        frame_count: usize,
        temperature: *const f32,
        top_p: *const f32,
        top_k: *const u32,
        frequency_penalty: *const f32,
        max_generated_tokens: *const u32,
        do_sample: *const bool,
        seed: *const u32,
        out: *mut *mut YuHailortVlmStream,
    ) -> ffi::HailoStatus;
    fn yu_hailort_vlm_stream_read(
        stream: *mut YuHailortVlmStream,
        timeout_ms: u32,
        out_token: *mut *mut c_char,
        out_status: *mut c_int,
    ) -> ffi::HailoStatus;
    fn yu_hailort_vlm_stream_release(stream: *mut YuHailortVlmStream);
    fn yu_hailort_string_free(value: *mut c_char);
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct VlmInputFrameInfo {
    pub(crate) frame_size: u32,
    pub(crate) height: u32,
    pub(crate) width: u32,
    pub(crate) features: u32,
    pub(crate) format_type: u32,
    pub(crate) format_order: u32,
}

/// Overrides for VLM generation sampling. `None` fields keep the model's
/// default value (queried on-device via `create_generator_params()`).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct VlmGenerationParams {
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<u32>,
    pub(crate) frequency_penalty: Option<f32>,
    pub(crate) max_generated_tokens: Option<u32>,
    pub(crate) do_sample: Option<bool>,
    pub(crate) seed: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VlmCompletionStatus {
    Generating,
    MaxTokensReached,
    LogicalEndOfGeneration,
    Aborted,
}

impl VlmCompletionStatus {
    fn from_raw(value: c_int) -> Self {
        match value {
            0 => Self::Generating,
            1 => Self::MaxTokensReached,
            2 => Self::LogicalEndOfGeneration,
            _ => Self::Aborted,
        }
    }

    pub(crate) fn is_generating(self) -> bool {
        matches!(self, Self::Generating)
    }
}

pub(crate) struct Vlm {
    raw: NonNull<YuHailortVlm>,
}

impl Vlm {
    pub(crate) fn create(model_path: &str, optimize_memory_on_device: bool) -> HailoRtResult<Self> {
        let model_path = CString::new(model_path)?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: C string and out pointer are valid until call returns.
        let status = unsafe {
            yu_hailort_vlm_create(model_path.as_ptr(), optimize_memory_on_device, &mut raw)
        };
        check_status("vlm_create", status)?;
        let raw =
            NonNull::new(raw).ok_or(HailoRtError::InvalidMetadata("vlm_create returned null"))?;
        Ok(Self { raw })
    }

    /// `frames` must already be preprocessed to the model's expected input
    /// format (resized, RGB, correct dtype) — the shim passes raw bytes
    /// through to HailoRT without any conversion.
    pub(crate) fn generate_text(
        &mut self,
        prompt: &str,
        frames: &[Vec<u8>],
        timeout_ms: u32,
    ) -> HailoRtResult<String> {
        let prompt = CString::new(prompt)?;
        let frame_ptrs: Vec<*const u8> = frames.iter().map(|frame| frame.as_ptr()).collect();
        let frame_sizes: Vec<usize> = frames.iter().map(|frame| frame.len()).collect();
        let mut out = std::ptr::null_mut();
        // SAFETY: prompt, frame buffers, and out pointer are valid until call returns.
        let status = unsafe {
            yu_hailort_vlm_generate_text(
                self.raw.as_ptr(),
                prompt.as_ptr(),
                frame_ptrs.as_ptr(),
                frame_sizes.as_ptr(),
                frame_ptrs.len(),
                timeout_ms,
                &mut out,
            )
        };
        check_status("vlm_generate_text", status)?;
        take_c_string(out)
    }

    pub(crate) fn tokenize(&mut self, text: &str) -> HailoRtResult<Vec<i32>> {
        let text = CString::new(text)?;
        let mut count = 0usize;
        // SAFETY: First call asks shim for required count.
        let status = unsafe {
            yu_hailort_vlm_tokenize(
                self.raw.as_ptr(),
                text.as_ptr(),
                std::ptr::null_mut(),
                &mut count,
            )
        };
        if status != ffi::HAILO_INSUFFICIENT_BUFFER {
            check_status("vlm_tokenize_count", status)?;
        }
        let mut tokens = vec![0; count];
        // SAFETY: tokens buffer has count entries requested by shim.
        let status = unsafe {
            yu_hailort_vlm_tokenize(
                self.raw.as_ptr(),
                text.as_ptr(),
                tokens.as_mut_ptr(),
                &mut count,
            )
        };
        check_status("vlm_tokenize", status)?;
        tokens.truncate(count);
        Ok(tokens)
    }

    pub(crate) fn context_usage(&mut self) -> HailoRtResult<usize> {
        let mut out = 0usize;
        // SAFETY: out pointer is valid.
        let status = unsafe { yu_hailort_vlm_context_usage(self.raw.as_ptr(), &mut out) };
        check_status("vlm_context_usage", status)?;
        Ok(out)
    }

    pub(crate) fn max_context_capacity(&mut self) -> HailoRtResult<usize> {
        let mut out = 0usize;
        // SAFETY: out pointer is valid.
        let status = unsafe { yu_hailort_vlm_max_context_capacity(self.raw.as_ptr(), &mut out) };
        check_status("vlm_max_context_capacity", status)?;
        Ok(out)
    }

    pub(crate) fn clear_context(&mut self) -> HailoRtResult<()> {
        // SAFETY: raw context is valid for self lifetime.
        let status = unsafe { yu_hailort_vlm_clear_context(self.raw.as_ptr()) };
        check_status("vlm_clear_context", status)
    }

    pub(crate) fn input_frame_info(&mut self) -> HailoRtResult<VlmInputFrameInfo> {
        let mut info = VlmInputFrameInfo {
            frame_size: 0,
            height: 0,
            width: 0,
            features: 0,
            format_type: 0,
            format_order: 0,
        };
        // SAFETY: all out pointers are valid stack locations.
        let status = unsafe {
            yu_hailort_vlm_input_frame_info(
                self.raw.as_ptr(),
                &mut info.frame_size,
                &mut info.height,
                &mut info.width,
                &mut info.features,
                &mut info.format_type,
                &mut info.format_order,
            )
        };
        check_status("vlm_input_frame_info", status)?;
        Ok(info)
    }
}

impl Drop for Vlm {
    fn drop(&mut self) {
        // SAFETY: raw is owned by this RAII wrapper and released once here.
        unsafe { yu_hailort_vlm_release(self.raw.as_ptr()) };
    }
}

/// A single in-flight VLM generation, read one token at a time.
///
/// Call [`Self::read_next`] in a loop until the returned status is no longer
/// [`VlmCompletionStatus::Generating`] — per the HailoRT SDK contract, no
/// further reads may be attempted once generation has ended.
///
/// Borrows the source [`Vlm`] mutably for its entire lifetime: the SDK
/// explicitly documents that only one generator/completion may be active on
/// a model at a time, so this stream must hold exclusive access to `vlm`
/// until it is dropped — otherwise another `generate_text`/`start` call could
/// run concurrently, which is undefined behavior per the HailoRT SDK.
pub(crate) struct VlmStream<'a> {
    raw: NonNull<YuHailortVlmStream>,
    _vlm: std::marker::PhantomData<&'a mut Vlm>,
}

impl<'a> VlmStream<'a> {
    pub(crate) fn start(
        vlm: &'a mut Vlm,
        prompt: &str,
        system_prompt: Option<&str>,
        frames: &[Vec<u8>],
        params: VlmGenerationParams,
    ) -> HailoRtResult<Self> {
        let prompt = CString::new(prompt)?;
        let system_prompt = system_prompt.map(CString::new).transpose()?;
        let frame_ptrs: Vec<*const u8> = frames.iter().map(|frame| frame.as_ptr()).collect();
        let frame_sizes: Vec<usize> = frames.iter().map(|frame| frame.len()).collect();
        let mut out = std::ptr::null_mut();
        // SAFETY: prompt, frame buffers, optional-override pointers, and out
        // pointer are all valid until the call returns.
        let status = unsafe {
            yu_hailort_vlm_generate_stream_start(
                vlm.raw.as_ptr(),
                prompt.as_ptr(),
                system_prompt
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
                frame_ptrs.as_ptr(),
                frame_sizes.as_ptr(),
                frame_ptrs.len(),
                params
                    .temperature
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value),
                params
                    .top_p
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value),
                params
                    .top_k
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value),
                params
                    .frequency_penalty
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value),
                params
                    .max_generated_tokens
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value),
                params
                    .do_sample
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value),
                params.seed.as_ref().map_or(std::ptr::null(), |value| value),
                &mut out,
            )
        };
        check_status("vlm_generate_stream_start", status)?;
        let raw = NonNull::new(out).ok_or(HailoRtError::InvalidMetadata(
            "vlm_generate_stream_start returned null",
        ))?;
        Ok(Self {
            raw,
            _vlm: std::marker::PhantomData,
        })
    }

    pub(crate) fn read_next(
        &mut self,
        timeout_ms: u32,
    ) -> HailoRtResult<(String, VlmCompletionStatus)> {
        let mut out_token = std::ptr::null_mut();
        let mut out_status: c_int = 0;
        // SAFETY: out pointers are valid stack locations for the duration of the call.
        let status = unsafe {
            yu_hailort_vlm_stream_read(
                self.raw.as_ptr(),
                timeout_ms,
                &mut out_token,
                &mut out_status,
            )
        };
        check_status("vlm_stream_read", status)?;
        let token = take_c_string(out_token)?;
        Ok((token, VlmCompletionStatus::from_raw(out_status)))
    }
}

impl Drop for VlmStream<'_> {
    fn drop(&mut self) {
        // SAFETY: raw is owned by this RAII wrapper and released once here.
        unsafe { yu_hailort_vlm_stream_release(self.raw.as_ptr()) };
    }
}

fn take_c_string(value: *mut c_char) -> HailoRtResult<String> {
    if value.is_null() {
        return Err(HailoRtError::InvalidMetadata("shim returned null string"));
    }
    // SAFETY: value is allocated by shim as a nul-terminated C string.
    let text = unsafe { std::ffi::CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: value was allocated by shim and must be freed by matching shim function.
    unsafe { yu_hailort_string_free(value) };
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vlm_rejects_paths_with_nul_bytes() {
        let err = match Vlm::create("bad\0path", false) {
            Ok(_) => panic!("expected nul byte error"),
            Err(err) => err,
        };
        assert!(matches!(err, HailoRtError::Nul(_)));
    }

    #[test]
    #[ignore = "requires /dev/hailo0 and HAILO_VLM_HEF"]
    fn smoke_vlm_create_and_tokenize() {
        let hef = std::env::var("HAILO_VLM_HEF").expect("HAILO_VLM_HEF");
        let mut vlm = Vlm::create(&hef, false).expect("create vlm");
        let tokens = vlm.tokenize("hello").expect("tokenize");
        assert!(!tokens.is_empty());
        assert!(vlm.max_context_capacity().expect("capacity") > 0);
    }

    #[test]
    #[ignore = "requires /dev/hailo0 and HAILO_VLM_HEF"]
    fn smoke_vlm_input_frame_info() {
        let hef = std::env::var("HAILO_VLM_HEF").expect("HAILO_VLM_HEF");
        let mut vlm = Vlm::create(&hef, false).expect("create vlm");
        let info = vlm.input_frame_info().expect("input_frame_info");
        eprintln!("VLM input_frame_info: {info:?}");
        assert!(info.frame_size > 0);
    }

    #[test]
    #[ignore = "requires /dev/hailo0 and HAILO_VLM_HEF"]
    fn smoke_vlm_generate_with_solid_frame() {
        let hef = std::env::var("HAILO_VLM_HEF").expect("HAILO_VLM_HEF");
        let mut vlm = Vlm::create(&hef, false).expect("create vlm");
        // Frame size/shape is model-specific — query it rather than assume
        // the Python extension's hardcoded 336x336 (that value belongs to
        // qwen2-vl-2b-instruct; other VLM HEFs use different shapes).
        let info = vlm.input_frame_info().expect("input_frame_info");
        let frame = vec![128u8; info.frame_size as usize];
        let text = vlm
            .generate_text("What color is this image?", &[frame], 60_000)
            .expect("generate_text");
        assert!(!text.is_empty());
    }

    #[test]
    #[ignore = "requires /dev/hailo0 and HAILO_VLM_HEF"]
    fn smoke_vlm_generate_stream_with_solid_frame() {
        let hef = std::env::var("HAILO_VLM_HEF").expect("HAILO_VLM_HEF");
        let mut vlm = Vlm::create(&hef, false).expect("create vlm");
        let info = vlm.input_frame_info().expect("input_frame_info");
        let frame = vec![128u8; info.frame_size as usize];
        let mut stream = VlmStream::start(
            &mut vlm,
            "What color is this image?",
            None,
            &[frame],
            VlmGenerationParams {
                max_generated_tokens: Some(16),
                ..VlmGenerationParams::default()
            },
        )
        .expect("generate_stream_start");

        let mut text = String::new();
        let mut status = VlmCompletionStatus::Generating;
        while status.is_generating() {
            let (token, next_status) = stream.read_next(60_000).expect("stream_read");
            eprintln!("VLM stream token: {token:?} status: {next_status:?}");
            text.push_str(&token);
            status = next_status;
        }
        assert!(!text.is_empty());
    }

    #[test]
    #[ignore = "requires /dev/hailo0 and HAILO_VLM_HEF"]
    fn smoke_vlm_generate_stream_accepts_all_generation_params() {
        // Regression test: HailoRT rejects an explicit 0.0 frequency_penalty
        // with HAILO_INVALID_ARGUMENT (found on real hardware) — the shim
        // must skip the setter call in that case. This exercises every
        // VlmGenerationParams field together, not just max_generated_tokens.
        let hef = std::env::var("HAILO_VLM_HEF").expect("HAILO_VLM_HEF");
        let mut vlm = Vlm::create(&hef, false).expect("create vlm");
        let info = vlm.input_frame_info().expect("input_frame_info");
        let frame = vec![128u8; info.frame_size as usize];
        let mut stream = VlmStream::start(
            &mut vlm,
            "What color is this image?",
            Some("You are a helpful assistant that analyzes images."),
            &[frame],
            VlmGenerationParams {
                temperature: Some(0.7),
                top_p: Some(0.9),
                top_k: Some(40),
                frequency_penalty: Some(0.0),
                max_generated_tokens: Some(16),
                do_sample: Some(true),
                seed: Some(42),
            },
        )
        .expect("generate_stream_start with all params set");

        let mut text = String::new();
        let mut status = VlmCompletionStatus::Generating;
        while status.is_generating() {
            let (token, next_status) = stream.read_next(60_000).expect("stream_read");
            text.push_str(&token);
            status = next_status;
        }
        assert!(!text.is_empty());
    }
}
