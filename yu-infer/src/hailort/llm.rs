use std::ffi::{c_char, c_int, CString};
use std::ptr::NonNull;

use super::{check_status, ffi, HailoRtError, HailoRtResult};

enum YuHailortLlm {}
enum YuHailortLlmStream {}

extern "C" {
    fn yu_hailort_llm_create(
        model_path: *const c_char,
        lora_name: *const c_char,
        optimize_memory_on_device: bool,
        out: *mut *mut YuHailortLlm,
    ) -> ffi::HailoStatus;
    fn yu_hailort_llm_release(ctx: *mut YuHailortLlm);
    fn yu_hailort_llm_generate_text(
        ctx: *mut YuHailortLlm,
        prompt: *const c_char,
        timeout_ms: u32,
        out_text: *mut *mut c_char,
    ) -> ffi::HailoStatus;
    fn yu_hailort_llm_tokenize(
        ctx: *mut YuHailortLlm,
        text: *const c_char,
        tokens: *mut c_int,
        tokens_count: *mut usize,
    ) -> ffi::HailoStatus;
    fn yu_hailort_llm_context_usage(ctx: *mut YuHailortLlm, out: *mut usize) -> ffi::HailoStatus;
    fn yu_hailort_llm_max_context_capacity(
        ctx: *mut YuHailortLlm,
        out: *mut usize,
    ) -> ffi::HailoStatus;
    fn yu_hailort_llm_clear_context(ctx: *mut YuHailortLlm) -> ffi::HailoStatus;
    fn yu_hailort_llm_generate_stream_start(
        ctx: *mut YuHailortLlm,
        messages_json: *const *const c_char,
        messages_count: usize,
        temperature: *const f32,
        top_p: *const f32,
        top_k: *const u32,
        frequency_penalty: *const f32,
        max_generated_tokens: *const u32,
        do_sample: *const bool,
        seed: *const u32,
        out: *mut *mut YuHailortLlmStream,
    ) -> ffi::HailoStatus;
    fn yu_hailort_llm_stream_read(
        stream: *mut YuHailortLlmStream,
        timeout_ms: u32,
        out_token: *mut *mut c_char,
        out_status: *mut c_int,
    ) -> ffi::HailoStatus;
    fn yu_hailort_llm_stream_release(stream: *mut YuHailortLlmStream);
    fn yu_hailort_string_free(value: *mut c_char);
}

/// One chat turn (`role` is e.g. "system"/"user"/"assistant"). A full
/// multi-turn conversation is expressed as an ordered slice of these,
/// letting the HailoRT SDK's chat template render the whole exchange rather
/// than a single flattened prompt string.
#[derive(Debug, Clone)]
pub(crate) struct LlmChatMessage {
    pub(crate) role: String,
    pub(crate) content: String,
}

/// Overrides for LLM generation sampling. `None` fields keep the model's
/// default value (queried on-device via `create_generator_params()`).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct LlmGenerationParams {
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<u32>,
    pub(crate) frequency_penalty: Option<f32>,
    pub(crate) max_generated_tokens: Option<u32>,
    pub(crate) do_sample: Option<bool>,
    pub(crate) seed: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LlmCompletionStatus {
    Generating,
    MaxTokensReached,
    LogicalEndOfGeneration,
    Aborted,
}

impl LlmCompletionStatus {
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

pub(crate) struct Llm {
    raw: NonNull<YuHailortLlm>,
}

impl Llm {
    pub(crate) fn create(
        model_path: &str,
        lora_name: Option<&str>,
        optimize_memory_on_device: bool,
    ) -> HailoRtResult<Self> {
        let model_path = CString::new(model_path)?;
        let lora_name = CString::new(lora_name.unwrap_or(""))?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: C strings and out pointer are valid until call returns.
        let status = unsafe {
            yu_hailort_llm_create(
                model_path.as_ptr(),
                lora_name.as_ptr(),
                optimize_memory_on_device,
                &mut raw,
            )
        };
        check_status("llm_create", status)?;
        let raw =
            NonNull::new(raw).ok_or(HailoRtError::InvalidMetadata("llm_create returned null"))?;
        Ok(Self { raw })
    }

    pub(crate) fn generate_text(&mut self, prompt: &str, timeout_ms: u32) -> HailoRtResult<String> {
        let prompt = CString::new(prompt)?;
        let mut out = std::ptr::null_mut();
        // SAFETY: prompt C string and out pointer are valid until call returns.
        let status = unsafe {
            yu_hailort_llm_generate_text(self.raw.as_ptr(), prompt.as_ptr(), timeout_ms, &mut out)
        };
        check_status("llm_generate_text", status)?;
        take_c_string(out)
    }

    pub(crate) fn tokenize(&mut self, text: &str) -> HailoRtResult<Vec<i32>> {
        let text = CString::new(text)?;
        let mut count = 0usize;
        // SAFETY: First call asks shim for required count.
        let status = unsafe {
            yu_hailort_llm_tokenize(
                self.raw.as_ptr(),
                text.as_ptr(),
                std::ptr::null_mut(),
                &mut count,
            )
        };
        if status != ffi::HAILO_INSUFFICIENT_BUFFER {
            check_status("llm_tokenize_count", status)?;
        }
        let mut tokens = vec![0; count];
        // SAFETY: tokens buffer has count entries requested by shim.
        let status = unsafe {
            yu_hailort_llm_tokenize(
                self.raw.as_ptr(),
                text.as_ptr(),
                tokens.as_mut_ptr(),
                &mut count,
            )
        };
        check_status("llm_tokenize", status)?;
        tokens.truncate(count);
        Ok(tokens)
    }

    pub(crate) fn context_usage(&mut self) -> HailoRtResult<usize> {
        let mut out = 0usize;
        // SAFETY: out pointer is valid.
        let status = unsafe { yu_hailort_llm_context_usage(self.raw.as_ptr(), &mut out) };
        check_status("llm_context_usage", status)?;
        Ok(out)
    }

    pub(crate) fn max_context_capacity(&mut self) -> HailoRtResult<usize> {
        let mut out = 0usize;
        // SAFETY: out pointer is valid.
        let status = unsafe { yu_hailort_llm_max_context_capacity(self.raw.as_ptr(), &mut out) };
        check_status("llm_max_context_capacity", status)?;
        Ok(out)
    }

    pub(crate) fn clear_context(&mut self) -> HailoRtResult<()> {
        // SAFETY: raw context is valid for self lifetime.
        let status = unsafe { yu_hailort_llm_clear_context(self.raw.as_ptr()) };
        check_status("llm_clear_context", status)
    }
}

impl Drop for Llm {
    fn drop(&mut self) {
        // SAFETY: raw is owned by this RAII wrapper and released once here.
        unsafe { yu_hailort_llm_release(self.raw.as_ptr()) };
    }
}

/// A single in-flight LLM generation, read one token at a time.
///
/// Call [`Self::read_next`] in a loop until the returned status is no longer
/// [`LlmCompletionStatus::Generating`] — per the HailoRT SDK contract, no
/// further reads may be attempted once generation has ended.
///
/// Borrows the source [`Llm`] mutably for its entire lifetime, mirroring
/// `VlmStream`: the SDK documents that only one generator/completion may be
/// active on a model at a time, so this stream must hold exclusive access to
/// `llm` until it is dropped.
pub(crate) struct LlmStream<'a> {
    raw: NonNull<YuHailortLlmStream>,
    _llm: std::marker::PhantomData<&'a mut Llm>,
}

impl<'a> LlmStream<'a> {
    pub(crate) fn start(
        llm: &'a mut Llm,
        messages: &[LlmChatMessage],
        params: LlmGenerationParams,
    ) -> HailoRtResult<Self> {
        let message_jsons = messages
            .iter()
            .map(|message| {
                let json = serde_json::json!({"role": message.role, "content": message.content})
                    .to_string();
                CString::new(json)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let message_ptrs: Vec<*const c_char> =
            message_jsons.iter().map(|value| value.as_ptr()).collect();
        let mut out = std::ptr::null_mut();
        // SAFETY: message C strings, optional-override pointers, and out
        // pointer are all valid until the call returns.
        let status = unsafe {
            yu_hailort_llm_generate_stream_start(
                llm.raw.as_ptr(),
                message_ptrs.as_ptr(),
                message_ptrs.len(),
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
        check_status("llm_generate_stream_start", status)?;
        let raw = NonNull::new(out).ok_or(HailoRtError::InvalidMetadata(
            "llm_generate_stream_start returned null",
        ))?;
        Ok(Self {
            raw,
            _llm: std::marker::PhantomData,
        })
    }

    pub(crate) fn read_next(
        &mut self,
        timeout_ms: u32,
    ) -> HailoRtResult<(String, LlmCompletionStatus)> {
        let mut out_token = std::ptr::null_mut();
        let mut out_status: c_int = 0;
        // SAFETY: out pointers are valid stack locations for the duration of the call.
        let status = unsafe {
            yu_hailort_llm_stream_read(
                self.raw.as_ptr(),
                timeout_ms,
                &mut out_token,
                &mut out_status,
            )
        };
        check_status("llm_stream_read", status)?;
        let token = take_c_string(out_token)?;
        Ok((token, LlmCompletionStatus::from_raw(out_status)))
    }
}

impl Drop for LlmStream<'_> {
    fn drop(&mut self) {
        // SAFETY: raw is owned by this RAII wrapper and released once here.
        unsafe { yu_hailort_llm_stream_release(self.raw.as_ptr()) };
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
    fn llm_rejects_paths_with_nul_bytes() {
        let err = match Llm::create("bad\0path", None, false) {
            Ok(_) => panic!("expected nul byte error"),
            Err(err) => err,
        };
        assert!(matches!(err, HailoRtError::Nul(_)));
    }

    #[test]
    #[ignore = "requires /dev/hailo0 and HAILO_LLM_HEF"]
    fn smoke_llm_create_and_tokenize() {
        let hef = std::env::var("HAILO_LLM_HEF").expect("HAILO_LLM_HEF");
        let mut llm = Llm::create(&hef, None, false).expect("create llm");
        let tokens = llm.tokenize("hello").expect("tokenize");
        assert!(!tokens.is_empty());
        assert!(llm.max_context_capacity().expect("capacity") > 0);
    }

    #[test]
    #[ignore = "requires /dev/hailo0 and HAILO_LLM_HEF"]
    fn smoke_llm_generate_stream_accepts_all_generation_params() {
        // Same regression coverage as the VLM equivalent: HailoRT rejects an
        // explicit 0.0 frequency_penalty, so this exercises every
        // LlmGenerationParams field together, including frequency_penalty: 0.0.
        let hef = std::env::var("HAILO_LLM_HEF").expect("HAILO_LLM_HEF");
        let mut llm = Llm::create(&hef, None, false).expect("create llm");
        let mut stream = LlmStream::start(
            &mut llm,
            &[
                LlmChatMessage {
                    role: "system".to_string(),
                    content: "You are a helpful assistant. Answer in one word.".to_string(),
                },
                LlmChatMessage {
                    role: "user".to_string(),
                    content: "What is the capital of France?".to_string(),
                },
            ],
            LlmGenerationParams {
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
        let mut status = LlmCompletionStatus::Generating;
        while status.is_generating() {
            let (token, next_status) = stream.read_next(60_000).expect("stream_read");
            eprintln!("LLM stream token: {token:?} status: {next_status:?}");
            text.push_str(&token);
            status = next_status;
        }
        assert!(!text.is_empty());
    }
}
