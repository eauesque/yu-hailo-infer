use std::ffi::{c_char, c_float, c_int, CString};
use std::ptr::NonNull;

use super::{check_status, ffi, HailoRtError, HailoRtResult};

enum YuHailortSpeech2Text {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Speech2TextTask {
    Transcribe = 0,
    Translate = 1,
}

extern "C" {
    fn yu_hailort_s2t_create(
        model_path: *const c_char,
        out: *mut *mut YuHailortSpeech2Text,
    ) -> ffi::HailoStatus;
    fn yu_hailort_s2t_release(ctx: *mut YuHailortSpeech2Text);
    fn yu_hailort_s2t_generate_text(
        ctx: *mut YuHailortSpeech2Text,
        audio: *const c_float,
        audio_count: usize,
        task: c_int,
        language: *const c_char,
        repetition_penalty: c_float,
        timeout_ms: u32,
        out_text: *mut *mut c_char,
    ) -> ffi::HailoStatus;
    fn yu_hailort_s2t_tokenize(
        ctx: *mut YuHailortSpeech2Text,
        text: *const c_char,
        tokens: *mut c_int,
        tokens_count: *mut usize,
    ) -> ffi::HailoStatus;
    fn yu_hailort_string_free(value: *mut c_char);
}

pub(crate) struct Speech2Text {
    raw: NonNull<YuHailortSpeech2Text>,
}

impl Speech2Text {
    pub(crate) fn create(model_path: &str) -> HailoRtResult<Self> {
        let model_path = CString::new(model_path)?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: model_path is a valid C string and raw out pointer is valid.
        let status = unsafe { yu_hailort_s2t_create(model_path.as_ptr(), &mut raw) };
        check_status("s2t_create", status)?;
        let raw =
            NonNull::new(raw).ok_or(HailoRtError::InvalidMetadata("s2t_create returned null"))?;
        Ok(Self { raw })
    }

    pub(crate) fn generate_text(
        &mut self,
        audio: &[f32],
        task: Speech2TextTask,
        language: Option<&str>,
        repetition_penalty: f32,
        timeout_ms: u32,
    ) -> HailoRtResult<String> {
        let language = CString::new(language.unwrap_or(""))?;
        let mut out = std::ptr::null_mut();
        // SAFETY: audio slice and language C string live until call returns; out pointer is valid.
        let status = unsafe {
            yu_hailort_s2t_generate_text(
                self.raw.as_ptr(),
                audio.as_ptr(),
                audio.len(),
                task as c_int,
                language.as_ptr(),
                repetition_penalty,
                timeout_ms,
                &mut out,
            )
        };
        check_status("s2t_generate_text", status)?;
        take_c_string(out)
    }

    pub(crate) fn tokenize(&mut self, text: &str) -> HailoRtResult<Vec<i32>> {
        let text = CString::new(text)?;
        let mut count = 0usize;
        // SAFETY: First call asks shim for required count.
        let status = unsafe {
            yu_hailort_s2t_tokenize(
                self.raw.as_ptr(),
                text.as_ptr(),
                std::ptr::null_mut(),
                &mut count,
            )
        };
        if status != ffi::HAILO_INSUFFICIENT_BUFFER {
            check_status("s2t_tokenize_count", status)?;
        }
        let mut tokens = vec![0; count];
        // SAFETY: tokens buffer has count entries requested by shim.
        let status = unsafe {
            yu_hailort_s2t_tokenize(
                self.raw.as_ptr(),
                text.as_ptr(),
                tokens.as_mut_ptr(),
                &mut count,
            )
        };
        check_status("s2t_tokenize", status)?;
        tokens.truncate(count);
        Ok(tokens)
    }
}

impl Drop for Speech2Text {
    fn drop(&mut self) {
        // SAFETY: raw is owned by this RAII wrapper and released once here.
        unsafe { yu_hailort_s2t_release(self.raw.as_ptr()) };
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
    fn speech2text_rejects_paths_with_nul_bytes() {
        let err = match Speech2Text::create("bad\0path") {
            Ok(_) => panic!("expected nul byte error"),
            Err(err) => err,
        };
        assert!(matches!(err, HailoRtError::Nul(_)));
    }

    #[test]
    #[ignore = "requires a Hailo device (/dev/hailo0 or /dev/h1x-0, depending on driver generation) and HAILO_S2T_HEF"]
    fn smoke_speech2text_create_and_tokenize() {
        let hef = std::env::var("HAILO_S2T_HEF").expect("HAILO_S2T_HEF");
        let mut s2t = Speech2Text::create(&hef).expect("create speech2text");
        let tokens = s2t.tokenize("hello").expect("tokenize");
        assert!(!tokens.is_empty());
    }
}
