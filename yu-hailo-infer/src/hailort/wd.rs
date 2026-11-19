//! WD loads validated sidecar-owned bytes before the caller checks metadata.
use std::ffi::CString;
use std::ptr::NonNull;

use super::shim::{convert_infos, YuHailortBuffer, YuHailortYoloMetadata};
use super::{check_status, ffi, HailoRtError, VStreamDirection, VStreamInfo};

#[derive(Debug, thiserror::Error)]
pub(crate) enum WdError {
    #[error("WD HEF rejected")]
    InvalidHef,
    #[error("WD HEF deadline expired")]
    Timeout,
    #[error(transparent)]
    HailoRt(#[from] HailoRtError),
}

enum YuHailortWd {}

extern "C" {
    fn yu_hailort_wd_is_timeout(status: ffi::HailoStatus) -> bool;
    fn yu_hailort_wd_is_invalid_hef(status: ffi::HailoStatus) -> bool;
    fn yu_hailort_wd_open(
        hef: *const u8,
        size: usize,
        out: *mut *mut YuHailortWd,
    ) -> ffi::HailoStatus;
    fn yu_hailort_wd_metadata(
        ctx: *mut YuHailortWd,
        out: *mut YuHailortYoloMetadata,
    ) -> ffi::HailoStatus;
    fn yu_hailort_wd_configure(
        ctx: *mut YuHailortWd,
        out: *mut *mut YuHailortWd,
    ) -> ffi::HailoStatus;
    fn yu_hailort_wd_run(
        ctx: *mut YuHailortWd,
        input: *const u8,
        size: usize,
        outputs: *mut YuHailortBuffer,
        count: usize,
        timeout_ms: u32,
    ) -> ffi::HailoStatus;
    fn yu_hailort_wd_release(ctx: *mut YuHailortWd);
}

struct Handle(NonNull<YuHailortWd>);

impl Handle {
    fn from_raw(raw: *mut YuHailortWd) -> Result<Self, WdError> {
        NonNull::new(raw)
            .map(Self)
            .ok_or_else(|| HailoRtError::InvalidMetadata("WD shim returned null").into())
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: this wrapper uniquely owns the live native handle.
        unsafe { yu_hailort_wd_release(self.0.as_ptr()) };
    }
}

pub(crate) struct OpenedWd(Handle);
pub(crate) struct ConfiguredWd(Option<Handle>);

impl OpenedWd {
    /// HailoRT 5.4.0 copies the HEF during open; `hef` may be freed on return.
    pub(crate) fn open(hef: &[u8]) -> Result<Self, WdError> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: hef and the out pointer remain valid throughout the call.
        let status = unsafe { yu_hailort_wd_open(hef.as_ptr(), hef.len(), &mut raw) };
        // SAFETY: the shim compares status with SDK enum names, without dereferencing pointers.
        if unsafe { yu_hailort_wd_is_invalid_hef(status) } {
            return Err(WdError::InvalidHef);
        }
        check_status("wd_open", status)?;
        Ok(Self(Handle::from_raw(raw)?))
    }

    pub(crate) fn metadata(&self) -> Result<(Vec<VStreamInfo>, Vec<VStreamInfo>), WdError> {
        let mut metadata = YuHailortYoloMetadata::default();
        // SAFETY: self owns the live handle and metadata is writable POD.
        let status = unsafe { yu_hailort_wd_metadata(self.0 .0.as_ptr(), &mut metadata) };
        check_status("wd_metadata", status)?;
        if metadata.inputs_count > metadata.inputs.len()
            || metadata.outputs_count > metadata.outputs.len()
        {
            return Err(HailoRtError::InvalidMetadata("WD shim returned too many vstreams").into());
        }
        Ok((
            convert_infos(
                &metadata.inputs[..metadata.inputs_count],
                VStreamDirection::Input,
            ),
            convert_infos(
                &metadata.outputs[..metadata.outputs_count],
                VStreamDirection::Output,
            ),
        ))
    }

    /// Consumes the opened handle; failure also releases all model resources.
    pub(crate) fn configure(self) -> Result<ConfiguredWd, WdError> {
        let raw = self.0 .0.as_ptr();
        std::mem::forget(self);
        let mut configured = std::ptr::null_mut();
        // SAFETY: ownership transfers to the shim, which consumes raw on every path.
        let status = unsafe { yu_hailort_wd_configure(raw, &mut configured) };
        if status == ffi::HAILO_INVALID_ARGUMENT {
            return Err(WdError::InvalidHef);
        }
        check_status("wd_configure", status)?;
        Ok(ConfiguredWd(Some(Handle::from_raw(configured)?)))
    }
}

impl ConfiguredWd {
    pub(crate) fn run(
        &mut self,
        input: &[u8],
        outputs: &mut [(&str, &mut [u8])],
        timeout_ms: u32,
    ) -> Result<(), WdError> {
        let handle = self.0.as_ref().ok_or(WdError::Timeout)?;
        let names = outputs
            .iter()
            .map(|(name, _)| CString::new(*name))
            .collect::<Result<Vec<_>, _>>()
            .map_err(HailoRtError::from)?;
        let mut buffers = outputs
            .iter_mut()
            .zip(&names)
            .map(|((_, data), name)| YuHailortBuffer {
                name: name.as_ptr(),
                data: data.as_mut_ptr().cast(),
                size: data.len(),
            })
            .collect::<Vec<_>>();
        // SAFETY: buffers stay borrowed through run and model release on timeout below.
        let status = unsafe {
            yu_hailort_wd_run(
                handle.0.as_ptr(),
                input.as_ptr(),
                input.len(),
                buffers.as_mut_ptr(),
                buffers.len(),
                timeout_ms,
            )
        };
        // SAFETY: the shim compares the integer with the SDK's enum, without dereferencing pointers.
        if unsafe { yu_hailort_wd_is_timeout(status) } {
            // A timed-out job may still reference these buffers: release the model before returning them.
            drop(self.0.take());
            return Err(WdError::Timeout);
        }
        check_status("wd_run", status)?;
        Ok(())
    }
}

#[cfg(all(test, hailo_stub))]
#[path = "wd_tests.rs"]
mod tests;
