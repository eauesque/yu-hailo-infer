use std::ffi::NulError;

use thiserror::Error;

use super::ffi;

pub(crate) type HailoRtResult<T> = Result<T, HailoRtError>;

#[derive(Debug, Error)]
pub(crate) enum HailoRtError {
    #[error("HailoRT {context} failed with status {code}")]
    Status {
        context: &'static str,
        code: ffi::HailoStatus,
    },
    #[error("HailoRT path contains a nul byte: {0}")]
    Nul(#[from] NulError),
    #[error("invalid HailoRT metadata: {0}")]
    InvalidMetadata(&'static str),
}

impl HailoRtError {
    pub(crate) fn status(context: &'static str, code: ffi::HailoStatus) -> Self {
        Self::Status { context, code }
    }

    pub(crate) fn status_code(&self) -> Option<ffi::HailoStatus> {
        match self {
            Self::Status { code, .. } => Some(*code),
            Self::Nul(_) | Self::InvalidMetadata(_) => None,
        }
    }
}

pub(crate) fn check_status(context: &'static str, code: ffi::HailoStatus) -> HailoRtResult<()> {
    if code == ffi::HAILO_SUCCESS {
        Ok(())
    } else {
        Err(HailoRtError::status(context, code))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_error_includes_context_and_code() {
        let err = HailoRtError::status("create_vdevice", 74);
        assert_eq!(err.status_code(), Some(74));
        assert!(err.to_string().contains("create_vdevice"));
        assert!(err.to_string().contains("74"));
    }
}
