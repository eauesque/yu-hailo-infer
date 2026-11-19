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
    #[error("GenAI HEF {requested} conflicts with resident HEF {loaded}")]
    GenAiConflict { requested: String, loaded: String },
}

impl HailoRtError {
    pub(crate) fn status(context: &'static str, code: ffi::HailoStatus) -> Self {
        super::session::observe(code);
        Self::Status { context, code }
    }

    pub(crate) fn status_code(&self) -> Option<ffi::HailoStatus> {
        match self {
            Self::Status { code, .. } => Some(*code),
            Self::Nul(_) | Self::InvalidMetadata(_) | Self::GenAiConflict { .. } => None,
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

// Status observation and process recovery share one process-wide state.
pub(crate) mod session {
    //! Process-wide HailoRT session loss; only a fresh process can recover.
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::super::ffi;

    static LOST: AtomicBool = AtomicBool::new(false);
    static SHUTDOWN: tokio::sync::Notify = tokio::sync::Notify::const_new();
    pub(crate) const EXIT_CODE: i32 = 70; // EX_SOFTWARE
    pub(crate) const LOST_MESSAGE: &str = "HailoRT session lost; sidecar restart required";

    extern "C" {
        fn yu_hailort_is_communication_closed(status: ffi::HailoStatus) -> bool;
    }

    pub(crate) fn is_lost() -> bool {
        LOST.load(Ordering::Acquire)
    }

    pub(super) fn observe(status: ffi::HailoStatus) {
        // SAFETY: the shim compares SDK enum names and does not dereference pointers.
        if unsafe { yu_hailort_is_communication_closed(status) }
            && !LOST.swap(true, Ordering::AcqRel)
        {
            tracing::error!("HailoRT session lost; scheduling graceful exit 70 (EX_SOFTWARE)");
            SHUTDOWN.notify_one();
        }
    }

    /// The exit hook runs only after HTTP drains, including the triggering response.
    pub(crate) async fn serve(
        listener: tokio::net::TcpListener,
        app: axum::Router,
        exit: impl FnOnce(i32),
    ) {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                SHUTDOWN.notified().await;
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            })
            .await
            .expect("yu-infer server error");
        exit(EXIT_CODE);
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
