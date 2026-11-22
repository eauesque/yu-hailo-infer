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

pub(super) fn check_teardown_status(
    context: &'static str,
    code: ffi::HailoStatus,
) -> HailoRtResult<()> {
    if code == ffi::HAILO_SUCCESS {
        Ok(())
    } else {
        Err(HailoRtError::Status { context, code })
    }
}

// Status observation and process recovery share one process-wide state.
pub(crate) mod session {
    //! Process-wide HailoRT session loss; only a fresh process can recover.
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};
    use tokio::{sync::oneshot, time};

    use super::super::ffi;

    static LOST: AtomicBool = AtomicBool::new(false);
    static SHUTDOWN: tokio::sync::Notify = tokio::sync::Notify::const_new();
    #[cfg(test)]
    static WATCHDOG_ENABLED: AtomicBool = AtomicBool::new(false);
    pub(crate) const EXIT_CODE: i32 = 70; // EX_SOFTWARE
    pub(crate) const LOST_MESSAGE: &str = "HailoRT session lost; sidecar restart required";
    const DRAIN_DEADLINE: Duration = Duration::from_secs(5);

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
            arm_watchdog();
            SHUTDOWN.notify_one();
            tracing::error!("HailoRT session lost; scheduling graceful exit 70 (EX_SOFTWARE)");
        }
    }

    #[cfg(test)]
    pub(crate) fn enable_test_watchdog() {
        WATCHDOG_ENABLED.store(true, Ordering::Release);
    }

    fn arm_watchdog() {
        #[cfg(test)]
        if !WATCHDOG_ENABLED.load(Ordering::Acquire) {
            return;
        }

        // A separate thread keeps the deadline through blocked logging, flush, or teardown.
        let armed_at = Instant::now();
        if std::thread::Builder::new()
            .spawn(move || {
                std::thread::sleep(DRAIN_DEADLINE.saturating_sub(armed_at.elapsed()));
                crate::hard_exit(EXIT_CODE);
            })
            .is_err()
        {
            crate::hard_exit(EXIT_CODE);
        }
    }

    /// Give in-flight responses up to the drain deadline before invoking the exit hook.
    pub(crate) async fn serve<L>(listener: L, app: axum::Router, exit: impl FnOnce(i32))
    where
        L: axum::serve::Listener,
        L::Addr: std::fmt::Debug,
    {
        let (stop_accepting, shutdown) = oneshot::channel();
        let server = axum::serve(listener, app).with_graceful_shutdown(async {
            let _ = shutdown.await;
        });
        let drain = async {
            SHUTDOWN.notified().await;
            let deadline = time::Instant::now() + DRAIN_DEADLINE;
            // Keep the existing window for a client to send a same-child fallback.
            time::sleep(Duration::from_millis(100)).await;
            let _ = stop_accepting.send(());
            time::sleep_until(deadline).await;
        };
        tokio::select! {
            result = server => result.expect("yu-infer server error"),
            () = drain => (),
        }
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

    #[cfg(all(hailo_stub, target_os = "linux"))]
    fn lose_session() {
        use crate::hailort::{wd::OpenedWd, wd_stub};
        wd_stub::setup(&[], 30, 0);
        let opened = OpenedWd::open(&[7, 8, 9]).unwrap();
        opened.metadata().unwrap();
        assert!(opened.configure().is_err());
        assert!(session::is_lost());
    }

    #[cfg(all(hailo_stub, target_os = "linux"))]
    #[tokio::test]
    async fn t27a_stalled_response_cannot_delay_exit() {
        use std::{sync::Arc, time::Duration};
        use tokio::{io::AsyncWriteExt, net::TcpListener, sync::Notify, time};

        if crate::hailort::wd_hef::tests::stub::isolated(
            "hailort::error::tests::t27a_stalled_response_cannot_delay_exit",
        ) {
            return;
        }
        let started = Arc::new(Notify::new());
        let handler_started = started.clone();
        let app = axum::Router::new().route(
            "/stalled",
            axum::routing::get(move || async move {
                handler_started.notify_one();
                std::future::pending::<()>().await;
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut client = tokio::net::TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let server = tokio::spawn(session::serve(listener, app, |code| assert_eq!(code, 70)));
        client
            .write_all(b"GET /stalled HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        time::timeout(Duration::from_secs(2), started.notified())
            .await
            .unwrap();

        time::pause();
        lose_session();
        time::timeout(Duration::from_millis(5100), server)
            .await
            .expect("stalled response held the exit hook past the 5 s drain deadline")
            .unwrap();
    }

    #[cfg(all(hailo_stub, target_os = "linux"))]
    mod launcher {
        use super::lose_session;
        use std::{
            future::pending,
            io::{self, Write},
            process::{Command, Output, Stdio},
            sync::Arc,
            thread,
            time::{Duration, Instant},
        };

        const CHILD: &str = "YU_T27A_EXIT_CHILD";

        struct IdleListener;

        impl axum::serve::Listener for IdleListener {
            type Io = tokio::io::DuplexStream;
            type Addr = ();

            async fn accept(&mut self) -> (Self::Io, Self::Addr) {
                pending().await
            }

            fn local_addr(&self) -> io::Result<Self::Addr> {
                Ok(())
            }
        }

        struct BlockingWriter;

        impl Write for BlockingWriter {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                eprintln!("entered loss logging");
                block_forever();
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        struct BlockingState;

        impl Drop for BlockingState {
            fn drop(&mut self) {
                eprintln!("entered router state drop");
                block_forever();
            }
        }

        fn block_forever() -> ! {
            loop {
                thread::park();
            }
        }

        // Every case runs the same launcher as the binary, with real exit semantics.
        async fn run_child(
            name: &str,
            app: axum::Router,
            trigger: impl FnOnce() + Send + 'static,
        ) -> Output {
            if std::env::var_os(CHILD).is_some() {
                tokio::spawn(async move { trigger() });
                crate::serve(IdleListener, app).await;
                panic!("production exit hook returned");
            }

            let test = format!("hailort::error::tests::launcher::{name}");
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", &test, "--nocapture"])
                .env(CHILD, "1")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            let limit = Instant::now() + Duration::from_secs(6);
            while child.try_wait().unwrap().is_none() {
                if Instant::now() >= limit {
                    child.kill().ok();
                    let output = child.wait_with_output().unwrap();
                    panic!("{name} did not exit within the drain budget plus 1 s: {output:?}");
                }
                thread::sleep(Duration::from_millis(10));
            }
            let output = child.wait_with_output().unwrap();
            assert_eq!(output.status.code(), Some(70), "{name}: {output:?}");
            output
        }

        #[tokio::test]
        async fn t27a_blocked_loss_logging_cannot_delay_exit() {
            let output = run_child(
                "t27a_blocked_loss_logging_cannot_delay_exit",
                axum::Router::new(),
                || {
                    let subscriber = tracing_subscriber::fmt()
                        .with_writer(|| BlockingWriter)
                        .finish();
                    tracing::subscriber::with_default(subscriber, lose_session);
                },
            )
            .await;
            assert!(String::from_utf8_lossy(&output.stderr).contains("entered loss logging"));
        }

        #[tokio::test]
        async fn t27a_blocked_flush_cannot_delay_exit() {
            let output = run_child(
                "t27a_blocked_flush_cannot_delay_exit",
                axum::Router::new(),
                || {
                    let (locked, ready) = std::sync::mpsc::channel();
                    thread::spawn(move || {
                        let _lock = io::stdout().lock();
                        eprintln!("locked stdout through exit");
                        locked.send(()).unwrap();
                        block_forever();
                    });
                    ready.recv().unwrap();
                    lose_session();
                },
            )
            .await;
            assert!(String::from_utf8_lossy(&output.stderr).contains("locked stdout through exit"));
        }

        #[tokio::test]
        async fn t27a_blocked_router_drop_cannot_delay_exit() {
            let name = "t27a_blocked_router_drop_cannot_delay_exit";
            // Only the child can own state whose destructor blocks.
            let app = if std::env::var_os(CHILD).is_some() {
                axum::Router::new()
                    .route("/", axum::routing::get(|| async {}))
                    .layer(axum::Extension(Arc::new(BlockingState)))
            } else {
                axum::Router::new()
            };
            let output = run_child(name, app, lose_session).await;
            assert!(String::from_utf8_lossy(&output.stderr).contains("entered router state drop"));
        }

        #[tokio::test]
        async fn t27a_exit_flushes_and_skips_atexit() {
            let output = run_child(
                "t27a_exit_flushes_and_skips_atexit",
                axum::Router::new(),
                || {
                    extern "C" fn on_atexit() {
                        // SAFETY: this callback only changes the isolated child's exit status.
                        unsafe { libc::_exit(99) }
                    }
                    // SAFETY: the callback is static and C-compatible.
                    assert_eq!(unsafe { libc::atexit(on_atexit) }, 0);
                    print!("T27a buffered stdout");
                    lose_session();
                },
            )
            .await;
            assert!(String::from_utf8_lossy(&output.stdout).contains("T27a buffered stdout"));
        }
    }
}
