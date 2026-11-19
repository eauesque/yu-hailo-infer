use super::*;
use std::ffi::CStr;

extern "C" {
    fn yu_hailort_stub_wd_open_status(status: i32);
    fn yu_hailort_stub_wd_invalid_hef_status(index: usize) -> i32;
    fn yu_hailort_stub_wd_fail(stage: i32);
    fn yu_hailort_stub_wd_count(event: usize) -> usize;
    fn yu_hailort_stub_wd_calls() -> *const std::ffi::c_char;
    fn yu_hailort_stub_vdevice_fail_initial();
}

#[test]
fn revfix_open_hef_statuses_are_invalid() {
    if isolated("revfix_open_hef_statuses_are_invalid") {
        return;
    }
    for index in 0..6 {
        // SAFETY: stub-only status injection, isolated process and valid index.
        unsafe {
            yu_hailort_stub_wd_open_status(yu_hailort_stub_wd_invalid_hef_status(index));
        }
        assert!(
            matches!(OpenedWd::open(&[7]), Err(WdError::InvalidHef)),
            "index={index}"
        );
    }
    // SAFETY: stub-only status injection on this thread.
    unsafe {
        yu_hailort_stub_wd_open_status(1);
    }
    assert!(matches!(OpenedWd::open(&[7]), Err(WdError::HailoRt(_))));
}

#[test]
fn revfix_exceptions_do_not_cross_wd_abi() {
    if isolated("revfix_exceptions_do_not_cross_wd_abi") {
        return;
    }
    fail(20);
    assert!(matches!(OpenedWd::open(&[7]), Err(WdError::HailoRt(_))));
    fail(0);
    let opened = OpenedWd::open(&[7]).unwrap();
    fail(21);
    assert!(matches!(opened.metadata(), Err(WdError::HailoRt(_))));
    fail(0);
    opened.metadata().unwrap();
    fail(22);
    assert!(matches!(opened.configure(), Err(WdError::HailoRt(_))));
    assert_eq!(counts()[4], counts()[6]);
    fail(0);
    let opened = OpenedWd::open(&[7]).unwrap();
    opened.metadata().unwrap();
    let mut model = opened.configure().unwrap();
    fail(23);
    assert!(matches!(
        model.run(
            &vec![7; 602_112],
            &mut [("output", &mut vec![0; 10_861])],
            1000
        ),
        Err(WdError::HailoRt(_))
    ));
    fail(24);
    drop(model);
    assert_eq!(counts()[4], counts()[6]);
}

// Each scenario starts a fresh process: VDevice caches success/failure for its lifetime.
fn isolated(name: &str) -> bool {
    if std::env::var("YU_WD_STUB_CHILD").as_deref() == Ok(name) {
        return false;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("hailort::wd::tests::{name}"),
            "--nocapture",
        ])
        .env("YU_WD_STUB_CHILD", name)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
    true
}

fn fail(stage: i32) {
    // SAFETY: stub-only setter; this test runs alone in its subprocess.
    unsafe { yu_hailort_stub_wd_fail(stage) };
}

fn counts() -> [usize; 7] {
    // SAFETY: stub-only counters, all indices are in range.
    std::array::from_fn(|event| unsafe { yu_hailort_stub_wd_count(event) })
}

fn calls() -> String {
    // SAFETY: the stub returns a live, nul-terminated string, copied before another call.
    unsafe { CStr::from_ptr(yu_hailort_stub_wd_calls()) }
        .to_str()
        .unwrap()
        .into()
}

#[test]
fn open_metadata_never_configures() {
    if isolated("open_metadata_never_configures") {
        return;
    }
    let bytes = vec![7, 8, 9];
    let opened = OpenedWd::open(&bytes).unwrap();
    drop(bytes);
    let (inputs, outputs) = opened.metadata().unwrap();
    assert_eq!(inputs[0].frame_size, 602_112);
    assert_eq!(outputs[0].frame_size, 10_861);
    assert_eq!(counts()[3], 0);
    assert_eq!(calls(), "open,metadata");
    drop(opened);
    assert_eq!(calls(), "open,metadata,release");
    assert_eq!(counts()[4], 1);
}

#[test]
fn rejected_metadata_drops_without_configure() {
    if isolated("rejected_metadata_drops_without_configure") {
        return;
    }
    fail(4);
    let opened = OpenedWd::open(&[7]).unwrap();
    let (inputs, _) = opened.metadata().unwrap();
    assert_ne!(inputs[0].format_order, ffi::HAILO_FORMAT_ORDER_NHWC);
    drop(opened);
    assert_eq!(counts()[3], 0);
    assert_eq!(counts()[4], 1);
    assert_eq!(calls(), "open,metadata,release");
}

#[test]
fn configure_consumes_and_configured_runs() {
    if isolated("configure_consumes_and_configured_runs") {
        return;
    }
    // This fn pointer rejects an accidental change to configure(&self) or &mut self.
    let configure: fn(OpenedWd) -> Result<ConfiguredWd, WdError> = OpenedWd::configure;
    let opened = OpenedWd::open(&[7]).unwrap();
    opened.metadata().unwrap();
    let mut configured = configure(opened).unwrap();
    let mut output = vec![0; 10_861];
    configured
        .run(&vec![7; 602_112], &mut [("output", &mut output)], 1000)
        .unwrap();
    assert_eq!(output, vec![7; 10_861]);
    drop(configured);
    assert_eq!(calls(), "open,metadata,configure,run,release");
    assert_eq!(counts()[6], counts()[4]);
}

#[test]
fn configure_failure_releases_and_is_distinguishable() {
    if isolated("configure_failure_releases_and_is_distinguishable") {
        return;
    }
    for stage in [3, 5] {
        fail(stage);
        let opened = OpenedWd::open(&[7]).unwrap();
        opened.metadata().unwrap();
        let err = opened.configure().err().unwrap();
        if stage == 3 {
            assert!(matches!(err, WdError::InvalidHef));
        } else {
            assert!(matches!(err, WdError::HailoRt(HailoRtError::Status { .. })));
        }
        assert_eq!(counts()[4], counts()[6]);
    }
    assert_eq!(
        calls(),
        "open,metadata,configure,release,open,metadata,configure,release"
    );
}

#[test]
fn wd_failures_preserve_healthy_shared_vdevice() {
    if isolated("wd_failures_preserve_healthy_shared_vdevice") {
        return;
    }
    for stage in [1, 2, 3] {
        fail(stage);
        match OpenedWd::open(&[7]) {
            Err(_) => assert_eq!(stage, 1),
            Ok(opened) => {
                if stage == 2 {
                    assert!(opened.metadata().is_err());
                } else {
                    opened.metadata().unwrap();
                    assert!(opened.configure().is_err());
                }
            }
        }
        let yolo = super::super::ShimYolo::create("model").unwrap();
        drop(yolo);
        assert_eq!(super::super::stub_counts().vdevice_create, 1);
        assert_eq!(counts()[4], counts()[6]);
    }
    assert_eq!(
        calls(),
        "open,open,metadata,release,open,metadata,configure,release"
    );
}

#[test]
fn initial_vdevice_failure_stays_cached() {
    if isolated("initial_vdevice_failure_stays_cached") {
        return;
    }
    // SAFETY: this process has not touched the shared VDevice yet.
    unsafe { yu_hailort_stub_vdevice_fail_initial() };
    assert!(OpenedWd::open(&[7]).is_err());
    assert!(OpenedWd::open(&[7]).is_err());
    assert!(super::super::ShimYolo::create("model").is_err());
    let c = super::super::stub_counts();
    assert_eq!(c.vdevice_create, 1);
    assert_eq!(c.yolo_create, 0);
    assert_eq!(counts()[6], 0);
}

#[test]
fn run_rejects_bad_buffers_and_releases() {
    if isolated("run_rejects_bad_buffers_and_releases") {
        return;
    }
    let opened = OpenedWd::open(&[7]).unwrap();
    opened.metadata().unwrap();
    let mut configured = opened.configure().unwrap();
    let mut output = vec![0; 10_861];
    assert!(configured
        .run(&[7], &mut [("output", &mut output)], 1000)
        .is_err());
    assert!(configured
        .run(&vec![7; 602_112], &mut [("output", &mut [0])], 1000)
        .is_err());
    drop(configured);
    assert_eq!(counts()[4], counts()[6]);
}
