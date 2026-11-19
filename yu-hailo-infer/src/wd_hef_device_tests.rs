use super::*;
use crate::hailort::stub_counts;
use crate::hailort::wd_hef::tests::digest;
use crate::hailort::wd_hef::tests::{stub::*, TempDir};

#[test]
fn wd_panic_evicts_only_wd_even_at_discard_budget() {
    if isolated("router::wd_hef_tests::wd_panic_evicts_only_wd_even_at_discard_budget") {
        return;
    }
    let dir = TempDir::new();
    let path = dir.file();
    let mut ctx = DeviceCtx {
        panic_discards: MAX_PANIC_HANDLE_DISCARDS,
        ..DeviceCtx::default()
    };
    let result: Result<(), HailoTaskError> = ctx.run(|ctx| {
        ctx.with_wd_hef(
            &path,
            digest(&path),
            |_, _| Ok(()),
            |_| -> Result<(), _> {
                panic!("WD run panic");
            },
        )
        .unwrap();
        Ok(())
    });
    assert!(matches!(result, Err(HailoTaskError::Panicked)));
    assert_eq!(counts()[4], 1);
    assert!(!ctx.degraded);
    let clip = ctx.infer_model("clip.hef").unwrap() as *mut _;
    let yolo = ctx.infer_model("yolo.hef").unwrap() as *mut _;
    assert_eq!(stub_counts().vdevice_create, 1);
    ctx.run(|ctx| {
        ctx.with_wd_hef(&path, digest(&path), |_, _| Ok(()), |_| Ok(()))
            .unwrap();
        Ok(())
    })
    .unwrap();
    assert_eq!(counts()[1], 2);
    // A panic while validating a replacement must also clear a poisoned WD mutex.
    std::fs::write(&path, [1]).unwrap();
    let result: Result<(), HailoTaskError> = ctx.run(|ctx| {
        ctx.with_wd_hef(
            &path,
            digest(&path),
            |_, _| panic!("WD metadata panic"),
            |_| Ok(()),
        )
        .unwrap();
        Ok(())
    });
    assert!(matches!(result, Err(HailoTaskError::Panicked)));
    assert_eq!(counts()[4], counts()[6]);
    assert_eq!(ctx.infer_model("clip.hef").unwrap() as *mut _, clip);
    assert_eq!(ctx.infer_model("yolo.hef").unwrap() as *mut _, yolo);
    assert_eq!(stub_counts().yolo_release, 0);
    assert_eq!(stub_counts().vdevice_create, 1);
    ctx.run(|ctx| {
        ctx.with_wd_hef(&path, digest(&path), |_, _| Ok(()), |_| Ok(()))
            .unwrap();
        Ok(())
    })
    .unwrap();
}

#[test]
fn wd_open_and_configure_failures_preserve_clip_yolo_and_device() {
    if isolated(
        "router::wd_hef_tests::wd_open_and_configure_failures_preserve_clip_yolo_and_device",
    ) {
        return;
    }
    let dir = TempDir::new();
    let path = dir.file();
    let mut ctx = DeviceCtx::default();
    let clip = ctx.infer_model("clip.hef").unwrap() as *mut _;
    let yolo = ctx.infer_model("yolo.hef").unwrap() as *mut _;
    for stage in [1, 3] {
        fail(0);
        ctx.run(|ctx| {
            ctx.with_wd_hef(&path, digest(&path), |_, _| Ok(()), |_| Ok(()))
                .unwrap();
            Ok(())
        })
        .unwrap();
        std::fs::write(&path, vec![7; stage as usize + 10]).unwrap();
        fail(stage);
        let result =
            ctx.run(|ctx| Ok(ctx.with_wd_hef(&path, digest(&path), |_, _| Ok(()), |_| Ok(()))));
        assert!(result.unwrap().is_err());
        assert_eq!(counts()[4], counts()[6]);
        assert_eq!(ctx.infer_model("clip.hef").unwrap() as *mut _, clip);
        assert_eq!(ctx.infer_model("yolo.hef").unwrap() as *mut _, yolo);
        // Also exercise creation of a new CLIP handle after each WD failure.
        ctx.infer_model(&format!("clip-after-{stage}.hef")).unwrap();
        assert_eq!(stub_counts().yolo_release, 0);
        assert_eq!(stub_counts().vdevice_create, 1);
        assert!(!ctx.degraded);
    }
}
