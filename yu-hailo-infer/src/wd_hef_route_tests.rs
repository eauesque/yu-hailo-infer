use super::*;
use crate::hailort::wd_stub;
use std::io::Write;
use std::sync::Mutex;
use std::time::{Duration, Instant};

// These fixtures mutate the process-wide device queue, cache, and preprocessing budget.
fn isolated(name: &str) -> bool {
    crate::hailort::wd_hef::tests::stub::isolated(&format!("router::tests::wd_hef_route_tests::{name}"))
}

struct Fixture {
    dir: crate::hailort::wd_hef::tests::TempDir,
    state: AppState,
    body: Value,
    configured: usize,
    runs: usize,
}

impl Fixture {
    async fn new(fields: Vec<(i32, f64)>, failure: i32, delay: u32) -> Self {
        let (root, state, mut body) = wd_wire_fixture();
        std::fs::write(root.join("model/selected_tags.csv"),
            "name,category\nsafe,9\ncat,0\nalice,4\n").unwrap();
        image::RgbImage::from_pixel(8, 4, image::Rgb([10, 20, 30]))
            .save(root.join("image.png")).unwrap();
        let hef = root.join("private-model.hef");
        std::fs::write(&hef, [7, 8, 9]).unwrap();
        body["backend"] = json!("hef");
        body["hef_path"] = json!(hef);
        body["hef_sha256"] = json!("66a6757151f8ee55db127716c7e3dce0be8074b64e20eda542e5c1e46ca9c41e");
        let (configured, runs) = run_hailort_task(move |ctx| {
            ctx.wd_hef.evict();
            wd_stub::setup(&fields, failure, delay);
            assert_eq!(wd_stub::last_run(), (0, vec![]));
            Ok((wd_stub::configure_count(), wd_stub::run_count()))
        }).await.unwrap();
        Self { dir: crate::hailort::wd_hef::tests::TempDir(root), state, body, configured, runs }
    }

    async fn request(&self) -> (StatusCode, Value) {
        wd_wire_request(self.state.clone(), self.body.clone()).await
    }

    async fn configure_delta(&self) -> usize {
        run_hailort_task(|_| Ok(wd_stub::configure_count())).await.unwrap() - self.configured
    }
}

fn error_eq(actual: (StatusCode, Value), status: StatusCode, code: &str) {
    assert_eq!(actual, (status, json!({"error": code})));
}

async fn dd_route(state: AppState, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().uri(uri)
        .method(if body.is_some() { "POST" } else { "GET" })
        .header("authorization", format!("Bearer {}", state.auth_token))
        .header("content-type", "application/json")
        .body(Body::from(body.map(|b| b.to_string()).unwrap_or_default())).unwrap();
    let response = build_router(state).oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

async fn dd_counts() -> ([usize; 7], usize) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    // Inspect thread-local stub counters without submitting Hailo work after session loss.
    hailort_device_sender().try_send(Box::new(move |_| {
        tx.send((crate::hailort::wd_hef::tests::stub::counts(), crate::hailort::stub_counts().yolo_create)).unwrap();
    })).unwrap_or_else(|_| panic!("stub counter inspection queue unavailable"));
    rx.await.unwrap()
}

#[tokio::test]
async fn dd_session_lost_from_wd_rejects_clip_and_degrades_health() {
    if isolated("dd_session_lost_from_wd_rejects_clip_and_degrades_health") { return; }
    let f = Fixture::new(vec![], 30, 0).await;
    let normal = dd_route(f.state.clone(), "/healthz", None).await.1;
    let health_keys = ["clip_image_onnx", "hailo_session_lost", "hailo_stub", "instance_id",
        "ok", "text_embed", "uptime_secs", "wd_hef"];
    assert_eq!(normal.as_object().unwrap().keys().map(String::as_str).collect::<Vec<_>>(), health_keys);
    assert_eq!(normal["ok"], true);
    assert_eq!(normal["hailo_session_lost"], false);
    error_eq(f.request().await, StatusCode::SERVICE_UNAVAILABLE, "device_error");
    assert!(crate::hailort::session::is_lost());
    let before = dd_counts().await;
    let health = dd_route(f.state.clone(), "/healthz", None).await.1;
    assert_eq!(health.as_object().unwrap().keys().map(String::as_str).collect::<Vec<_>>(), health_keys);
    assert_eq!(health["ok"], false);
    assert_eq!(health["hailo_session_lost"], true);
    assert_eq!(health["wd_hef"], normal["wd_hef"]);
    let clip = dd_route(f.state.clone(), "/v1/infer/clip-image", Some(json!({"image_base64": "AA=="}))).await;
    assert_eq!(clip.0, StatusCode::SERVICE_UNAVAILABLE);
    error_eq(f.request().await, StatusCode::SERVICE_UNAVAILABLE, "device_error");
    for (uri, body) in [
        ("/v1/infer/yolo/metadata", None),
        ("/v1/infer/yolo/smoke-zero", None),
        ("/v1/infer/yolo/detect", Some(json!({"input_base64": ""}))),
        ("/v1/infer/speech2text/tokenize", Some(json!({"text": ""}))),
        ("/v1/infer/speech2text/transcribe", Some(json!({"audio_base64": ""}))),
        ("/v1/infer/llm/tokenize", Some(json!({"text": ""}))),
        ("/v1/infer/llm/generate", Some(json!({"prompt": ""}))),
        ("/v1/infer/llm/generate/stream", Some(json!({"messages": []}))),
        ("/v1/infer/vlm/generate", Some(json!({"prompt": ""}))),
        ("/v1/infer/vlm/generate/stream", Some(json!({"prompt": ""}))),
    ] {
        let (status, body) = dd_route(f.state.clone(), uri, body).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{uri}: {body}");
        assert_eq!(body["ok"], false);
    }
    // The wire fixture's ONNX model has a single output score.
    std::fs::write(f.dir.0.join("model/selected_tags.csv"), "name,category\ncat,0\n").unwrap();
    let mut onnx = f.body.clone();
    onnx["backend"] = json!("onnx");
    let (status, body) = wd_wire_request(f.state.clone(), onnx).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["backend_applied"], "onnx");
    assert_eq!(body["data"]["tags"][0]["tag"], "cat");
    assert_eq!(dd_counts().await, before);
    assert!(matches!(run_hailort_task(|_| Ok(())).await, Err(HailoTaskError::Unavailable(_))));
    let mut ctx = DeviceCtx::default();
    assert!(matches!(ctx.run(|_| -> HailoRtResult<()> { panic!("queued work ran") }), Err(HailoTaskError::Unavailable(_))));
}

#[tokio::test]
async fn dd_session_lost_from_clip_create_degrades_health() {
    if isolated("dd_session_lost_from_clip_create_degrades_health") { return; }
    extern "C" { fn yu_hailort_stub_yolo_communication_closed(closed: bool); }
    let f = Fixture::new(vec![], 0, 0).await;
    run_hailort_task(|_| {
        // SAFETY: thread-local stub setter runs on its device thread.
        unsafe { yu_hailort_stub_yolo_communication_closed(true) };
        Ok(())
    }).await.unwrap();
    let clip = dd_route(f.state.clone(), "/v1/infer/clip-image", Some(json!({"image_base64": "AA=="}))).await;
    assert_eq!(clip.0, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(dd_route(f.state.clone(), "/healthz", None).await.1["hailo_session_lost"], true);
    let before = dd_counts().await;
    assert_eq!(dd_route(f.state.clone(), "/v1/infer/clip-image", Some(json!({"image_base64": "AA=="}))).await.0, StatusCode::SERVICE_UNAVAILABLE);
    error_eq(f.request().await, StatusCode::SERVICE_UNAVAILABLE, "device_error");
    assert_eq!(dd_counts().await, before);
}

#[tokio::test]
async fn dd_session_lost_exit_drains_response_once() {
    if isolated("dd_session_lost_exit_drains_response_once") { return; }
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let f = Fixture::new(vec![], 30, 0).await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let exits = Arc::new(AtomicUsize::new(0));
    let calls = exits.clone();
    let server = tokio::spawn(crate::hailort::session::serve(listener, build_router(f.state.clone()), move |code| {
        assert_eq!(code, 70);
        calls.fetch_add(1, Ordering::SeqCst);
    }));
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let body = f.body.to_string();
    let request = format!("POST /v1/infer/wd HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", f.state.auth_token, body.len(), body);
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut response = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut response)).await.unwrap().unwrap();
    let response = String::from_utf8(response).unwrap();
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");
    assert!(response.ends_with("{\"error\":\"device_error\"}"), "{response}");
    error_eq(f.request().await, StatusCode::SERVICE_UNAVAILABLE, "device_error");
    tokio::time::timeout(Duration::from_secs(5), server).await.unwrap().unwrap();
    assert_eq!(exits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn dd_hef_sha256_missing_or_malformed_is_invalid() {
    if isolated("dd_hef_sha256_missing_or_malformed_is_invalid") { return; }
    let mut f = Fixture::new(vec![], 0, 0).await;
    let before = dd_counts().await;
    f.body.as_object_mut().unwrap().remove("hef_sha256");
    error_eq(f.request().await, StatusCode::SERVICE_UNAVAILABLE, "hef_invalid");
    for hash in [Value::Null, json!("A".repeat(64)), json!("0".repeat(63)), json!("g".repeat(64))] {
        f.body["hef_sha256"] = hash;
        error_eq(f.request().await, StatusCode::SERVICE_UNAVAILABLE, "hef_invalid");
    }
    assert_eq!(dd_counts().await, before, "invalid hash reached HailoRT");
    assert!(!crate::hailort::session::is_lost());
}

#[tokio::test]
async fn dd_hef_sha256_wrong_never_opens() {
    if isolated("dd_hef_sha256_wrong_never_opens") { return; }
    let mut f = Fixture::new(vec![], 0, 0).await;
    let before = run_hailort_task(|_| Ok(crate::hailort::wd_hef::tests::stub::counts())).await.unwrap();
    f.body["hef_sha256"] = json!("0".repeat(64));
    error_eq(f.request().await, StatusCode::SERVICE_UNAVAILABLE, "hef_invalid");
    let after = run_hailort_task(|_| Ok(crate::hailort::wd_hef::tests::stub::counts())).await.unwrap();
    assert_eq!(&after[1..4], &before[1..4], "wrong digest reached open/metadata/configure");
    assert!(!crate::hailort::session::is_lost());
}

#[tokio::test]
async fn dd_hef_sha256_changed_request_does_not_trust_resident() {
    if isolated("dd_hef_sha256_changed_request_does_not_trust_resident") { return; }
    let mut f = Fixture::new(vec![], 0, 0).await;
    assert_eq!(f.request().await.0, StatusCode::OK);
    f.body["hef_sha256"] = json!("0".repeat(64));
    error_eq(f.request().await, StatusCode::SERVICE_UNAVAILABLE, "hef_invalid");
}

#[tokio::test]
async fn revfix_empty_hef_returns_hef_invalid() {
    if isolated("revfix_empty_hef_returns_hef_invalid") { return; }
    let f = Fixture::new(vec![], 0, 0).await;
    std::fs::write(f.body["hef_path"].as_str().unwrap(), []).unwrap();
    error_eq(f.request().await, StatusCode::SERVICE_UNAVAILABLE, "hef_invalid");
}

#[tokio::test]
async fn revfix_expired_device_closure_preserves_resident_without_open() {
    if isolated("revfix_expired_device_closure_preserves_resident_without_open") { return; }
    let mut f = Fixture::new(vec![], 0, 0).await;
    assert_eq!(f.request().await.0, StatusCode::OK);
    let before = run_hailort_task(|_| Ok(crate::hailort::wd_hef::tests::stub::counts())).await.unwrap();
    let resident = f.body["hef_path"].clone();
    let other = f.dir.0.join("other.hef");
    std::fs::write(&other, [4, 5]).unwrap();
    f.body["hef_path"] = json!(other);
    let result = crate::wd_hef_route::execute(f.state.clone(), serde_json::from_value(f.body.clone()).unwrap(), Instant::now() - Duration::from_secs(1)).await;
    assert_eq!(result.err(), Some("hef_timeout"));
    let after = run_hailort_task(|_| Ok(crate::hailort::wd_hef::tests::stub::counts())).await.unwrap();
    assert_eq!(after[1] - before[1], 0, "expired closure opened a HEF");
    assert_eq!(after[4] - before[4], 0, "expired closure evicted the resident");
    f.body["hef_path"] = resident;
    assert_eq!(f.request().await.0, StatusCode::OK);
    assert_eq!(f.configure_delta().await, 1);
}

#[tokio::test]
async fn t10_scan_root_refusal_matches_onnx_and_missing_image_stays_422() {
    if isolated("t10_scan_root_refusal_matches_onnx_and_missing_image_stays_422") { return; }
    let f = Fixture::new(vec![], 0, 0).await;
    f.state.scan_roots.write().unwrap().roots = vec![f.dir.0.join("model").canonicalize().unwrap()];
    let mut bodies = Vec::new();
    for backend in ["onnx", "hef"] {
        let mut body = f.body.clone();
        body["backend"] = json!(backend);
        let response = build_router(f.state.clone()).oneshot(
            Request::builder().method("POST").uri("/v1/infer/wd")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", f.state.auth_token))
                .body(Body::from(body.to_string())).unwrap()
        ).await.unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        bodies.push(axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap());
    }
    assert_eq!(bodies[0], bodies[1], "HEF and ONNX response bytes must match");
    assert_eq!(serde_json::from_slice::<Value>(&bodies[0]).unwrap(),
        json!({"error": "path outside allowed roots"}));

    std::fs::remove_file(f.dir.0.join("image.png")).unwrap();
    error_eq(f.request().await, StatusCode::UNPROCESSABLE_ENTITY, "image_invalid");
    assert_eq!(f.configure_delta().await, 0);
}

#[tokio::test]
async fn t9_metadata_table_rejects_before_configure() {
    if isolated("t9_metadata_table_rejects_before_configure") { return; }
    let violations = [
        (0, 0.0), (0, 2.0), (0, 41.0), (2, 447.0), (3, 447.0), (4, 4.0),
        (5, 2.0), (6, 0.0), (7, 602_111.0), (1, 0.0), (1, 2.0), (1, 41.0),
        (11, 2.0), (10, 4.0), (12, 4.0), (13, 0.0), (13, -0.1),
        (13, f64::NAN), (13, f64::INFINITY), (14, f64::NAN),
        (15, 0.0), (16, f64::INFINITY), (8, 0.0),
    ];
    let mut failures = Vec::new();
    for field in violations {
        let f = Fixture::new(vec![field], 0, 0).await;
        let actual = f.request().await;
        let configured = f.configure_delta().await;
        if actual != (StatusCode::SERVICE_UNAVAILABLE, json!({"error":"hef_invalid"})) || configured != 0 {
            failures.push(format!("{field:?}: {actual:?}, configure_delta={configured}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[tokio::test]
async fn t9_flattened_output_configures_once_and_dequantizes_nonzero_zp() {
    if isolated("t9_flattened_output_configures_once_and_dequantizes_nonzero_zp") { return; }
    for fields in [vec![], vec![(9, 3.0), (10, 1.0)], vec![(8, 3.0), (10, 1.0)]] {
        let f = Fixture::new(fields, 0, 0).await;
        for _ in 0..2 {
            let (status, response) = f.request().await;
            assert_eq!(status, StatusCode::OK, "{response}");
            assert_eq!(response["backend_applied"], "hef");
            assert_eq!(response["profile_applied"], false);
            assert_eq!(response["data"]["rating"], "general"); // (5 - 10) * 0.1 < 0
            assert_eq!(response["data"]["tags"], json!([
                {"tag":"cat", "category":"general", "confidence": f64::from(9.0_f32 * 0.1)},
            ])); // character score 0.8 is below 0.85
            assert_eq!(response["data"]["path"], f.body["path"]);
            assert_eq!(response["data"]["model_id"], "model");
        }
        assert_eq!(f.configure_delta().await, 1);
        assert!(f.state.wd_infer.read().unwrap().is_empty());
    }
}

#[tokio::test]
async fn t9_cache_hit_revalidates_vocabulary() {
    if isolated("t9_cache_hit_revalidates_vocabulary") { return; }
    let f = Fixture::new(vec![], 0, 0).await;
    assert_eq!(f.request().await.0, StatusCode::OK);
    std::fs::write(f.dir.0.join("model/selected_tags.csv"), "name,category\ncat,0\n").unwrap();
    error_eq(f.request().await, StatusCode::SERVICE_UNAVAILABLE, "hef_invalid");
    assert_eq!(f.configure_delta().await, 1);
}

#[tokio::test]
async fn t9_profile_table_and_acknowledgement() {
    if isolated("t9_profile_table_and_acknowledgement") { return; }
    let f = Fixture::new(vec![], 0, 0).await;
    let profile = serde_json::to_value(infer_core::builtin_wd_profile("model.onnx", 0.4, 0.7)).unwrap();
    let mut body = f.body.clone();
    for (key, value) in [
        ("input_size", json!(224)), ("pad_color", json!([0,0,0])),
        ("channel_order", json!("RGB")), ("layout", json!("NCHW")),
        ("scale", json!(0.5)), ("mean", json!([0,0,0])), ("std", json!([1,1,1])),
        ("resize_strategy", json!("stretch")),
    ] {
        body["profile"] = profile.clone();
        body["profile"]["preprocess_spec"][key] = value;
        error_eq(wd_wire_request(f.state.clone(), body.clone()).await,
            StatusCode::SERVICE_UNAVAILABLE, "hef_invalid");
    }
    assert_eq!(f.configure_delta().await, 0);
    body["profile"] = profile;
    let (status, response) = wd_wire_request(f.state.clone(), body).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["backend_applied"], "hef");
    assert_eq!(response["profile_applied"], true);
    assert_eq!(response["data"]["tags"].as_array().unwrap().len(), 2);
}

#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<u8>>>);

impl Write for Logs {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

#[tokio::test]
async fn t9_http_error_table_and_path_free_logs() {
    if isolated("t9_http_error_table_and_path_free_logs") { return; }
    let logs = Logs::default();
    let writer = logs.clone();
    let subscriber = tracing_subscriber::fmt().without_time().with_ansi(false)
        .with_writer(move || writer.clone()).finish();
    let _guard = tracing::subscriber::set_default(subscriber);
    for (scenario, status, code) in [
        (0, StatusCode::SERVICE_UNAVAILABLE, "hef_not_found"),
        (1, StatusCode::SERVICE_UNAVAILABLE, "hef_invalid"),
        (2, StatusCode::SERVICE_UNAVAILABLE, "device_error"),
        (3, StatusCode::UNPROCESSABLE_ENTITY, "image_invalid"),
        (4, StatusCode::SERVICE_UNAVAILABLE, "model_not_downloaded"),
        (5, StatusCode::SERVICE_UNAVAILABLE, "model_not_downloaded"),
        (6, StatusCode::SERVICE_UNAVAILABLE, "hef_invalid"),
    ] {
        let f = Fixture::new(vec![], if scenario == 2 { 6 } else if scenario == 6 { 3 } else { 0 }, 0).await;
        match scenario {
            0 => std::fs::remove_file(f.dir.0.join("private-model.hef")).unwrap(),
            1 => { std::fs::remove_file(f.dir.0.join("private-model.hef")).unwrap();
                std::fs::create_dir(f.dir.0.join("private-model.hef")).unwrap(); }
            3 => std::fs::write(f.dir.0.join("image.png"), b"invalid image").unwrap(),
            4 => std::fs::remove_file(f.dir.0.join("model/model.onnx")).unwrap(),
            5 => std::fs::remove_file(f.dir.0.join("model/selected_tags.csv")).unwrap(),
            _ => {}
        }
        error_eq(f.request().await, status, code);
        let text = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
        assert!(!text.contains(f.dir.0.to_str().unwrap()));
        assert!(!text.contains("private-model.hef"));
        assert!(!text.contains("image.png"));
        if let Ok(home) = std::env::var("HOME") { assert!(!text.contains(&home)); }
    }
    assert!(!logs.0.lock().unwrap().is_empty(), "capture must observe HEF error codes");
}

#[tokio::test]
async fn t9_preprocessing_busy_is_separate_from_queue_full() {
    if isolated("t9_preprocessing_busy_is_separate_from_queue_full") { return; }
    let f = Fixture::new(vec![], 0, 0).await;
    let permit = media_preprocessing_semaphore().try_acquire_many_owned(
        media_preprocessing_permits(MAX_MEDIA_PREPROCESSING_BYTES)).unwrap();
    let actual = f.request().await;
    drop(permit);
    error_eq(actual, StatusCode::SERVICE_UNAVAILABLE, "device_busy");
    assert_eq!(f.configure_delta().await, 0);
}

#[tokio::test]
async fn t9_queue_full_is_device_busy() {
    if isolated("t9_queue_full_is_device_busy") { return; }
    let f = Fixture::new(vec![], 0, 0).await;
    let (started, release) = enqueue_device_blocker();
    started.await.unwrap();
    fill_device_queue();
    let actual = f.request().await;
    let clip = dd_route(f.state.clone(), "/v1/infer/clip-image", Some(json!({"image_base64": "AA=="}))).await;
    release.send(()).unwrap();
    error_eq(actual, StatusCode::SERVICE_UNAVAILABLE, "device_busy");
    assert_eq!(clip, (StatusCode::SERVICE_UNAVAILABLE, json!({
        "ok": false,
        "error": "hailort_clip_image_metadata_failed",
        "message": "HailoRT device queue is full",
    })));
}

fn enqueue_device_blocker() -> (tokio::sync::oneshot::Receiver<()>, mpsc::Sender<()>) {
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = mpsc::channel();
    assert!(hailort_device_sender().try_send(Box::new(move |_| {
        started_tx.send(()).unwrap();
        let _ = release_rx.recv_timeout(Duration::from_secs(10));
    })).is_ok());
    (started_rx, release_tx)
}

fn fill_device_queue() {
    for _ in 0..MAX_PENDING_DEVICE_TASKS {
        assert!(hailort_device_sender().try_send(Box::new(|_| {})).is_ok());
    }
}

#[tokio::test]
async fn t27a_clip_image_run_queue_full_keeps_error_shape() {
    use base64::Engine;

    if isolated("t27a_clip_image_run_queue_full_keeps_error_shape") { return; }
    let f = Fixture::new(vec![], 0, 0).await;
    let hef_path = f.body["hef_path"].as_str().unwrap().to_owned();
    run_hailort_task({
        let hef_path = hef_path.clone();
        move |ctx| {
            ctx.infer_models.insert(hef_path, ShimYolo::stub_with_metadata());
            Ok(())
        }
    }).await.unwrap();
    let image_base64 = base64::engine::general_purpose::STANDARD
        .encode(std::fs::read(f.dir.0.join("image.png")).unwrap());

    let (started, release) = enqueue_device_blocker();
    started.await.unwrap();
    let clip = dd_route(f.state.clone(), "/v1/infer/clip-image",
        Some(json!({"hef_path": hef_path, "image_base64": image_base64})));
    tokio::pin!(clip);
    assert!(futures_util::poll!(&mut clip).is_pending());
    // This blocker runs after metadata. The next poll reaches the run with a full queue.
    let (run_started, run_release) = enqueue_device_blocker();
    release.send(()).unwrap();
    run_started.await.unwrap();
    fill_device_queue();
    let actual = clip.await;
    run_release.send(()).unwrap();
    assert_eq!(actual, (StatusCode::SERVICE_UNAVAILABLE, json!({
        "ok": false,
        "error": "hailort_clip_image_failed",
        "message": "HailoRT device queue is full",
    })));
}

#[test]
fn t9_preprocessing_deadline_skips_run() {
    if isolated("t9_preprocessing_deadline_skips_run") { return; }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build().unwrap();
    let mut f = runtime.block_on(Fixture::new(vec![], 0, 0));
    f.body["timeout_ms"] = json!(150);
    let permits = media_preprocessing_semaphore().available_permits();
    let (actual, elapsed, pending_permits, runs, snapshot) = runtime.block_on(async {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = mpsc::channel();
        // Keep preprocessing queued until its deadline, independent of CPU/image speed.
        let blocker = tokio::task::spawn_blocking(move || {
            started_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        });
        started_rx.await.unwrap();
        let started = Instant::now();
        let actual = f.request().await;
        let elapsed = started.elapsed();
        let pending_permits = media_preprocessing_semaphore().available_permits();
        let (runs, snapshot) = run_hailort_task(|_| {
            Ok((wd_stub::run_count(), wd_stub::last_run()))
        }).await.unwrap();
        release_tx.send(()).unwrap();
        blocker.await.unwrap();
        (actual, elapsed, pending_permits, runs, snapshot)
    });
    // Drain the canceled request's blocking preprocessing while its fixture still exists.
    drop(runtime);
    error_eq(actual, StatusCode::SERVICE_UNAVAILABLE, "hef_timeout");
    assert!(elapsed < Duration::from_millis(450), "preprocessing deadline: {elapsed:?}");
    assert_eq!(permits - pending_permits,
        media_preprocessing_permits(MAX_IMAGE_PREPROCESSING_RESERVATION_BYTES) as usize);
    assert_eq!(runs - f.runs, 0);
    assert_eq!(snapshot, (0, vec![]));
}

#[tokio::test]
async fn t9_run_deadline_bounds_worker_reply() {
    if isolated("t9_run_deadline_bounds_worker_reply") { return; }
    let mut f = Fixture::new(vec![], 0, 5000).await;
    image::RgbImage::from_pixel(1, 1, image::Rgb([10, 20, 30]))
        .save(f.dir.0.join("image.png")).unwrap();
    f.body["timeout_ms"] = json!(2000);
    let started = Instant::now();
    let actual = f.request().await;
    let elapsed = started.elapsed();
    let (timeout, _) = run_hailort_task(|_| Ok(wd_stub::last_run())).await.unwrap();
    let runs = run_hailort_task(|_| Ok(wd_stub::run_count())).await.unwrap() - f.runs;
    error_eq(actual, StatusCode::SERVICE_UNAVAILABLE, "hef_timeout");
    assert!(elapsed < Duration::from_millis(2300), "sidecar waited for the slow stub: {elapsed:?}");
    assert_eq!(runs, 1);
    assert!((1..=2000).contains(&timeout), "run timeout={timeout}");
}

#[tokio::test]
async fn t9_uint8_bgr_letterbox_and_run_timeout() {
    if isolated("t9_uint8_bgr_letterbox_and_run_timeout") { return; }
    let mut f = Fixture::new(vec![], 0, 0).await;
    f.body["timeout_ms"] = json!(120_000);
    let (status, response) = f.request().await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let (timeout, pixels) = run_hailort_task(|_| Ok(wd_stub::last_run())).await.unwrap();
    assert!((1..=120_000).contains(&timeout));
    assert_eq!(pixels.len(), 602_112);
    assert_eq!(&pixels[..3], &[255, 255, 255]);
    let centre = (224 * 448 + 224) * 3;
    assert_eq!(&pixels[centre..centre + 3], &[30, 20, 10]);
}

#[tokio::test]
async fn t9_degraded_worker_maps_device_error() {
    if isolated("t9_degraded_worker_maps_device_error") { return; }
    let f = Fixture::new(vec![], 0, 0).await;
    run_hailort_task(|ctx| { ctx.degraded = true; Ok(()) }).await.unwrap();
    let actual = f.request().await;
    let (tx, rx) = tokio::sync::oneshot::channel();
    assert!(hailort_device_sender().try_send(Box::new(|ctx| {
        ctx.degraded = false;
        tx.send(()).unwrap();
    })).is_ok());
    rx.await.unwrap();
    error_eq(actual, StatusCode::SERVICE_UNAVAILABLE, "device_error");
}

#[tokio::test]
async fn t9_native_timeout_is_sidecar_hef_timeout() {
    if isolated("t9_native_timeout_is_sidecar_hef_timeout") { return; }
    let f = Fixture::new(vec![], 7, 0).await;
    error_eq(f.request().await, StatusCode::SERVICE_UNAVAILABLE, "hef_timeout");
    assert_eq!(f.configure_delta().await, 1);
}
