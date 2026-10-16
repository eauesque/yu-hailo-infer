use std::{
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::Response,
    Extension, Json,
};
use infer_core::embeddinggemma2::{self as eg2, EmbeddingGemma2Cache, TextTask};
use serde::Deserialize;
use serde_json::json;

use crate::router::{api_error, api_ok, auth_error, AppState};

#[derive(Deserialize)]
pub(crate) struct TextEmbedRequest {
    text: String,
    title: Option<String>,
    #[serde(default)]
    task: TextTask,
    #[serde(default = "default_dim")]
    dim: usize,
    model_dir: Option<PathBuf>,
}

fn default_dim() -> usize {
    eg2::DIM
}

fn model_dir(requested: Option<PathBuf>) -> Result<PathBuf, &'static str> {
    if let Some(dir) = requested {
        if !dir.is_absolute() {
            return Err("model_dir must be absolute");
        }
        return Ok(dir);
    }
    let home = std::env::var_os("HOME").ok_or("HOME is unavailable; supply model_dir")?;
    Ok(PathBuf::from(home)
        .join(".cache/yu_ai_manager/embeddinggemma2")
        .join(eg2::REVISION))
}

const DEADLINE: std::time::Duration = std::time::Duration::from_secs(170);

fn execution_slots() -> &'static Arc<tokio::sync::Semaphore> {
    static SLOTS: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
    SLOTS.get_or_init(|| Arc::new(tokio::sync::Semaphore::new(1)))
}

fn admission_slots() -> &'static Arc<tokio::sync::Semaphore> {
    static SLOTS: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
    SLOTS.get_or_init(|| Arc::new(tokio::sync::Semaphore::new(5)))
}

pub(crate) async fn text_embed(
    State(state): State<AppState>,
    Extension(cache): Extension<Arc<Mutex<EmbeddingGemma2Cache>>>,
    headers: HeaderMap,
    Json(body): Json<TextEmbedRequest>,
) -> Response {
    if let Some(response) = auth_error(&state, &headers) {
        return response;
    }
    if body.dim != eg2::DIM {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_text_embed_dimension",
            "only dim=256 is supported".into(),
        );
    }
    if body
        .text
        .len()
        .saturating_add(body.title.as_ref().map_or(0, |v| v.len()))
        > eg2::MAX_TEXT_BYTES
    {
        return api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "text_embed_too_large",
            "text and title exceed 16384 bytes".into(),
        );
    }
    if let Err(message) = eg2::prepare_text(&body.text, body.title.as_deref(), body.task) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_text_embed_input",
            message.into(),
        );
    }
    let dir = match model_dir(body.model_dir) {
        Ok(dir) => dir,
        Err(message) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_text_embed_model_dir",
                message.into(),
            )
        }
    };
    // Admit at most one running request and four FIFO waiters.
    let admission = match admission_slots().clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "text_embed_busy",
                "text embedding capacity is exhausted".into(),
            )
        }
    };
    let deadline = tokio::time::Instant::now() + DEADLINE;
    let execution =
        match tokio::time::timeout_at(deadline, execution_slots().clone().acquire_owned()).await {
            Ok(permit) => permit.expect("semaphore never closed"),
            Err(_) => {
                return api_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "text_embed_busy",
                    "text embedding capacity is exhausted".into(),
                )
            }
        };
    let task = tokio::task::spawn_blocking(move || {
        // Response deadline, not cancellation: even a wedged ORT run retains both permits and the slot.
        let (_admission, _execution) = (admission, execution);
        let mut cache = cache.lock().unwrap_or_else(|poisoned| {
            let mut guard = poisoned.into_inner();
            *guard = EmbeddingGemma2Cache::default();
            cache.clear_poison();
            guard
        });
        cache.encode(&dir, &body.text, body.title.as_deref(), body.task)
    });
    let result = match tokio::time::timeout_at(deadline, task).await {
        Ok(result) => result,
        Err(_) => {
            return api_error(
                StatusCode::GATEWAY_TIMEOUT,
                "text_embed_timeout",
                "text embedding timed out".into(),
            )
        }
    };
    match result {
        Ok(Ok(vector)) => api_ok(
            json!({"vector": vector, "dim": eg2::DIM, "model": eg2::MODEL, "revision": eg2::REVISION, "precision": "fp32", "preprocessing": eg2::PREPROCESSING}),
        ),
        Ok(Err(infer_core::InferError::ModelNotDownloaded(_))) => api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "text_embed_model_not_downloaded",
            "EmbeddingGemma 2 FP32 model is not downloaded".into(),
        ),
        Ok(Err(error)) => {
            tracing::warn!(%error, "text embedding failed");
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "text_embed_inference_failed",
                "text embedding failed".into(),
            )
        }
        Err(error) => {
            tracing::warn!(%error, "text embedding task failed");
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "text_embed_task_failed",
                "text embedding task failed".into(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::{build_router, ScanRoots};
    use axum::{body::Body, http::Request};
    use std::{collections::HashMap, sync::RwLock};
    use tower::ServiceExt;

    static TEST_SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    fn test_state() -> AppState {
        AppState {
            started_at: std::time::Instant::now(),
            instance_id: "test".into(),
            scan_roots: Arc::new(RwLock::new(ScanRoots {
                roots: vec![],
                generation: 0,
            })),
            auth_token: "test-token".into(),
            wd_cache_dir: PathBuf::new(),
            clip_model_dir: PathBuf::new(),
            wd_infer: Arc::new(RwLock::new(HashMap::new())),
            clip_text: Arc::new(RwLock::new(HashMap::new())),
            clip_image: Arc::new(RwLock::new(HashMap::new())),
            yolo_onnx: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn request(value: serde_json::Value, token: Option<&str>) -> Response {
        let state = test_state();
        let mut builder = Request::builder()
            .method("POST")
            .uri("/v1/infer/text-embed")
            .header("content-type", "application/json");
        if let Some(token) = token {
            builder = builder.header("authorization", token);
        }
        build_router(state)
            .oneshot(builder.body(Body::from(value.to_string())).unwrap())
            .await
            .unwrap()
    }

    fn direct_request(
        cache: Arc<Mutex<EmbeddingGemma2Cache>>,
    ) -> impl std::future::Future<Output = Response> {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer test-token".parse().unwrap());
        text_embed(
            State(test_state()),
            Extension(cache),
            headers,
            Json(
                serde_json::from_value(json!({"text":"cat", "model_dir":"/nonexistent-eg2-test"}))
                    .unwrap(),
            ),
        )
    }

    async fn response_body(response: Response) -> serde_json::Value {
        serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap(),
        )
        .unwrap()
    }

    // A real clock guard keeps timeout sentinels finite even with Tokio time paused.
    async fn real_guard<T>(future: impl std::future::Future<Output = T>) -> T {
        let (tx, rx) = tokio::sync::oneshot::channel();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let _ = tx.send(());
        });
        tokio::select! {
            value = future => value,
            _ = rx => panic!("real-time guard expired"),
        }
    }

    #[tokio::test]
    async fn poisoned_cache_recovers_and_clears_poison() {
        let _serial = TEST_SERIAL.lock().await;
        let cache = Arc::new(Mutex::new(EmbeddingGemma2Cache::default()));
        let other = cache.clone();
        assert!(std::thread::spawn(move || {
            let _guard = other.lock().unwrap();
            panic!("poison fixture");
        })
        .join()
        .is_err());
        let response = direct_request(cache.clone()).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response_body(response).await["error"],
            "text_embed_model_not_downloaded"
        );
        assert!(!cache.is_poisoned());
    }

    #[tokio::test]
    async fn queue_deadline_releases_admission() {
        let _serial = TEST_SERIAL.lock().await;
        let execution = execution_slots().clone().acquire_owned().await.unwrap();
        tokio::time::pause();
        let task = tokio::spawn(direct_request(Arc::new(Mutex::new(
            EmbeddingGemma2Cache::default(),
        ))));
        while admission_slots().available_permits() != 4 {
            tokio::task::yield_now().await;
        }
        tokio::time::advance(DEADLINE + std::time::Duration::from_secs(1)).await;
        let response = real_guard(task).await.unwrap();
        drop(execution);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response_body(response).await["error"], "text_embed_busy");
        assert_eq!(admission_slots().available_permits(), 5);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // Deliberately block cache.lock() to test the response deadline.
    async fn run_deadline_retains_permits_until_blocking_task_finishes() {
        let _serial = TEST_SERIAL.lock().await;
        let cache = Arc::new(Mutex::new(EmbeddingGemma2Cache::default()));
        let guard = cache.lock().unwrap();
        tokio::time::pause();
        let task = tokio::spawn(direct_request(cache.clone()));
        while execution_slots().available_permits() != 0 {
            tokio::task::yield_now().await;
        }
        tokio::time::advance(DEADLINE + std::time::Duration::from_secs(1)).await;
        // Release the std lock even when the sentinel guard expires.
        let outcome = std::panic::AssertUnwindSafe(real_guard(task));
        use futures_util::FutureExt;
        let outcome = outcome.catch_unwind().await;
        let held = (
            admission_slots().available_permits(),
            execution_slots().available_permits(),
        );
        drop(guard);
        real_guard(async {
            while admission_slots().available_permits() != 5
                || execution_slots().available_permits() != 1
            {
                tokio::task::yield_now().await;
            }
        })
        .await;
        let response = outcome.unwrap().unwrap();
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(response_body(response).await["error"], "text_embed_timeout");
        assert_eq!(held, (4, 0));
    }

    #[tokio::test]
    async fn inference_error_body_hides_artifact_and_path() {
        let _serial = TEST_SERIAL.lock().await;
        let dir = std::env::temp_dir().join(format!("eg2-error-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("onnx")).unwrap();
        std::fs::write(dir.join("onnx/model.onnx"), b"bad").unwrap();
        let response = request(
            json!({"text":"cat", "model_dir":dir}),
            Some("Bearer test-token"),
        )
        .await;
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = response_body(response).await;
        assert_eq!(body["message"], "text embedding failed");
        assert!(!body.to_string().contains("model.onnx"));
        assert!(!body.to_string().contains(dir.to_str().unwrap()));
    }

    #[tokio::test]
    async fn rejects_missing_or_wrong_auth_before_model_access() {
        let _serial = TEST_SERIAL.lock().await;
        for token in [None, Some("Bearer wrong")] {
            assert_eq!(
                request(json!({"text":"猫", "model_dir":"/missing"}), token)
                    .await
                    .status(),
                StatusCode::UNAUTHORIZED
            );
        }
    }

    #[tokio::test]
    async fn validates_dimensions_tasks_titles_and_lengths() {
        let _serial = TEST_SERIAL.lock().await;
        for value in [
            json!({"text":"cat", "dim":768}),
            json!({"text":""}),
            json!({"text":"", "task":"document"}),
            json!({"text":"cat", "model_dir":"relative"}),
        ] {
            assert_eq!(
                request(value, Some("Bearer test-token")).await.status(),
                StatusCode::BAD_REQUEST
            );
        }
        for value in [
            json!({"text":"cat", "task":"image"}),
            json!({"text":"cat", "title":42}),
        ] {
            assert_eq!(
                request(value, Some("Bearer test-token")).await.status(),
                StatusCode::UNPROCESSABLE_ENTITY
            );
        }
        assert_eq!(
            request(
                json!({"text":"cat", "title":"a".repeat(eg2::MAX_TEXT_BYTES)}),
                Some("Bearer test-token")
            )
            .await
            .status(),
            StatusCode::PAYLOAD_TOO_LARGE
        );
    }

    #[tokio::test]
    async fn defaults_and_title_only_documents_reach_model_loader() {
        let _serial = TEST_SERIAL.lock().await;
        for value in [
            json!({"text":"cat", "model_dir":"/nonexistent-eg2-test"}),
            json!({"text":"", "title":"猫.jpg", "task":"document", "model_dir":"/nonexistent-eg2-test"}),
        ] {
            assert_eq!(
                request(value, Some("Bearer test-token")).await.status(),
                StatusCode::SERVICE_UNAVAILABLE
            );
        }
    }

    #[tokio::test]
    async fn full_queue_rejects_overflow_and_cancelled_waiter_releases_capacity() {
        let _serial = TEST_SERIAL.lock().await;
        let execution = execution_slots().clone().acquire_owned().await.unwrap();
        let body = json!({"text":"cat", "model_dir":"/nonexistent-eg2-test"});
        let mut waiters = Vec::new();
        for _ in 0..5 {
            waiters.push(tokio::spawn(request(
                body.clone(),
                Some("Bearer test-token"),
            )));
        }
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while admission_slots().available_permits() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let overflow = request(body, Some("Bearer test-token")).await;
        assert_eq!(overflow.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = axum::body::to_bytes(overflow.into_body(), 4096)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()["error"],
            "text_embed_busy"
        );
        let cancelled = waiters.pop().unwrap();
        cancelled.abort();
        assert!(cancelled.await.unwrap_err().is_cancelled());
        assert_eq!(admission_slots().available_permits(), 1);
        drop(execution);
        for waiter in waiters {
            assert_eq!(
                waiter.await.unwrap().status(),
                StatusCode::SERVICE_UNAVAILABLE
            );
        }
        assert_eq!(admission_slots().available_permits(), 5);
        assert_eq!(execution_slots().available_permits(), 1);
    }
}
