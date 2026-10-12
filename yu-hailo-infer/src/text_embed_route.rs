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
    let execution = execution_slots()
        .clone()
        .acquire_owned()
        .await
        .expect("semaphore never closed");
    let result = tokio::task::spawn_blocking(move || {
        // Permits remain held even if the HTTP request is cancelled.
        let (_admission, _execution) = (admission, execution);
        let mut cache = cache.lock().map_err(|_| {
            infer_core::InferError::InvalidModelOutput("text embedding cache lock poisoned".into())
        })?;
        cache.encode(&dir, &body.text, body.title.as_deref(), body.task)
    })
    .await;
    match result {
        Ok(Ok(vector)) => api_ok(
            json!({"vector": vector, "dim": eg2::DIM, "model": eg2::MODEL, "revision": eg2::REVISION, "precision": "fp32", "preprocessing": eg2::PREPROCESSING}),
        ),
        Ok(Err(infer_core::InferError::ModelNotDownloaded(_))) => api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "text_embed_model_not_downloaded",
            "EmbeddingGemma 2 FP32 model is not downloaded".into(),
        ),
        Ok(Err(error)) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "text_embed_inference_failed",
            error.to_string(),
        ),
        Err(error) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "text_embed_task_failed",
            error.to_string(),
        ),
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

    async fn request(value: serde_json::Value, token: Option<&str>) -> Response {
        let state = AppState {
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
        };
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
