use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, OnceLock, RwLock};

use axum::{
    body::Body,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::hailort::{
    clip_image_metadata, run_clip_image_once, run_yolo_once,
    yolo_metadata as resident_yolo_metadata, HailoRtError, HailoRtResult, Llm, LlmChatMessage,
    LlmGenerationParams, LlmStream, ShimYolo, Speech2Text, Vlm, VlmGenerationParams, VlmStream,
    YoloModelMetadata,
};
use crate::speech2text_route::speech2text_transcribe_route;

pub use crate::speech2text_route::TranscribeRequest;

const MAX_TEXT_BYTES: usize = 16 * 1024;
const MAX_PROMPT_BYTES: usize = 16 * 1024;
/// Upper bound accepted for a caller-supplied inference timeout. Prevents a
/// client from holding the HailoRT device thread indefinitely.
pub(crate) const MAX_TIMEOUT_MS: u32 = 120_000;
const MAX_FRAMES: usize = 8;
const MAX_FRAME_BASE64_BYTES: usize = 8 * 1024 * 1024;
/// axum's DefaultBodyLimit defaults to 2 MiB, which is below
/// MAX_FRAME_BASE64_BYTES on its own; a request with MAX_FRAMES frames each
/// near the per-frame cap would be rejected by axum before reaching this
/// file's own (more specific, better-error-messaged) size checks. Routes
/// that accept up to MAX_FRAMES base64 frames raise their body limit to
/// this value (worst case: MAX_FRAMES frames + prompt + JSON overhead).
const MAX_VLM_BODY_BYTES: usize = MAX_FRAMES * MAX_FRAME_BASE64_BYTES + MAX_PROMPT_BYTES + 4096;
/// Upper bound accepted for a caller-supplied `max_generated_tokens`
/// override. Prevents a client from requesting an unbounded-length
/// generation that would hold the HailoRT device thread indefinitely.
const MAX_VLM_GENERATED_TOKENS: u32 = 4096;
/// Independent, defense-in-depth cap on the number of stream reads per
/// request, applied regardless of `max_generated_tokens` — guards against a
/// model/SDK bug that never reaches a terminal `generation_status()`.
const MAX_VLM_STREAM_READS: usize = 8192;
/// Same caps as MAX_VLM_GENERATED_TOKENS/MAX_VLM_STREAM_READS, for the LLM
/// streaming endpoint.
const MAX_LLM_GENERATED_TOKENS: u32 = 4096;
const MAX_LLM_STREAM_READS: usize = 8192;
/// Upper bound on the number of chat turns accepted per request — a
/// generously large multi-turn history, not a realistic conversation length.
const MAX_LLM_MESSAGES: usize = 256;
/// Upper bound on the number of tool definitions accepted per request — a
/// generously large tool set, not a realistic agent tool count.
const MAX_LLM_TOOLS: usize = 64;
const MAX_PANIC_HANDLE_DISCARDS: usize = 3;
const HAILORT_DEGRADED_MESSAGE: &str =
    "HailoRT handle discard limit reached; reboot required before further inference";

/// Media parsing runs outside Tokio's cooperative workers. Reservations are
/// counted in MiB so image and audio routes share one process-wide budget.
const MEDIA_PREPROCESSING_PERMIT_BYTES: u64 = 1024 * 1024;
const MAX_MEDIA_PREPROCESSING_BYTES: u64 = 256 * 1024 * 1024;
/// Includes decoded pixels, encoded input, and transient resize allocations.
const MAX_IMAGE_PREPROCESSING_RESERVATION_BYTES: u64 = 96 * 1024 * 1024;

#[derive(Debug)]
pub(crate) enum MediaPreprocessError<E> {
    Busy,
    Task(E),
    Join(String),
}

fn media_preprocessing_semaphore() -> Arc<tokio::sync::Semaphore> {
    static SEMAPHORE: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
    SEMAPHORE
        .get_or_init(|| {
            Arc::new(tokio::sync::Semaphore::new(
                (MAX_MEDIA_PREPROCESSING_BYTES / MEDIA_PREPROCESSING_PERMIT_BYTES) as usize,
            ))
        })
        .clone()
}

fn media_preprocessing_permits(reservation_bytes: u64) -> u32 {
    u32::try_from(reservation_bytes.div_ceil(MEDIA_PREPROCESSING_PERMIT_BYTES))
        .expect("media preprocessing reservation exceeds semaphore capacity type")
}

#[derive(Clone)]
pub struct AppState {
    pub started_at: std::time::Instant,
    pub instance_id: String,
    pub scan_roots: Arc<RwLock<Vec<PathBuf>>>,
    pub auth_token: String,
    pub wd_cache_dir: PathBuf,
    pub wd_infer: Arc<RwLock<HashMap<String, Arc<infer_core::WdInferEngine>>>>,
    pub clip_text: Arc<RwLock<HashMap<String, Arc<infer_core::ClipTextEncoder>>>>,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/_internal/scan-roots-changed", post(scan_roots_changed))
        .route("/v1/infer/wd", post(infer_wd))
        .route("/v1/infer/yolo/metadata", get(yolo_metadata))
        .route("/v1/infer/yolo/smoke-zero", get(yolo_smoke_zero))
        .route(
            "/v1/infer/yolo/detect",
            post(yolo_detect).layer(axum::extract::DefaultBodyLimit::max(
                MAX_FRAME_BASE64_BYTES + 4096,
            )),
        )
        .route(
            "/v1/infer/clip-image",
            post(clip_image).layer(axum::extract::DefaultBodyLimit::max(
                MAX_FRAME_BASE64_BYTES + 4096,
            )),
        )
        .route("/v1/infer/clip-text", post(clip_text))
        .route("/v1/infer/speech2text/tokenize", post(speech2text_tokenize))
        .route(
            "/v1/infer/speech2text/transcribe",
            speech2text_transcribe_route(),
        )
        .route("/v1/infer/llm/tokenize", post(llm_tokenize))
        .route("/v1/infer/llm/generate", post(llm_generate))
        .route("/v1/infer/llm/generate/stream", post(llm_generate_stream))
        .route(
            "/v1/infer/vlm/generate",
            post(vlm_generate).layer(axum::extract::DefaultBodyLimit::max(MAX_VLM_BODY_BYTES)),
        )
        .route(
            "/v1/infer/vlm/generate/stream",
            post(vlm_generate_stream)
                .layer(axum::extract::DefaultBodyLimit::max(MAX_VLM_BODY_BYTES)),
        )
        .with_state(state)
}

async fn healthz(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "ok": true,
        "uptime_secs": state.started_at.elapsed().as_secs(),
        "instance_id": state.instance_id,
    }))
}

#[derive(Deserialize)]
pub struct ScanRootsChangedRequest {
    pub scan_roots: Vec<String>,
}

async fn scan_roots_changed(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<ScanRootsChangedRequest>,
) -> impl IntoResponse {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let roots = body
        .scan_roots
        .into_iter()
        .filter_map(|path| std::fs::canonicalize(path).ok())
        .collect();
    *state.scan_roots.write().unwrap() = roots;
    Json(json!({"ok": true})).into_response()
}

#[derive(Deserialize)]
pub struct InferWdRequest {
    pub path: String,
    pub model_id: String,
    #[serde(default = "default_general_thr")]
    pub general_thr: f32,
    #[serde(default = "default_character_thr")]
    pub character_thr: f32,
}

#[derive(Debug, Deserialize)]
pub struct YoloDetectRequest {
    hef_path: Option<String>,
    input_base64: String,
    conf_threshold: Option<f64>,
    iou_threshold: Option<f64>,
    num_classes: Option<usize>,
    input_size: Option<u32>,
    orig_w: Option<u32>,
    orig_h: Option<u32>,
    scale: Option<f64>,
    pad_x: Option<f64>,
    pad_y: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct ClipImageRequest {
    hef_path: Option<String>,
    image_base64: String,
}

#[derive(Debug, Deserialize)]
pub struct ClipTextRequest {
    model_dir: Option<String>,
    text: String,
}

fn default_general_thr() -> f32 {
    0.35
}

fn default_character_thr() -> f32 {
    0.85
}

fn check_auth(state: &AppState, headers: &HeaderMap) -> bool {
    let Some(header_value) = headers.get(axum::http::header::AUTHORIZATION) else {
        return false;
    };
    let Ok(header_str) = header_value.to_str() else {
        return false;
    };
    let Some(token) = header_str.strip_prefix("Bearer ") else {
        return false;
    };
    auth_core::verify_token(token, &state.auth_token)
}

fn check_scan_roots(state: &AppState, real_path: &std::path::Path) -> bool {
    let scan_roots = state.scan_roots.read().unwrap();
    !scan_roots.is_empty() && scan_roots.iter().any(|root| real_path.starts_with(root))
}

fn check_model_id(model_id: &str) -> bool {
    !(model_id.contains('/') || model_id.contains('\\') || model_id.contains(".."))
}

#[derive(Debug, Deserialize)]
pub struct HefQuery {
    hef_path: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TokenizeRequest {
    hef_path: Option<String>,
    text: String,
}

#[derive(Debug, Deserialize)]
pub struct LlmGenerateRequest {
    hef_path: Option<String>,
    prompt: String,
    timeout_ms: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct LlmChatMessageRequest {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
pub struct LlmGenerateStreamRequest {
    hef_path: Option<String>,
    /// Ordered chat turns (e.g. `[{"role":"system",...}, {"role":"user",...},
    /// {"role":"assistant",...}, {"role":"user",...}]`) — the HailoRT chat
    /// template renders the whole exchange, not just the latest turn.
    messages: Vec<LlmChatMessageRequest>,
    /// Tool definitions (OpenAI-function-style objects, e.g.
    /// `{"name":...,"description":...,"parameters":...}`), forwarded as-is to
    /// the HailoRT SDK's native `write(messages, tools)` so the model's own
    /// chat template renders them (rather than the caller embedding a tool
    /// listing into a prose system prompt). Empty/absent means no tools.
    #[serde(default)]
    tools: Vec<serde_json::Value>,
    /// Per-token read timeout, not an overall generation deadline.
    timeout_ms: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    frequency_penalty: Option<f32>,
    max_generated_tokens: Option<u32>,
    do_sample: Option<bool>,
    seed: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct VlmGenerateRequest {
    hef_path: Option<String>,
    prompt: String,
    /// Base64-encoded image frames (JPEG/PNG/WebP). The Hailo-10H VLM model
    /// requires at least one frame — image-free generation is not supported.
    #[serde(default)]
    frames: Vec<String>,
    timeout_ms: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct VlmGenerateStreamRequest {
    hef_path: Option<String>,
    prompt: String,
    system_prompt: Option<String>,
    #[serde(default)]
    frames: Vec<String>,
    /// Per-token read timeout, not an overall generation deadline.
    timeout_ms: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    frequency_penalty: Option<f32>,
    max_generated_tokens: Option<u32>,
    do_sample: Option<bool>,
    seed: Option<u32>,
}

pub(crate) fn api_ok(data: Value) -> Response {
    Json(json!({"ok": true, "error": null, "data": data})).into_response()
}

pub(crate) fn api_error(status: StatusCode, code: &'static str, message: String) -> Response {
    (
        status,
        Json(json!({"ok": false, "error": code, "message": message})),
    )
        .into_response()
}

pub(crate) fn auth_error(state: &AppState, headers: &HeaderMap) -> Option<Response> {
    if check_auth(state, headers) {
        return None;
    }
    Some(
        (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "invalid auth token"})),
        )
            .into_response(),
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LlmKey {
    model_path: String,
    lora_name: Option<String>,
    optimize_memory_on_device: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VlmKey {
    model_path: String,
    optimize_memory_on_device: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ActiveHandle {
    InferModel(String),
    Llm(LlmKey),
    Vlm(VlmKey),
}

enum ResidentGenAi {
    Llm { key: LlmKey, handle: Llm },
    Vlm { key: VlmKey, handle: Vlm },
}

impl ResidentGenAi {
    fn hef_path(&self) -> &str {
        match self {
            Self::Llm { key, .. } => &key.model_path,
            Self::Vlm { key, .. } => &key.model_path,
        }
    }

    fn matches_active(&self, active: &ActiveHandle) -> bool {
        matches!((self, active),
            (Self::Llm { key, .. }, ActiveHandle::Llm(active_key)) if key == active_key
        ) || matches!((self, active),
            (Self::Vlm { key, .. }, ActiveHandle::Vlm(active_key)) if key == active_key
        )
    }
}

#[derive(Default)]
pub(crate) struct DeviceCtx {
    // No eviction: 1-2 InferModel handles are hardware-verified; 3+ is extrapolated.
    infer_models: HashMap<String, ShimYolo>,
    genai: Option<ResidentGenAi>,
    active: Option<ActiveHandle>,
    panic_discards: usize,
    degraded: bool,
}

impl DeviceCtx {
    fn infer_model(&mut self, hef_path: &str) -> HailoRtResult<&mut ShimYolo> {
        if !self.infer_models.contains_key(hef_path) {
            self.infer_models
                .insert(hef_path.to_string(), ShimYolo::create(hef_path)?);
        }
        self.active = Some(ActiveHandle::InferModel(hef_path.to_string()));
        Ok(self
            .infer_models
            .get_mut(hef_path)
            .expect("inserted or existing InferModel handle"))
    }

    fn llm(
        &mut self,
        model_path: &str,
        lora_name: Option<&str>,
        optimize_memory_on_device: bool,
    ) -> HailoRtResult<&mut Llm> {
        let key = LlmKey {
            model_path: model_path.to_string(),
            lora_name: lora_name.map(str::to_string),
            optimize_memory_on_device,
        };
        if self.genai.is_none() {
            self.genai = Some(ResidentGenAi::Llm {
                handle: Llm::create(model_path, lora_name, optimize_memory_on_device)?,
                key: key.clone(),
            });
        }
        if !matches!(self.genai.as_ref(), Some(ResidentGenAi::Llm { key: loaded, .. }) if loaded == &key)
        {
            return Err(HailoRtError::GenAiConflict {
                requested: model_path.to_string(),
                loaded: self
                    .genai
                    .as_ref()
                    .expect("GenAI resident checked above")
                    .hef_path()
                    .to_string(),
            });
        }
        self.active = Some(ActiveHandle::Llm(key));
        match self.genai.as_mut() {
            Some(ResidentGenAi::Llm { handle, .. }) => Ok(handle),
            _ => unreachable!("matching LLM resident checked above"),
        }
    }

    fn llm_for_generation(
        &mut self,
        model_path: &str,
        lora_name: Option<&str>,
        optimize_memory_on_device: bool,
    ) -> HailoRtResult<&mut Llm> {
        let llm = self.llm(model_path, lora_name, optimize_memory_on_device)?;
        llm.clear_context()?;
        Ok(llm)
    }

    fn vlm(
        &mut self,
        model_path: &str,
        optimize_memory_on_device: bool,
    ) -> HailoRtResult<&mut Vlm> {
        let key = VlmKey {
            model_path: model_path.to_string(),
            optimize_memory_on_device,
        };
        if self.genai.is_none() {
            self.genai = Some(ResidentGenAi::Vlm {
                handle: Vlm::create(model_path, optimize_memory_on_device)?,
                key: key.clone(),
            });
        }
        if !matches!(self.genai.as_ref(), Some(ResidentGenAi::Vlm { key: loaded, .. }) if loaded == &key)
        {
            return Err(HailoRtError::GenAiConflict {
                requested: model_path.to_string(),
                loaded: self
                    .genai
                    .as_ref()
                    .expect("GenAI resident checked above")
                    .hef_path()
                    .to_string(),
            });
        }
        self.active = Some(ActiveHandle::Vlm(key));
        match self.genai.as_mut() {
            Some(ResidentGenAi::Vlm { handle, .. }) => Ok(handle),
            _ => unreachable!("matching VLM resident checked above"),
        }
    }

    fn vlm_for_generation(
        &mut self,
        model_path: &str,
        optimize_memory_on_device: bool,
    ) -> HailoRtResult<&mut Vlm> {
        let vlm = self.vlm(model_path, optimize_memory_on_device)?;
        vlm.clear_context()?;
        Ok(vlm)
    }

    pub(crate) fn speech2text(&mut self, model_path: &str) -> HailoRtResult<Speech2Text> {
        self.active = None;
        Speech2Text::create(model_path)
    }

    fn discard_active_after_panic(&mut self) -> HailoTaskError {
        let Some(active) = self.active.take() else {
            return HailoTaskError::Panicked;
        };
        if self.panic_discards >= MAX_PANIC_HANDLE_DISCARDS {
            self.degraded = true;
            return HailoTaskError::Unavailable(HAILORT_DEGRADED_MESSAGE);
        }
        match &active {
            ActiveHandle::InferModel(hef_path) => {
                self.infer_models.remove(hef_path);
            }
            ActiveHandle::Llm(_) | ActiveHandle::Vlm(_) => {
                if self
                    .genai
                    .as_ref()
                    .is_some_and(|resident| resident.matches_active(&active))
                {
                    self.genai = None;
                }
            }
        }
        self.panic_discards += 1;
        HailoTaskError::Panicked
    }

    fn run<T, F>(&mut self, task: F) -> Result<T, HailoTaskError>
    where
        F: FnOnce(&mut Self) -> HailoRtResult<T>,
    {
        if self.degraded {
            return Err(HailoTaskError::Unavailable(HAILORT_DEGRADED_MESSAGE));
        }
        self.active = None;
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| task(self))) {
            Ok(result) => {
                self.active = None;
                result.map_err(HailoTaskError::HailoRt)
            }
            Err(_) => Err(self.discard_active_after_panic()),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum HailoTaskError {
    #[error(transparent)]
    HailoRt(#[from] HailoRtError),
    #[error("HailoRT task panicked")]
    Panicked,
    #[error("{0}")]
    Unavailable(&'static str),
}

impl HailoTaskError {
    fn status_code(&self) -> StatusCode {
        match self {
            Self::HailoRt(HailoRtError::GenAiConflict { .. }) => StatusCode::CONFLICT,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::HailoRt(_) | Self::Panicked => StatusCode::BAD_REQUEST,
        }
    }
}

pub(crate) fn hailort_api_error(error: HailoTaskError, code: &'static str) -> Response {
    api_error(error.status_code(), code, error.to_string())
}

// Only this closure and its Send result cross threads; DeviceCtx and device handles stay here.
type DeviceTask = Box<dyn FnOnce(&mut DeviceCtx) + Send + 'static>;

fn hailort_device_sender() -> &'static mpsc::Sender<DeviceTask> {
    static SENDER: OnceLock<mpsc::Sender<DeviceTask>> = OnceLock::new();
    SENDER.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<DeviceTask>();
        std::thread::Builder::new()
            .name("hailort-device".to_string())
            .spawn(move || {
                let mut ctx = DeviceCtx::default();
                for task in receiver {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        task(&mut ctx);
                    }));
                }
            })
            .expect("failed to start HailoRT device thread");
        sender
    })
}

pub(crate) async fn run_hailort_task<T, F>(task: F) -> Result<T, HailoTaskError>
where
    T: Send + 'static,
    F: FnOnce(&mut DeviceCtx) -> HailoRtResult<T> + Send + 'static,
{
    // Device tasks must never wait for other device work: this single queue is non-reentrant.
    let (sender, receiver) = tokio::sync::oneshot::channel();
    hailort_device_sender()
        .send(Box::new(move |ctx| {
            let _ = sender.send(ctx.run(task));
        }))
        .map_err(|_| HailoTaskError::Unavailable("HailoRT device thread stopped"))?;
    receiver.await.map_err(|_| HailoTaskError::Panicked)?
}

/// Runs bounded media parsing on Tokio's blocking pool after atomically
/// reserving its worst-case working-set budget. Rejecting rather than queuing
/// work prevents a request burst from creating an unbounded preprocessing queue.
pub(crate) async fn run_media_preprocessing<T, E, F>(
    reservation_bytes: u64,
    task: F,
) -> Result<T, MediaPreprocessError<E>>
where
    T: Send + 'static,
    E: Send + 'static,
    F: FnOnce() -> Result<T, E> + Send + 'static,
{
    let permit = media_preprocessing_semaphore()
        .try_acquire_many_owned(media_preprocessing_permits(reservation_bytes))
        .map_err(|_| MediaPreprocessError::Busy)?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        task()
    })
    .await
    .map_err(|error| {
        MediaPreprocessError::Join(format!("media preprocessing task failed: {error}"))
    })?
    .map_err(MediaPreprocessError::Task)
}

fn validate_text_len(text: &str) -> Option<Response> {
    if text.len() > MAX_TEXT_BYTES {
        return Some(api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "text_too_large",
            format!("text exceeds {MAX_TEXT_BYTES} bytes"),
        ));
    }
    None
}

fn env_or_default_path(env_key: &str, default_name: &str) -> PathBuf {
    if let Some(path) = std::env::var_os(env_key) {
        return PathBuf::from(path);
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/pi"));
    home.join("hailo_models").join(default_name)
}

fn yolo_hef_path(query: &HefQuery) -> PathBuf {
    query
        .hef_path
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| env_or_default_path("HAILO_YOLO_HEF", "yolov8n.hef"))
}

pub(crate) fn s2t_hef_path(path: Option<&str>) -> PathBuf {
    path.map(PathBuf::from)
        .unwrap_or_else(|| env_or_default_path("HAILO_S2T_HEF", "Whisper-Tiny.hef"))
}

fn llm_hef_path(path: Option<&str>) -> PathBuf {
    path.map(PathBuf::from)
        .unwrap_or_else(|| env_or_default_path("HAILO_LLM_HEF", "Llama3.2-1B-Instruct.hef"))
}

fn vlm_hef_path(path: Option<&str>) -> PathBuf {
    path.map(PathBuf::from)
        .unwrap_or_else(|| env_or_default_path("HAILO_VLM_HEF", "qwen2-vl-2b-instruct.hef"))
}

fn clip_image_hef_path(path: Option<&str>) -> PathBuf {
    path.map(PathBuf::from)
        .unwrap_or_else(|| env_or_default_path("HAILO_CLIP_HEF", "clip_vit_b_16_image_encoder.hef"))
}

fn clip_text_model_dir(path: Option<&str>) -> PathBuf {
    path.map(PathBuf::from).unwrap_or_else(|| {
        std::env::var_os("HAILO_CLIP_TEXT_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let home = std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("/home/pi"));
                home.join(".cache")
                    .join("yu_ai_manager")
                    .join("clip_onnx")
                    .join("Xenova_clip-vit-base-patch16")
            })
    })
}

/// Upper bound on decoded (pre-resize) image dimensions. VLM inputs are
/// downscaled to a few hundred pixels per side, so legitimate photos never
/// need to exceed this — it exists solely to reject decompression bombs
/// (a tiny compressed file that decodes to a huge pixel buffer) before the
/// decoder allocates the full frame.
const MAX_DECODED_IMAGE_DIMENSION: u32 = 4096;
/// Upper bound on total decoded bytes across *all* frames in one request
/// (not per frame — with `MAX_FRAMES` frames, a per-frame-only cap still
/// lets `MAX_FRAMES * per_frame_cap` accumulate in memory at once). Each
/// decode call spends from this shared budget and fails once it is
/// exhausted, so worst case for the whole request is bounded regardless of
/// how many frames are sent.
const MAX_TOTAL_DECODED_IMAGE_BYTES: u64 = 64 * 1024 * 1024;

/// Decodes a base64-encoded image (JPEG/PNG/WebP) without resizing.
/// Resizing happens after the VLM model is loaded — its expected input
/// shape is model-specific (queried via `Vlm::input_frame_info()`), not a
/// fixed constant like the Python extension's hardcoded 336x336 (which only
/// applies to qwen2-vl-2b-instruct).
///
/// `remaining_budget` is shared across all frames in a request: this call
/// both caps the decoder's allocation at whatever is left and, on success,
/// deducts the decoded frame's actual size — bounding the *request's* total
/// decoded memory, not just any single frame's.
fn decode_base64_image(
    base64_frame: &str,
    remaining_budget: &mut u64,
) -> Result<image::DynamicImage, String> {
    if base64_frame.len() > MAX_FRAME_BASE64_BYTES {
        return Err(format!(
            "frame exceeds {MAX_FRAME_BASE64_BYTES} base64 bytes"
        ));
    }
    if *remaining_budget == 0 {
        return Err("request-level decoded image budget exhausted".to_string());
    }
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(base64_frame)
        .map_err(|error| format!("invalid base64 frame: {error}"))?;

    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_DECODED_IMAGE_DIMENSION);
    limits.max_image_height = Some(MAX_DECODED_IMAGE_DIMENSION);
    limits.max_alloc = Some(*remaining_budget);

    let mut reader = image::ImageReader::new(std::io::Cursor::new(&bytes))
        .with_guessed_format()
        .map_err(|error| format!("invalid image: {error}"))?;
    reader.limits(limits);
    let decoded = reader
        .decode()
        .map_err(|error| format!("invalid image (or exceeds decode budget): {error}"))?;
    *remaining_budget = remaining_budget.saturating_sub(decoded.as_bytes().len() as u64);
    Ok(decoded)
}

/// Resizes (no letterboxing — a plain squash, matching the Python
/// extension's `cv2.resize()` behavior) and flattens to raw interleaved
/// RGB8 bytes for the shim's frame buffer.
fn resize_frame(image: &image::DynamicImage, width: u32, height: u32) -> Vec<u8> {
    image::imageops::resize(
        &image.to_rgb8(),
        width,
        height,
        image::imageops::FilterType::Triangle,
    )
    .into_raw()
}

fn yolo_metadata_json(metadata: &YoloModelMetadata, hef_path: &str) -> Value {
    json!({
        "hef_path": hef_path,
        "inputs": metadata.inputs.iter().map(vstream_json).collect::<Vec<_>>(),
        "outputs": metadata.outputs.iter().map(vstream_json).collect::<Vec<_>>(),
    })
}

fn vstream_json(info: &crate::hailort::VStreamInfo) -> Value {
    json!({
        "name": info.name,
        "direction": match info.direction {
            crate::hailort::VStreamDirection::Input => "input",
            crate::hailort::VStreamDirection::Output => "output",
        },
        "shape": info.shape.dimensions(),
        "frame_size": info.frame_size,
        "format_type": info.format_type,
        "quant": {
            "zero_point": info.quant.zero_point,
            "scale": info.quant.scale,
        },
    })
}

async fn yolo_metadata(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HefQuery>,
) -> Response {
    if let Some(response) = auth_error(&state, &headers) {
        return response;
    }
    let hef_path = yolo_hef_path(&query);
    let hef_path_str = hef_path.to_string_lossy().to_string();
    let result = run_hailort_task({
        let hef_path_str = hef_path_str.clone();
        move |ctx| resident_yolo_metadata(ctx.infer_model(&hef_path_str)?)
    })
    .await;
    match result {
        Ok(metadata) => api_ok(yolo_metadata_json(&metadata, &hef_path_str)),
        Err(error) => hailort_api_error(error, "hailort_yolo_metadata_failed"),
    }
}

async fn yolo_smoke_zero(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HefQuery>,
) -> Response {
    if let Some(response) = auth_error(&state, &headers) {
        return response;
    }
    let hef_path = yolo_hef_path(&query);
    let hef_path_str = hef_path.to_string_lossy().to_string();
    let metadata = match run_hailort_task({
        let hef_path_str = hef_path_str.clone();
        move |ctx| resident_yolo_metadata(ctx.infer_model(&hef_path_str)?)
    })
    .await
    {
        Ok(metadata) => metadata,
        Err(error) => {
            return hailort_api_error(error, "hailort_yolo_metadata_failed");
        }
    };
    let input_len = match metadata.inputs.first() {
        Some(input) => input.frame_size,
        None => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "hailort_yolo_invalid_metadata",
                "missing YOLO input".to_string(),
            );
        }
    };
    let result = run_hailort_task({
        let hef_path_str = hef_path_str.clone();
        move |ctx| {
            let model = ctx.infer_model(&hef_path_str)?;
            run_yolo_once(model, &vec![0u8; input_len])
        }
    })
    .await;
    match result {
        Ok(result) => api_ok(json!({
            "hef_path": hef_path_str,
            "input_frame_size": input_len,
            "outputs": result.outputs.iter().map(|output| json!({
                "name": output.name,
                "bytes": output.data.len(),
                "frame_size": output.info.frame_size,
            })).collect::<Vec<_>>(),
        })),
        Err(error) => hailort_api_error(error, "hailort_yolo_smoke_failed"),
    }
}

async fn yolo_detect(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<YoloDetectRequest>,
) -> Response {
    if let Some(response) = auth_error(&state, &headers) {
        return response;
    }

    use base64::Engine as _;

    let conf_threshold = body.conf_threshold.unwrap_or(0.25);
    let iou_threshold = body.iou_threshold.unwrap_or(0.45);
    let num_classes = body.num_classes.unwrap_or(80);
    // This must match the caller's letterbox target (currently always 640).
    let input_size = body.input_size.unwrap_or(640);
    let scale_info = infer_core::yolo_postprocess::ScaleInfo {
        orig_w: body.orig_w.unwrap_or(input_size),
        orig_h: body.orig_h.unwrap_or(input_size),
        scale: body.scale.unwrap_or(0.0),
        pad_x: body.pad_x.unwrap_or(0.0),
        pad_y: body.pad_y.unwrap_or(0.0),
    };

    if body.input_base64.len() > MAX_FRAME_BASE64_BYTES {
        return api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "hailort_yolo_detect_input_too_large",
            format!("input_base64 exceeds {MAX_FRAME_BASE64_BYTES} bytes"),
        );
    }

    let hef_path = body
        .hef_path
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| env_or_default_path("HAILO_YOLO_HEF", "yolov8n.hef"));
    let hef_path_str = hef_path.to_string_lossy().to_string();
    let input_bytes = match base64::engine::general_purpose::STANDARD.decode(&body.input_base64) {
        Ok(input_bytes) => input_bytes,
        Err(error) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "hailort_yolo_detect_failed",
                format!("invalid input_base64: {error}"),
            );
        }
    };
    let metadata = match run_hailort_task({
        let hef_path_str = hef_path_str.clone();
        move |ctx| resident_yolo_metadata(ctx.infer_model(&hef_path_str)?)
    })
    .await
    {
        Ok(metadata) => metadata,
        Err(error) => {
            return hailort_api_error(error, "hailort_yolo_metadata_failed");
        }
    };
    let input_frame_size = match metadata.inputs.first() {
        Some(input) => input.frame_size,
        None => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "hailort_yolo_invalid_metadata",
                "missing YOLO input".to_string(),
            );
        }
    };
    if input_bytes.len() != input_frame_size {
        return api_error(
            StatusCode::BAD_REQUEST,
            "hailort_yolo_detect_input_size_mismatch",
            format!(
                "YOLO input size mismatch: expected {input_frame_size} bytes, got {}",
                input_bytes.len()
            ),
        );
    }

    let result = run_hailort_task({
        let hef_path_str = hef_path_str.clone();
        move |ctx| {
            let model = ctx.infer_model(&hef_path_str)?;
            run_yolo_once(model, &input_bytes)
        }
    })
    .await;
    match result {
        Ok(result) => {
            let outputs = result
                .outputs
                .into_iter()
                .map(|output| infer_core::yolo_postprocess::YoloOutputBuffer {
                    data: output.data,
                    meta: infer_core::yolo_postprocess::QuantMeta {
                        name: output.name,
                        scale: output.info.quant.scale,
                        zero_point: output.info.quant.zero_point,
                        is_float32: output.info.format_type == 3,
                        shape: output.info.shape.dimensions(),
                        format_type: output.info.format_type as u64,
                    },
                })
                .collect::<Vec<_>>();
            match infer_core::yolo_postprocess::postprocess_yolo_outputs(
                &outputs,
                conf_threshold,
                iou_threshold,
                num_classes,
                input_size,
                &scale_info,
            ) {
                Ok(detections) => api_ok(json!({"detections": detections})),
                Err(error) => {
                    api_error(StatusCode::BAD_REQUEST, "hailort_yolo_detect_failed", error)
                }
            }
        }
        Err(error) => hailort_api_error(error, "hailort_yolo_detect_failed"),
    }
}

async fn clip_image(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<ClipImageRequest>,
) -> Response {
    if let Some(response) = auth_error(&state, &headers) {
        return response;
    }
    let ClipImageRequest {
        hef_path,
        image_base64,
    } = body;
    if image_base64.len() > MAX_FRAME_BASE64_BYTES {
        return api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "clip_image_too_large",
            format!("image_base64 exceeds {MAX_FRAME_BASE64_BYTES} bytes"),
        );
    }

    let hef_path = clip_image_hef_path(hef_path.as_deref());
    let hef_path_str = hef_path.to_string_lossy().to_string();
    let metadata = match run_hailort_task({
        let hef_path_str = hef_path_str.clone();
        move |ctx| clip_image_metadata(ctx.infer_model(&hef_path_str)?)
    })
    .await
    {
        Ok(metadata) => metadata,
        Err(error) => {
            return hailort_api_error(error, "hailort_clip_image_metadata_failed");
        }
    };
    let dimensions = metadata.input.shape.dimensions();
    let (height, width, channels) = (dimensions[0], dimensions[1], dimensions[2]);
    if width == 0 || height == 0 || channels != 3 {
        return api_error(
            StatusCode::BAD_REQUEST,
            "hailort_clip_image_invalid_metadata",
            "CLIP image input must use a non-empty HWC RGB tensor".to_string(),
        );
    }
    let input =
        match run_media_preprocessing(MAX_IMAGE_PREPROCESSING_RESERVATION_BYTES, move || {
            let mut decode_budget = MAX_TOTAL_DECODED_IMAGE_BYTES;
            let image = decode_base64_image(&image_base64, &mut decode_budget)?;
            let input = resize_frame(&image, width as u32, height as u32);
            if input.len() != metadata.input.frame_size {
                return Err(format!(
                    "CLIP image input size mismatch: expected {} bytes, got {}",
                    metadata.input.frame_size,
                    input.len()
                ));
            }
            Ok(input)
        })
        .await
        {
            Ok(input) => input,
            Err(MediaPreprocessError::Busy) => {
                return api_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "media_preprocessing_busy",
                    "media preprocessing capacity is temporarily exhausted".to_string(),
                )
            }
            Err(MediaPreprocessError::Task(error)) => {
                return api_error(StatusCode::BAD_REQUEST, "clip_image_invalid_image", error)
            }
            Err(MediaPreprocessError::Join(error)) => {
                return api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "media_preprocessing_failed",
                    error,
                )
            }
        };
    let vector = match run_hailort_task({
        let hef_path_str = hef_path_str.clone();
        move |ctx| {
            let model = ctx.infer_model(&hef_path_str)?;
            run_clip_image_once(model, &input)
        }
    })
    .await
    {
        Ok(vector) => vector,
        Err(error) => {
            return hailort_api_error(error, "hailort_clip_image_failed");
        }
    };
    api_ok(json!({
        "hef_path": hef_path_str,
        "vector": vector,
        "dim": 512,
    }))
}

async fn clip_text(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<ClipTextRequest>,
) -> Response {
    if let Some(response) = auth_error(&state, &headers) {
        return response;
    }
    if let Some(response) = validate_text_len(&body.text) {
        return response;
    }

    let model_dir = clip_text_model_dir(body.model_dir.as_deref());
    let cache_key = model_dir.to_string_lossy().to_string();
    let encoders = state.clip_text.clone();
    let text = body.text;
    let result =
        tokio::task::spawn_blocking(move || -> Result<Vec<f32>, infer_core::InferError> {
            let encoder = {
                let mut cached = encoders.write().map_err(|_| {
                    infer_core::InferError::InvalidModelOutput(
                        "CLIP text encoder cache lock poisoned".to_string(),
                    )
                })?;
                if let Some(encoder) = cached.get(&cache_key) {
                    encoder.clone()
                } else {
                    let encoder = Arc::new(infer_core::ClipTextEncoder::new(&model_dir)?);
                    cached.insert(cache_key, encoder.clone());
                    encoder
                }
            };
            encoder.encode(&text)
        })
        .await;

    match result {
        Ok(Ok(vector)) => api_ok(json!({"vector": vector, "dim": 512})),
        Ok(Err(infer_core::InferError::ModelNotDownloaded(_))) => api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "clip_text_model_not_downloaded",
            "CLIP text model is not downloaded".to_string(),
        ),
        Ok(Err(error)) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "clip_text_inference_failed",
            error.to_string(),
        ),
        Err(error) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "clip_text_task_failed",
            format!("CLIP text task failed: {error}"),
        ),
    }
}

async fn speech2text_tokenize(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<TokenizeRequest>,
) -> Response {
    if let Some(response) = auth_error(&state, &headers) {
        return response;
    }
    if let Some(response) = validate_text_len(&body.text) {
        return response;
    }
    let hef_path = s2t_hef_path(body.hef_path.as_deref());
    let hef_path_str = hef_path.to_string_lossy().to_string();
    let result = run_hailort_task({
        let hef_path_str = hef_path_str.clone();
        let text = body.text;
        move |ctx| {
            ctx.speech2text(&hef_path_str)
                .and_then(|mut s2t| s2t.tokenize(&text))
        }
    })
    .await;
    match result {
        Ok(tokens) => api_ok(json!({"hef_path": hef_path_str, "tokens": tokens})),
        Err(error) => hailort_api_error(error, "hailort_s2t_tokenize_failed"),
    }
}

async fn llm_tokenize(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<TokenizeRequest>,
) -> Response {
    if let Some(response) = auth_error(&state, &headers) {
        return response;
    }
    if let Some(response) = validate_text_len(&body.text) {
        return response;
    }
    let hef_path = llm_hef_path(body.hef_path.as_deref());
    let hef_path_str = hef_path.to_string_lossy().to_string();
    let result = run_hailort_task({
        let hef_path_str = hef_path_str.clone();
        let text = body.text;
        move |ctx| ctx.llm(&hef_path_str, None, false)?.tokenize(&text)
    })
    .await;
    match result {
        Ok(tokens) => api_ok(json!({"hef_path": hef_path_str, "tokens": tokens})),
        Err(error) => hailort_api_error(error, "hailort_llm_tokenize_failed"),
    }
}

async fn llm_generate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<LlmGenerateRequest>,
) -> Response {
    if let Some(response) = auth_error(&state, &headers) {
        return response;
    }
    if body.prompt.len() > MAX_PROMPT_BYTES {
        return api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "prompt_too_large",
            format!("prompt exceeds {MAX_PROMPT_BYTES} bytes"),
        );
    }
    let timeout_ms = body.timeout_ms.unwrap_or(30_000).min(MAX_TIMEOUT_MS);
    let hef_path = llm_hef_path(body.hef_path.as_deref());
    let hef_path_str = hef_path.to_string_lossy().to_string();
    let result = run_hailort_task({
        let hef_path_str = hef_path_str.clone();
        let prompt = body.prompt;
        move |ctx| {
            ctx.llm_for_generation(&hef_path_str, None, false)?
                .generate_text(&prompt, timeout_ms)
        }
    })
    .await;
    match result {
        Ok(text) => api_ok(json!({"hef_path": hef_path_str, "text": text})),
        Err(error) => hailort_api_error(error, "hailort_llm_generate_failed"),
    }
}

async fn llm_generate_stream(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<LlmGenerateStreamRequest>,
) -> Response {
    if let Some(response) = auth_error(&state, &headers) {
        return response;
    }
    if body.messages.is_empty() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "llm_messages_required",
            "at least one message is required".to_string(),
        );
    }
    if body.messages.len() > MAX_LLM_MESSAGES {
        return api_error(
            StatusCode::BAD_REQUEST,
            "llm_too_many_messages",
            format!("at most {MAX_LLM_MESSAGES} messages are allowed"),
        );
    }
    let total_content_bytes: usize = body.messages.iter().map(|m| m.content.len()).sum();
    if total_content_bytes > MAX_PROMPT_BYTES {
        return api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "prompt_too_large",
            format!("combined message content exceeds {MAX_PROMPT_BYTES} bytes"),
        );
    }
    if body.tools.len() > MAX_LLM_TOOLS {
        return api_error(
            StatusCode::BAD_REQUEST,
            "llm_too_many_tools",
            format!("at most {MAX_LLM_TOOLS} tools are allowed"),
        );
    }
    let tools: Vec<String> = body
        .tools
        .iter()
        .map(serde_json::Value::to_string)
        .collect();
    let total_tool_bytes: usize = tools.iter().map(String::len).sum();
    if total_tool_bytes > MAX_PROMPT_BYTES {
        return api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "tools_too_large",
            format!("combined tool definitions exceed {MAX_PROMPT_BYTES} bytes"),
        );
    }
    if let Some(max_generated_tokens) = body.max_generated_tokens {
        if max_generated_tokens > MAX_LLM_GENERATED_TOKENS {
            return api_error(
                StatusCode::BAD_REQUEST,
                "llm_max_generated_tokens_too_large",
                format!("max_generated_tokens exceeds {MAX_LLM_GENERATED_TOKENS}"),
            );
        }
    }

    let timeout_ms = body.timeout_ms.unwrap_or(30_000).min(MAX_TIMEOUT_MS);
    let hef_path = llm_hef_path(body.hef_path.as_deref());
    let hef_path_str = hef_path.to_string_lossy().to_string();
    let messages: Vec<LlmChatMessage> = body
        .messages
        .into_iter()
        .map(|m| LlmChatMessage {
            role: m.role,
            content: m.content,
        })
        .collect();
    let params = LlmGenerationParams {
        temperature: body.temperature,
        top_p: body.top_p,
        top_k: body.top_k,
        frequency_penalty: body.frequency_penalty,
        max_generated_tokens: body.max_generated_tokens,
        do_sample: body.do_sample,
        seed: body.seed,
    };

    if let Err(error) = run_hailort_task({
        let hef_path_str = hef_path_str.clone();
        move |ctx| ctx.llm(&hef_path_str, None, false).map(|_| ())
    })
    .await
    {
        return hailort_api_error(error, "hailort_llm_generate_failed");
    }

    // SSE token delivery stays unbounded in Stage 2; do not add device-thread backpressure here.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    tokio::spawn(async move {
        let work_tx = tx.clone();
        let result = run_hailort_task(move |ctx| {
            let llm = ctx.llm_for_generation(&hef_path_str, None, false)?;
            // LlmStream borrows the handle, so the whole generation stays in one device task.
            let mut stream = LlmStream::start(llm, &messages, &tools, params)?;
            let mut full_text = String::new();
            for _ in 0..MAX_LLM_STREAM_READS {
                let (token, status) = stream.read_next(timeout_ms)?;
                full_text.push_str(&token);
                let _ = work_tx.send(sse_data_event(json!({"token": token})));
                if !status.is_generating() {
                    let _ = work_tx.send(sse_data_event(
                        json!({"done": true, "full_text": full_text}),
                    ));
                    return Ok(());
                }
            }
            let _ = work_tx.send(sse_data_event(
                json!({"error": format!("generation exceeded {MAX_LLM_STREAM_READS} token reads without reaching a terminal state")}),
            ));
            Ok(())
        })
        .await;
        if let Err(error) = result {
            let _ = tx.send(sse_data_event(sse_hailort_error(error)));
        }
    });

    let body_stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|chunk| (Ok::<_, std::io::Error>(axum::body::Bytes::from(chunk)), rx))
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
        .header(axum::http::header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(body_stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

/// Cheap request validation is kept on the async handler; parsing and image
/// allocation are delegated to the bounded preprocessing pool below.
fn validate_vlm_input(prompt: &str, frames: &[String]) -> Result<(), Box<Response>> {
    if prompt.len() > MAX_PROMPT_BYTES {
        return Err(Box::new(api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "prompt_too_large",
            format!("prompt exceeds {MAX_PROMPT_BYTES} bytes"),
        )));
    }
    if frames.is_empty() {
        return Err(Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "vlm_frames_required",
            "at least one image frame is required".to_string(),
        )));
    }
    if frames.len() > MAX_FRAMES {
        return Err(Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "vlm_too_many_frames",
            format!("at most {MAX_FRAMES} frames are allowed"),
        )));
    }
    Ok(())
}

fn decode_vlm_frames(frames: Vec<String>) -> Result<Vec<image::DynamicImage>, String> {
    let mut decode_budget = MAX_TOTAL_DECODED_IMAGE_BYTES;
    frames
        .iter()
        .map(|frame| decode_base64_image(frame, &mut decode_budget))
        .collect()
}

async fn vlm_generate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<VlmGenerateRequest>,
) -> Response {
    if let Some(response) = auth_error(&state, &headers) {
        return response;
    }
    if let Err(response) = validate_vlm_input(&body.prompt, &body.frames) {
        return *response;
    }
    let frames = body.frames;
    let images =
        match run_media_preprocessing(MAX_IMAGE_PREPROCESSING_RESERVATION_BYTES, move || {
            decode_vlm_frames(frames)
        })
        .await
        {
            Ok(images) => images,
            Err(MediaPreprocessError::Busy) => {
                return api_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "media_preprocessing_busy",
                    "media preprocessing capacity is temporarily exhausted".to_string(),
                )
            }
            Err(MediaPreprocessError::Task(error)) => {
                return api_error(StatusCode::BAD_REQUEST, "vlm_invalid_frame", error)
            }
            Err(MediaPreprocessError::Join(error)) => {
                return api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "media_preprocessing_failed",
                    error,
                )
            }
        };

    let timeout_ms = body.timeout_ms.unwrap_or(30_000).min(MAX_TIMEOUT_MS);
    let hef_path = vlm_hef_path(body.hef_path.as_deref());
    let hef_path_str = hef_path.to_string_lossy().to_string();
    let result = run_hailort_task({
        let hef_path_str = hef_path_str.clone();
        let prompt = body.prompt;
        move |ctx| {
            let vlm = ctx.vlm_for_generation(&hef_path_str, false)?;
            let info = vlm.input_frame_info()?;
            // Keep resizing here in Stage 3: splitting the device query from preprocessing
            // is a separate latency optimization.
            let frames: Vec<Vec<u8>> = images
                .iter()
                .map(|image| resize_frame(image, info.width, info.height))
                .collect();
            vlm.generate_text(&prompt, &frames, timeout_ms)
        }
    })
    .await;
    match result {
        Ok(text) => api_ok(json!({"hef_path": hef_path_str, "text": text})),
        Err(error) => hailort_api_error(error, "hailort_vlm_generate_failed"),
    }
}

fn sse_data_event(payload: Value) -> String {
    format!("data: {payload}\n\n")
}

fn sse_hailort_error(error: HailoTaskError) -> Value {
    let message = error.to_string();
    if error.status_code() == StatusCode::SERVICE_UNAVAILABLE {
        json!({"error": message, "status": StatusCode::SERVICE_UNAVAILABLE.as_u16()})
    } else {
        json!({"error": message})
    }
}

async fn vlm_generate_stream(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<VlmGenerateStreamRequest>,
) -> Response {
    if let Some(response) = auth_error(&state, &headers) {
        return response;
    }
    if let Some(max_generated_tokens) = body.max_generated_tokens {
        if max_generated_tokens > MAX_VLM_GENERATED_TOKENS {
            return api_error(
                StatusCode::BAD_REQUEST,
                "vlm_max_generated_tokens_too_large",
                format!("max_generated_tokens exceeds {MAX_VLM_GENERATED_TOKENS}"),
            );
        }
    }
    if let Err(response) = validate_vlm_input(&body.prompt, &body.frames) {
        return *response;
    }
    let frames = body.frames;
    let images =
        match run_media_preprocessing(MAX_IMAGE_PREPROCESSING_RESERVATION_BYTES, move || {
            decode_vlm_frames(frames)
        })
        .await
        {
            Ok(images) => images,
            Err(MediaPreprocessError::Busy) => {
                return api_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "media_preprocessing_busy",
                    "media preprocessing capacity is temporarily exhausted".to_string(),
                )
            }
            Err(MediaPreprocessError::Task(error)) => {
                return api_error(StatusCode::BAD_REQUEST, "vlm_invalid_frame", error)
            }
            Err(MediaPreprocessError::Join(error)) => {
                return api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "media_preprocessing_failed",
                    error,
                )
            }
        };

    let timeout_ms = body.timeout_ms.unwrap_or(30_000).min(MAX_TIMEOUT_MS);
    let hef_path = vlm_hef_path(body.hef_path.as_deref());
    let hef_path_str = hef_path.to_string_lossy().to_string();
    let prompt = body.prompt;
    let system_prompt = body.system_prompt;
    let params = VlmGenerationParams {
        temperature: body.temperature,
        top_p: body.top_p,
        top_k: body.top_k,
        frequency_penalty: body.frequency_penalty,
        max_generated_tokens: body.max_generated_tokens,
        do_sample: body.do_sample,
        seed: body.seed,
    };

    if let Err(error) = run_hailort_task({
        let hef_path_str = hef_path_str.clone();
        move |ctx| ctx.vlm(&hef_path_str, false).map(|_| ())
    })
    .await
    {
        return hailort_api_error(error, "hailort_vlm_generate_failed");
    }

    // SSE token delivery stays unbounded in Stage 2; do not add device-thread backpressure here.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    tokio::spawn(async move {
        let work_tx = tx.clone();
        let result = run_hailort_task(move |ctx| {
            let vlm = ctx.vlm_for_generation(&hef_path_str, false)?;
            let info = vlm.input_frame_info()?;
            // Keep resizing here in Stage 3: splitting the device query from preprocessing
            // is a separate latency optimization.
            let frames: Vec<Vec<u8>> = images
                .iter()
                .map(|image| resize_frame(image, info.width, info.height))
                .collect();
            let mut stream =
                VlmStream::start(vlm, &prompt, system_prompt.as_deref(), &frames, params)?;
            // VlmStream borrows the handle, so the whole generation stays in one device task.
            let mut full_text = String::new();
            for _ in 0..MAX_VLM_STREAM_READS {
                let (token, status) = stream.read_next(timeout_ms)?;
                full_text.push_str(&token);
                let _ = work_tx.send(sse_data_event(json!({"token": token})));
                if !status.is_generating() {
                    let _ = work_tx.send(sse_data_event(
                        json!({"done": true, "full_text": full_text}),
                    ));
                    return Ok(());
                }
            }
            let _ = work_tx.send(sse_data_event(
                json!({"error": format!("generation exceeded {MAX_VLM_STREAM_READS} token reads without reaching a terminal state")}),
            ));
            Ok(())
        })
        .await;
        if let Err(error) = result {
            let _ = tx.send(sse_data_event(sse_hailort_error(error)));
        }
    });

    let body_stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|chunk| (Ok::<_, std::io::Error>(axum::body::Bytes::from(chunk)), rx))
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
        .header(axum::http::header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(body_stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

async fn infer_wd(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<InferWdRequest>,
) -> impl IntoResponse {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "invalid auth token"})),
        )
            .into_response();
    }

    let real_path = match std::fs::canonicalize(&body.path) {
        Ok(p) => p,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "invalid path"})),
            )
                .into_response()
        }
    };

    if !check_scan_roots(&state, &real_path) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"error": "path outside allowed roots"})),
        )
            .into_response();
    }

    if !check_model_id(&body.model_id) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "invalid model_id"})),
        )
            .into_response();
    }

    let model_dir = state.wd_cache_dir.join(&body.model_id);
    if !infer_core::is_model_downloaded(&state.wd_cache_dir, &body.model_id) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "model not downloaded"})),
        )
            .into_response();
    }

    let wd_infer = state.wd_infer.clone();
    let model_id = body.model_id.clone();
    let general_thr = body.general_thr;
    let character_thr = body.character_thr;

    let result = tokio::task::spawn_blocking(move || {
        let engine = {
            let mut engines = wd_infer.write().unwrap();
            engines
                .entry(model_id)
                .or_insert_with(|| {
                    Arc::new(
                        infer_core::WdInferEngine::new(&model_dir)
                            .expect("WdInferEngine init failed after is_model_downloaded check"),
                    )
                })
                .clone()
        };
        engine.run(&real_path, general_thr, character_thr)
    })
    .await;

    match result {
        Ok(Ok(tag_result)) => Json(json!({"ok": true, "data": tag_result})).into_response(),
        Ok(Err(e)) => {
            tracing::error!("yu-infer wd inference error: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "inference failed"})),
            )
                .into_response()
        }
        Err(e) => {
            tracing::error!("yu-infer spawn_blocking panic: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "internal error"})),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(hailo_stub)]
    use crate::hailort::stub_counts;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn test_state(scan_roots: Vec<PathBuf>) -> AppState {
        AppState {
            started_at: std::time::Instant::now(),
            instance_id: "test-instance".to_string(),
            scan_roots: Arc::new(RwLock::new(scan_roots)),
            auth_token: "test-token".to_string(),
            wd_cache_dir: std::env::temp_dir().join("yu-infer-test-cache"),
            wd_infer: Arc::new(RwLock::new(HashMap::new())),
            clip_text: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    #[tokio::test]
    async fn media_preprocessing_rejects_work_that_exceeds_the_global_budget() {
        let result = run_media_preprocessing::<(), (), _>(
            MAX_MEDIA_PREPROCESSING_BYTES + MEDIA_PREPROCESSING_PERMIT_BYTES,
            || Ok(()),
        )
        .await;
        assert!(matches!(result, Err(MediaPreprocessError::Busy)));
    }

    #[tokio::test]
    async fn panicking_device_work_does_not_stop_the_thread() {
        let panicked = run_hailort_task(|_| -> HailoRtResult<()> {
            panic!("device work panic");
        })
        .await;
        assert_eq!(panicked.unwrap_err().to_string(), "HailoRT task panicked");

        let next = run_hailort_task(|_| Ok(42)).await;
        assert_eq!(next.unwrap(), 42);
    }

    #[cfg(hailo_stub)]
    #[test]
    fn same_create_key_reuses_one_handle() {
        let before = stub_counts();
        let mut ctx = DeviceCtx::default();
        ctx.llm("same.hef", Some("adapter"), true).unwrap();
        ctx.llm("same.hef", Some("adapter"), true).unwrap();
        let after = stub_counts();

        assert_eq!(after.llm_create - before.llm_create, 1);
    }

    #[cfg(hailo_stub)]
    #[test]
    fn create_keys_include_non_path_arguments() {
        let mut llm_ctx = DeviceCtx::default();
        llm_ctx.llm("same.hef", None, false).unwrap();
        assert!(matches!(
            llm_ctx.llm("same.hef", Some("adapter"), false),
            Err(HailoRtError::GenAiConflict { .. })
        ));
        assert!(matches!(
            llm_ctx.llm("same.hef", None, true),
            Err(HailoRtError::GenAiConflict { .. })
        ));

        let mut vlm_ctx = DeviceCtx::default();
        vlm_ctx.vlm("same.hef", false).unwrap();
        assert!(matches!(
            vlm_ctx.vlm("same.hef", true),
            Err(HailoRtError::GenAiConflict { .. })
        ));
    }

    #[cfg(hailo_stub)]
    #[test]
    fn infer_model_allows_two_different_hefs() {
        let before = stub_counts();
        let mut ctx = DeviceCtx::default();
        ctx.infer_model("yolo.hef").unwrap();
        ctx.infer_model("clip.hef").unwrap();
        let after = stub_counts();

        assert_eq!(after.yolo_create - before.yolo_create, 2);
    }

    #[cfg(hailo_stub)]
    #[test]
    fn different_genai_hef_names_the_resident_hef() {
        let before = stub_counts();
        let mut ctx = DeviceCtx::default();
        ctx.llm("resident.hef", None, false).unwrap();
        let error = match ctx.vlm("requested.hef", false) {
            Ok(_) => panic!("expected a conflicting GenAI HEF error"),
            Err(error) => error,
        };
        let after = stub_counts();

        assert!(matches!(
            &error,
            HailoRtError::GenAiConflict { loaded, requested }
                if loaded == "resident.hef" && requested == "requested.hef"
        ));
        assert!(error.to_string().contains("resident.hef"));
        assert_eq!(after.vlm_create - before.vlm_create, 0);
        assert_eq!(
            HailoTaskError::HailoRt(error).status_code(),
            StatusCode::CONFLICT
        );
    }

    #[cfg(hailo_stub)]
    #[test]
    fn speech2text_is_never_cached() {
        let before = stub_counts();
        let mut ctx = DeviceCtx::default();
        drop(ctx.speech2text("speech.hef").unwrap());
        drop(ctx.speech2text("speech.hef").unwrap());
        let after = stub_counts();

        assert_eq!(after.s2t_create - before.s2t_create, 2);
        assert_eq!(after.s2t_release - before.s2t_release, 2);
    }

    #[cfg(hailo_stub)]
    #[test]
    fn reused_llm_is_cleared_before_second_generation() {
        let mut ctx = DeviceCtx::default();
        ctx.llm_for_generation("chat.hef", None, false)
            .unwrap()
            .generate_text("turn one", 1)
            .unwrap();
        let after_first = stub_counts();

        ctx.llm_for_generation("chat.hef", None, false)
            .unwrap()
            .generate_text("turn two", 1)
            .unwrap();
        let after_second = stub_counts();

        assert_eq!(
            after_second.llm_clear_context,
            after_first.llm_clear_context + 1,
            "clear_context must run before the second generation on a reused handle"
        );
    }

    #[cfg(hailo_stub)]
    #[test]
    fn conversation_b_does_not_inherit_conversation_a() {
        let mut ctx = DeviceCtx::default();
        let first = ctx
            .llm_for_generation("chat.hef", None, false)
            .unwrap()
            .generate_text("conversation A", 1)
            .unwrap();
        let second = ctx
            .llm_for_generation("chat.hef", None, false)
            .unwrap()
            .generate_text("conversation B", 1)
            .unwrap();

        assert_eq!(first, "conversation A");
        assert_eq!(second, "conversation B");
    }

    #[cfg(hailo_stub)]
    #[test]
    fn panic_discards_the_active_handle() {
        let before = stub_counts();
        let mut ctx = DeviceCtx::default();
        let result = ctx.run(|ctx| -> HailoRtResult<()> {
            ctx.llm("panic.hef", None, false)?;
            panic!("panic after handle acquisition");
        });
        assert!(matches!(result, Err(HailoTaskError::Panicked)));

        ctx.run(|ctx| ctx.llm("panic.hef", None, false).map(|_| ()))
            .unwrap();
        let after = stub_counts();
        assert_eq!(after.llm_release - before.llm_release, 1);
        assert_eq!(after.llm_create - before.llm_create, 2);
    }

    #[cfg(hailo_stub)]
    #[test]
    fn panic_discard_cap_degrades_without_dropping_another_handle() {
        let before = stub_counts();
        let mut ctx = DeviceCtx::default();
        for _ in 0..MAX_PANIC_HANDLE_DISCARDS {
            let result = ctx.run(|ctx| -> HailoRtResult<()> {
                ctx.llm("panic-cap.hef", None, false)?;
                panic!("discardable panic");
            });
            assert!(matches!(result, Err(HailoTaskError::Panicked)));
        }

        let capped = ctx.run(|ctx| -> HailoRtResult<()> {
            ctx.llm("panic-cap.hef", None, false)?;
            panic!("panic past discard cap");
        });
        let capped_error = capped.unwrap_err();
        assert!(matches!(&capped_error, HailoTaskError::Unavailable(_)));
        assert_eq!(capped_error.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        let after_cap = stub_counts();
        assert_eq!(
            after_cap.llm_release - before.llm_release,
            MAX_PANIC_HANDLE_DISCARDS
        );

        let next = ctx.run(|_| Ok(()));
        assert!(matches!(next, Err(HailoTaskError::Unavailable(_))));
        assert_eq!(
            stub_counts().llm_release,
            after_cap.llm_release,
            "degraded mode must leave recovery to an operator reboot"
        );
    }

    #[tokio::test]
    async fn device_work_is_serialized() {
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let first = run_hailort_task({
            let order = order.clone();
            move |_| {
                order.lock().unwrap().push("first-start");
                std::thread::sleep(std::time::Duration::from_millis(50));
                order.lock().unwrap().push("first-end");
                Ok(())
            }
        });
        let second = run_hailort_task({
            let order = order.clone();
            move |_| {
                order.lock().unwrap().push("second-start");
                order.lock().unwrap().push("second-end");
                Ok(())
            }
        });

        let (first_result, second_result) = tokio::join!(first, second);
        first_result.unwrap();
        second_result.unwrap();
        assert_eq!(
            *order.lock().unwrap(),
            ["first-start", "first-end", "second-start", "second-end"]
        );
    }

    #[test]
    fn decode_base64_image_rejects_oversized_dimensions() {
        // A solid-color PNG compresses tiny regardless of declared
        // dimensions — the classic decompression-bomb shape (small
        // encoded bytes, huge decoded pixel buffer).
        let huge = image::RgbImage::from_pixel(5000, 5000, image::Rgb([0, 0, 0]));
        let mut encoded = Vec::new();
        image::DynamicImage::ImageRgb8(huge)
            .write_to(
                &mut std::io::Cursor::new(&mut encoded),
                image::ImageFormat::Png,
            )
            .expect("encode test png");
        use base64::Engine as _;
        let base64_frame = base64::engine::general_purpose::STANDARD.encode(&encoded);

        let mut budget = MAX_TOTAL_DECODED_IMAGE_BYTES;
        let result = decode_base64_image(&base64_frame, &mut budget);
        assert!(
            result.is_err(),
            "5000x5000 frame should exceed decode limits"
        );
    }

    #[test]
    fn decode_base64_image_enforces_request_level_budget() {
        // A single frame small enough to pass the per-frame dimension cap,
        // but a budget too small for even one such frame — the second
        // frame in a multi-frame request must fail once the shared budget
        // (not a per-frame-only cap) is exhausted.
        let frame = image::RgbImage::from_pixel(1000, 1000, image::Rgb([1, 2, 3]));
        let mut encoded = Vec::new();
        image::DynamicImage::ImageRgb8(frame)
            .write_to(
                &mut std::io::Cursor::new(&mut encoded),
                image::ImageFormat::Png,
            )
            .expect("encode test png");
        use base64::Engine as _;
        let base64_frame = base64::engine::general_purpose::STANDARD.encode(&encoded);

        // 1000*1000*3 bytes decoded == 3_000_000; budget for one frame only.
        let mut budget = 3_000_000u64;
        let first = decode_base64_image(&base64_frame, &mut budget);
        assert!(first.is_ok(), "first frame should fit the initial budget");
        assert_eq!(budget, 0, "budget should be fully spent after one frame");

        let second = decode_base64_image(&base64_frame, &mut budget);
        assert!(
            second.is_err(),
            "second identical frame should fail once the shared budget is exhausted"
        );
    }

    #[test]
    fn decode_base64_image_accepts_normal_frame() {
        let small = image::RgbImage::from_pixel(64, 64, image::Rgb([128, 64, 32]));
        let mut encoded = Vec::new();
        image::DynamicImage::ImageRgb8(small)
            .write_to(
                &mut std::io::Cursor::new(&mut encoded),
                image::ImageFormat::Png,
            )
            .expect("encode test png");
        use base64::Engine as _;
        let base64_frame = base64::engine::general_purpose::STANDARD.encode(&encoded);

        let mut budget = MAX_TOTAL_DECODED_IMAGE_BYTES;
        let decoded = decode_base64_image(&base64_frame, &mut budget).expect("decode small frame");
        assert_eq!(decoded.width(), 64);
        assert_eq!(decoded.height(), 64);
    }

    #[tokio::test]
    async fn infer_wd_rejects_missing_auth_header() {
        let app = build_router(test_state(vec![]));
        let body = json!({"path": "/tmp", "model_id": "wd_vit_v3"}).to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/wd")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn infer_wd_rejects_empty_scan_roots() {
        let dir = std::env::temp_dir().join("yu-infer-empty-scan-roots-test");
        std::fs::create_dir_all(&dir).unwrap();
        let app = build_router(test_state(vec![]));
        let body = json!({"path": dir.to_str().unwrap(), "model_id": "wd_vit_v3"}).to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/wd")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn infer_wd_rejects_path_outside_scan_roots() {
        let allowed_root = std::env::temp_dir().join("yu-infer-allowed-root");
        std::fs::create_dir_all(&allowed_root).unwrap();
        let outside_dir = std::env::temp_dir().join("yu-infer-outside-root");
        std::fs::create_dir_all(&outside_dir).unwrap();

        let app = build_router(test_state(vec![allowed_root]));
        let body =
            json!({"path": outside_dir.to_str().unwrap(), "model_id": "wd_vit_v3"}).to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/wd")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn infer_wd_rejects_invalid_model_id() {
        let dir = std::env::temp_dir().join("yu-infer-model-id-test");
        std::fs::create_dir_all(&dir).unwrap();
        let app = build_router(test_state(vec![dir.clone()]));
        let body = json!({"path": dir.to_str().unwrap(), "model_id": "../etc/passwd"}).to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/wd")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn healthz_returns_ok_true() {
        let app = build_router(test_state(vec![]));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["instance_id"], "test-instance");
    }

    #[tokio::test]
    async fn vlm_generate_stream_rejects_missing_auth() {
        let app = build_router(test_state(vec![]));
        let body = json!({"prompt": "hi", "frames": []}).to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/vlm/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn vlm_generate_stream_rejects_missing_frames() {
        let app = build_router(test_state(vec![]));
        let body = json!({"prompt": "hi", "frames": []}).to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/vlm/generate/stream")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn vlm_generate_stream_rejects_oversized_max_generated_tokens() {
        let app = build_router(test_state(vec![]));
        let body = json!({
            "prompt": "hi",
            "frames": ["AAAA"],
            "max_generated_tokens": MAX_VLM_GENERATED_TOKENS + 1,
        })
        .to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/vlm/generate/stream")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn llm_generate_stream_rejects_missing_auth() {
        let app = build_router(test_state(vec![]));
        let body = json!({"messages": [{"role": "user", "content": "hi"}]}).to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/llm/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn llm_generate_stream_rejects_missing_messages() {
        let app = build_router(test_state(vec![]));
        let body = json!({"messages": []}).to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/llm/generate/stream")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn llm_generate_stream_rejects_oversized_prompt() {
        let app = build_router(test_state(vec![]));
        let body = json!({
            "messages": [{"role": "user", "content": "a".repeat(MAX_PROMPT_BYTES + 1)}],
        })
        .to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/llm/generate/stream")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn llm_generate_stream_rejects_oversized_max_generated_tokens() {
        let app = build_router(test_state(vec![]));
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_generated_tokens": MAX_LLM_GENERATED_TOKENS + 1,
        })
        .to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/llm/generate/stream")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn llm_generate_stream_rejects_too_many_tools() {
        let app = build_router(test_state(vec![]));
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": (0..MAX_LLM_TOOLS + 1)
                .map(|i| json!({"name": format!("tool_{i}")}))
                .collect::<Vec<_>>(),
        })
        .to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/llm/generate/stream")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn llm_generate_stream_rejects_oversized_tools() {
        let app = build_router(test_state(vec![]));
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "big", "description": "a".repeat(MAX_PROMPT_BYTES + 1)}],
        })
        .to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/llm/generate/stream")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn clip_image_rejects_missing_auth() {
        let app = build_router(test_state(vec![]));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/clip-image")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({"image_base64": "not-used"}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn clip_image_rejects_invalid_base64() {
        let app = build_router(test_state(vec![]));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/clip-image")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(json!({"image_base64": "%%%"}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn clip_image_rejects_oversized_base64() {
        let app = build_router(test_state(vec![]));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/clip-image")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(
                        json!({"image_base64": "A".repeat(MAX_FRAME_BASE64_BYTES + 1)}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn clip_image_body_between_axum_default_and_frame_cap_reaches_handler() {
        // Regression test for the route-scoped DefaultBodyLimit fix: axum's
        // own default body limit (2 MiB) is below MAX_FRAME_BASE64_BYTES (8
        // MiB), so without a route-level override a body in that gap would
        // be rejected by axum itself with a generic 413 before ever reaching
        // this handler's own validation. Sending 4 MiB of garbage base64
        // must reach decode_base64_image and fail there (400
        // clip_image_invalid_image), not get stopped by axum's body limit.
        let app = build_router(test_state(vec![]));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/clip-image")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(
                        json!({"image_base64": "A".repeat(4 * 1024 * 1024)}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn clip_text_rejects_missing_auth() {
        let app = build_router(test_state(vec![]));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/clip-text")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({"text": "cat"}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn clip_text_rejects_oversized_text() {
        let app = build_router(test_state(vec![]));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/clip-text")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(
                        json!({"text": "a".repeat(MAX_TEXT_BYTES + 1)}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }
}
