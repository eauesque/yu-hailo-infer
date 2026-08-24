use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::Response,
    routing::{post, MethodRouter},
    Json,
};
use serde::Deserialize;
use serde_json::json;

use crate::{
    hailort::Speech2TextTask,
    router::{
        api_error, api_ok, auth_error, hailort_api_error, run_hailort_task,
        run_media_preprocessing, s2t_hef_path, AppState, MediaPreprocessError, MAX_TIMEOUT_MS,
    },
};

const MAX_AUDIO_BASE64_BYTES: usize = 32 * 1024 * 1024;
/// Maximum decoded audio length: 16 kHz × 600 seconds = 10 minutes.
const MAX_AUDIO_SAMPLES: usize = 16_000 * 600;
/// Includes the decoded source, downmix, resample, and final audio buffers.
const MAX_AUDIO_PREPROCESSING_RESERVATION_BYTES: u64 = 160 * 1024 * 1024;

#[derive(Debug, Deserialize)]
pub struct TranscribeRequest {
    audio_base64: String,
    task: Option<String>,
    language: Option<String>,
    repetition_penalty: Option<f32>,
    timeout_ms: Option<u32>,
    hef_path: Option<String>,
}

pub(crate) fn speech2text_transcribe_route() -> MethodRouter<AppState> {
    post(speech2text_transcribe).layer(axum::extract::DefaultBodyLimit::max(
        MAX_AUDIO_BASE64_BYTES + 4096,
    ))
}

// Unlike image decoding's single String error, these variants retain whether
// audio exceeded a client-remediable limit or has an invalid format, so the
// handler can return the appropriate status code and error message.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum WavDecodeError {
    TooLarge,
    InvalidBase64,
    UnsupportedFormat,
    TooLong,
    Empty,
}

/// Decodes a base64-encoded PCM WAV into the 16 kHz mono samples expected by
/// the speech-to-text model. Incomplete final interleaved frames are dropped.
pub(crate) fn decode_base64_wav(audio_base64: &str) -> Result<Vec<f32>, WavDecodeError> {
    if audio_base64.len() > MAX_AUDIO_BASE64_BYTES {
        return Err(WavDecodeError::TooLarge);
    }

    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(audio_base64)
        .map_err(|_| WavDecodeError::InvalidBase64)?;
    let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes))
        .map_err(|_| WavDecodeError::UnsupportedFormat)?;
    let spec = reader.spec();
    if spec.channels == 0 || spec.sample_rate == 0 {
        return Err(WavDecodeError::UnsupportedFormat);
    }
    if !matches!(
        (spec.sample_format, spec.bits_per_sample),
        (hound::SampleFormat::Int, 16) | (hound::SampleFormat::Float, 32)
    ) {
        return Err(WavDecodeError::UnsupportedFormat);
    }

    let duration_frames = u64::from(reader.duration());
    let projected_len = duration_frames
        .checked_mul(16_000)
        .and_then(|value| value.checked_add(u64::from(spec.sample_rate) - 1))
        .and_then(|value| value.checked_div(u64::from(spec.sample_rate)))
        .ok_or(WavDecodeError::TooLong)?;
    if projected_len > MAX_AUDIO_SAMPLES as u64 {
        return Err(WavDecodeError::TooLong);
    }

    let interleaved = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|sample| {
                sample
                    .map(|sample| sample as f32 / 32_768.0)
                    .map_err(|_| WavDecodeError::UnsupportedFormat)
            })
            .collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|sample| {
                sample
                    .map(|sample| if sample.is_finite() { sample } else { 0.0 })
                    .map_err(|_| WavDecodeError::UnsupportedFormat)
            })
            .collect::<Result<Vec<_>, _>>()?,
    };

    let channels = usize::from(spec.channels);
    let mono: Vec<f32> = interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    if mono.is_empty() {
        return Err(WavDecodeError::Empty);
    }

    let samples = resample_linear(&mono, spec.sample_rate, 16_000);
    if samples.len() > MAX_AUDIO_SAMPLES {
        return Err(WavDecodeError::TooLong);
    }
    Ok(samples)
}

fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    debug_assert!(from_rate != 0);
    debug_assert!(to_rate != 0);
    if samples.is_empty() {
        return Vec::new();
    }
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let output_len = (samples.len() * to_rate as usize).div_ceil(from_rate as usize);
    (0..output_len)
        .map(|index| {
            let position = index as f64 * from_rate as f64 / to_rate as f64;
            let left = position.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let fraction = (position - left as f64) as f32;
            samples[left] + (samples[right] - samples[left]) * fraction
        })
        .collect()
}

async fn speech2text_transcribe(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<TranscribeRequest>,
) -> Response {
    if let Some(response) = auth_error(&state, &headers) {
        return response;
    }
    if body.audio_base64.is_empty() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "audio_required",
            "audio_base64 is required".to_string(),
        );
    }

    let audio_base64 = body.audio_base64;
    let audio =
        match run_media_preprocessing(MAX_AUDIO_PREPROCESSING_RESERVATION_BYTES, move || {
            decode_base64_wav(&audio_base64)
        })
        .await
        {
            Ok(audio) => audio,
            Err(MediaPreprocessError::Busy) => {
                return api_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "media_preprocessing_busy",
                    "media preprocessing capacity is temporarily exhausted".to_string(),
                )
            }
            Err(MediaPreprocessError::Task(WavDecodeError::TooLarge)) => {
                return api_error(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "audio_too_large",
                    format!("audio_base64 exceeds {MAX_AUDIO_BASE64_BYTES} bytes"),
                );
            }
            Err(MediaPreprocessError::Task(WavDecodeError::InvalidBase64)) => {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_audio_base64",
                    "audio_base64 is not valid base64".to_string(),
                );
            }
            Err(MediaPreprocessError::Task(WavDecodeError::UnsupportedFormat)) => {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    "unsupported_wav_format",
                    "audio_base64 must contain a 16-bit PCM or 32-bit float WAV".to_string(),
                );
            }
            Err(MediaPreprocessError::Task(WavDecodeError::TooLong)) => {
                return api_error(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "audio_too_long",
                    format!("decoded audio exceeds {MAX_AUDIO_SAMPLES} samples"),
                );
            }
            Err(MediaPreprocessError::Task(WavDecodeError::Empty)) => {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    "audio_empty",
                    "decoded audio contains no samples".to_string(),
                );
            }
            Err(MediaPreprocessError::Join(error)) => {
                return api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "media_preprocessing_failed",
                    error,
                )
            }
        };

    let task_name = body.task.as_deref().unwrap_or("transcribe");
    let task = match task_name {
        "transcribe" => Speech2TextTask::Transcribe,
        "translate" => Speech2TextTask::Translate,
        _ => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_task",
                "task must be transcribe or translate".to_string(),
            );
        }
    };
    let repetition_penalty = body.repetition_penalty.unwrap_or(1.0);
    if !(repetition_penalty.is_finite() && 0.0 < repetition_penalty && repetition_penalty <= 4.0) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_repetition_penalty",
            "repetition_penalty must be finite and in the range (0.0, 4.0]".to_string(),
        );
    }

    let timeout_ms = body.timeout_ms.unwrap_or(30_000).min(MAX_TIMEOUT_MS);
    let hef_path = s2t_hef_path(body.hef_path.as_deref());
    let hef_path_str = hef_path.to_string_lossy().to_string();
    let language = body.language;
    let task_name = task_name.to_string();
    let result = run_hailort_task({
        let hef_path_str = hef_path_str.clone();
        move |ctx| {
            ctx.speech2text(&hef_path_str).and_then(|mut s2t| {
                s2t.generate_segments(
                    &audio,
                    task,
                    language.as_deref(),
                    repetition_penalty,
                    timeout_ms,
                )
            })
        }
    })
    .await;
    match result {
        Ok(segments) => {
            // The flat `text` field is kept for callers that only want the
            // transcript (and for backward compat with the pre-segment
            // response shape) -- it is exactly what generate_all_text()
            // would have produced, joined the same way Whisper's own
            // segment-to-text concatenation does.
            let text = segments
                .iter()
                .map(|segment| segment.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let segments_json: Vec<_> = segments
                .into_iter()
                .map(|segment| {
                    json!({
                        "text": segment.text,
                        "start": segment.start_sec,
                        "end": segment.end_sec,
                    })
                })
                .collect();
            api_ok(json!({
                "hef_path": hef_path_str,
                "text": text,
                "segments": segments_json,
                "task": task_name,
            }))
        }
        Err(error) => hailort_api_error(error, "hailort_s2t_generate_failed"),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        path::PathBuf,
        sync::{Arc, RwLock},
    };

    use axum::{body::Body, http::Request};
    use serde_json::json;
    use tower::ServiceExt;

    use super::*;
    use crate::router::build_router;

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

    fn encode_wav(spec: hound::WavSpec, samples: &[f32]) -> String {
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).expect("create test WAV");
            for &sample in samples {
                if spec.sample_format == hound::SampleFormat::Int {
                    writer
                        .write_sample((sample * 32_767.0) as i16)
                        .expect("write test PCM sample");
                } else {
                    writer
                        .write_sample(sample)
                        .expect("write test float sample");
                }
            }
            writer.finalize().expect("finalize test WAV");
        }
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD.encode(cursor.into_inner())
    }

    fn pcm16_spec(channels: u16, sample_rate: u32) -> hound::WavSpec {
        hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        }
    }

    #[test]
    fn decode_base64_wav_mono_passthrough() {
        let decoded = decode_base64_wav(&encode_wav(pcm16_spec(1, 16_000), &[0.0, 0.5, -0.5]))
            .expect("decode mono WAV");
        assert_eq!(decoded.len(), 3);
        assert!((decoded[1] - 0.5).abs() < 0.0001);
        assert!((decoded[2] + 0.5).abs() < 0.0001);
    }

    #[test]
    fn decode_base64_wav_averages_stereo() {
        let decoded = decode_base64_wav(&encode_wav(pcm16_spec(2, 16_000), &[1.0, -1.0, 0.5, 0.0]))
            .expect("decode stereo WAV");
        assert_eq!(decoded.len(), 2);
        assert!(decoded[0].abs() < 0.0001);
        assert!((decoded[1] - 0.25).abs() < 0.0001);
    }

    #[test]
    fn decode_base64_wav_upsamples_linearly() {
        let decoded = decode_base64_wav(&encode_wav(pcm16_spec(1, 8_000), &[0.0, 1.0]))
            .expect("upsample WAV");
        assert_eq!(decoded.len(), 4);
        assert!((decoded[1] - 0.5).abs() < 0.0001);
    }

    #[test]
    fn decode_base64_wav_downsamples_linearly() {
        let decoded =
            decode_base64_wav(&encode_wav(pcm16_spec(1, 32_000), &[0.0, 0.25, 0.5, 0.75]))
                .expect("downsample WAV");
        assert_eq!(decoded.len(), 2);
        assert!((decoded[1] - 0.5).abs() < 0.0001);
    }

    #[test]
    fn decode_base64_wav_rejects_invalid_base64() {
        assert_eq!(
            decode_base64_wav("%%%").unwrap_err(),
            WavDecodeError::InvalidBase64
        );
    }

    #[test]
    fn decode_base64_wav_rejects_unsupported_sample_format() {
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(
                &mut cursor,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 16_000,
                    bits_per_sample: 8,
                    sample_format: hound::SampleFormat::Int,
                },
            )
            .expect("create 8-bit WAV");
            writer.write_sample(0i8).expect("write 8-bit sample");
            writer.finalize().expect("finalize 8-bit WAV");
        }
        use base64::Engine as _;
        let encoded = base64::engine::general_purpose::STANDARD.encode(cursor.into_inner());
        assert_eq!(
            decode_base64_wav(&encoded).unwrap_err(),
            WavDecodeError::UnsupportedFormat
        );
    }

    #[test]
    fn decode_base64_wav_rejects_zero_channels_before_division() {
        use base64::Engine as _;
        let mut bytes = base64::engine::general_purpose::STANDARD
            .decode(encode_wav(pcm16_spec(1, 16_000), &[0.0]))
            .unwrap();
        bytes[22..24].copy_from_slice(&0u16.to_le_bytes());
        assert_eq!(
            decode_base64_wav(&base64::engine::general_purpose::STANDARD.encode(bytes))
                .unwrap_err(),
            WavDecodeError::UnsupportedFormat
        );
    }

    #[test]
    fn decode_base64_wav_rejects_zero_sample_rate_before_division() {
        use base64::Engine as _;
        let mut bytes = base64::engine::general_purpose::STANDARD
            .decode(encode_wav(pcm16_spec(1, 16_000), &[0.0]))
            .unwrap();
        bytes[24..28].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            decode_base64_wav(&base64::engine::general_purpose::STANDARD.encode(bytes))
                .unwrap_err(),
            WavDecodeError::UnsupportedFormat
        );
    }

    #[test]
    fn decode_base64_wav_rejects_oversized_base64_before_decode() {
        assert_eq!(
            decode_base64_wav(&"%".repeat(MAX_AUDIO_BASE64_BYTES + 1)).unwrap_err(),
            WavDecodeError::TooLarge
        );
    }

    #[test]
    fn decode_base64_wav_rejects_projected_length_before_resampling() {
        let encoded = encode_wav(pcm16_spec(1, 1), &vec![0.0; 601]);
        assert_eq!(
            decode_base64_wav(&encoded).unwrap_err(),
            WavDecodeError::TooLong
        );
    }

    #[test]
    fn decode_base64_wav_sanitizes_non_finite_float_samples() {
        let decoded = decode_base64_wav(&encode_wav(
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
            &[f32::NAN, f32::INFINITY, -f32::INFINITY, 0.25],
        ))
        .expect("decode float WAV");
        assert_eq!(decoded, vec![0.0, 0.0, 0.0, 0.25]);
    }

    #[test]
    fn decode_base64_wav_rejects_empty_pcm() {
        let encoded = encode_wav(pcm16_spec(1, 16_000), &[]);
        assert_eq!(
            decode_base64_wav(&encoded).unwrap_err(),
            WavDecodeError::Empty
        );
    }

    #[tokio::test]
    async fn speech2text_transcribe_rejects_missing_auth() {
        let app = build_router(test_state(vec![]));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/speech2text/transcribe")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({"audio_base64": "not-used"}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn speech2text_transcribe_rejects_empty_audio() {
        let app = build_router(test_state(vec![]));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/speech2text/transcribe")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(json!({"audio_base64": ""}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn speech2text_transcribe_rejects_invalid_task() {
        let app = build_router(test_state(vec![]));
        let audio_base64 = encode_wav(pcm16_spec(1, 16_000), &[0.0]);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/speech2text/transcribe")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(
                        json!({"audio_base64": audio_base64, "task": "summarize"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn speech2text_transcribe_rejects_invalid_repetition_penalty() {
        let app = build_router(test_state(vec![]));
        let audio_base64 = encode_wav(pcm16_spec(1, 16_000), &[0.0]);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/speech2text/transcribe")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(
                        json!({"audio_base64": audio_base64, "repetition_penalty": 4.1})
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    #[ignore = "requires a Hailo device (/dev/hailo0 or /dev/h1x-0, depending on driver generation) and HAILO_S2T_HEF"]
    async fn smoke_speech2text_transcribe() {
        let hef_path = std::env::var("HAILO_S2T_HEF").expect("HAILO_S2T_HEF must be set");
        let audio_base64 = encode_wav(pcm16_spec(1, 16_000), &vec![0.0; 16_000]);
        let app = build_router(test_state(vec![]));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/infer/speech2text/transcribe")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(
                        json!({
                            "audio_base64": audio_base64,
                            "hef_path": hef_path,
                            "task": "transcribe",
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Lock in the response contract added for segment support: `segments`
        // must be present and array-shaped even for silence (which may
        // legitimately transcribe to zero segments) -- this is what would
        // regress silently if generate_all_segments()'s JSON marshaling
        // broke, since the flat `text` field alone wouldn't catch it.
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read response body");
        let parsed: serde_json::Value =
            serde_json::from_slice(&body).expect("response body is valid JSON");
        let data = &parsed["data"];
        assert!(
            data["segments"].is_array(),
            "expected data.segments to be an array, got: {parsed}"
        );
        assert!(
            data["text"].is_string(),
            "expected data.text to be a string, got: {parsed}"
        );
    }
}
