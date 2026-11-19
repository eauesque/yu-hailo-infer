use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::router::MAX_TIMEOUT_MS;

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WdBackend {
    #[default]
    Onnx,
    Hef,
}

#[derive(Deserialize)]
pub struct InferWdRequest {
    pub path: String,
    pub model_id: String,
    #[serde(default = "default_general_thr")]
    pub general_thr: f32,
    #[serde(default = "default_character_thr")]
    pub character_thr: f32,
    /// Full inference recipe, resolved by yu-server from its profile
    /// registry. Absent means the built-in WD recipe, which is what every
    /// caller predating this field expects.
    ///
    /// `default_thresholds` inside it is the *resolved* threshold table: the
    /// caller has already decided whether the user's configured thresholds or
    /// the profile's own defaults win, because that decision differs per
    /// adapter family and belongs with the registry, not here.
    #[serde(default)]
    pub profile: Option<infer_core::WdProfileSpec>,
    /// Variant subdirectory under the model's cache directory, for repos that
    /// ship several builds (the Python side calls this `hf_subdir`). Validated
    /// as a plain relative path before being joined.
    #[serde(default)]
    pub model_subdir: Option<String>,
    pub backend: Option<WdBackend>,
    pub hef_path: Option<String>,
    pub hef_sha256: Option<String>,
    pub timeout_ms: Option<u32>,
}

impl InferWdRequest {
    pub(crate) fn hef_options(&self) -> Result<Option<(&str, u32)>, Box<Response>> {
        if self.backend.unwrap_or_default() == WdBackend::Onnx {
            return Ok(None);
        }
        let path = self
            .hef_path
            .as_deref()
            .ok_or_else(|| Box::new(hef_error(StatusCode::SERVICE_UNAVAILABLE, "hef_invalid")))?;
        self.hef_digest()
            .ok_or_else(|| Box::new(hef_error(StatusCode::SERVICE_UNAVAILABLE, "hef_invalid")))?;
        Ok(Some((path, hef_timeout_ms(self.timeout_ms))))
    }

    pub(crate) fn hef_digest(&self) -> Option<[u8; 32]> {
        let hash = self.hef_sha256.as_deref()?;
        if hash.len() != 64
            || !hash
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return None;
        }
        Some(std::array::from_fn(|i| {
            u8::from_str_radix(&hash[i * 2..i * 2 + 2], 16).unwrap()
        }))
    }
}

#[derive(Serialize)]
pub(crate) struct InferWdResponse {
    pub backend_applied: WdBackend,
    pub data: infer_core::TagResult,
    pub ok: bool,
    pub profile_applied: bool,
}

pub(crate) fn hef_error(status: StatusCode, code: &'static str) -> Response {
    (status, Json(json!({"error": code}))).into_response()
}

pub(crate) fn hef_timeout_ms(timeout_ms: Option<u32>) -> u32 {
    timeout_ms.unwrap_or(120_000).min(MAX_TIMEOUT_MS)
}

fn default_general_thr() -> f32 {
    0.35
}

fn default_character_thr() -> f32 {
    0.85
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wd_wire_hef_timeout_defaults_accepts_120000_and_clamps() {
        for (requested, expected) in [
            (None, 120_000),
            (Some(120_000), 120_000),
            (Some(120_001), MAX_TIMEOUT_MS),
            (Some(u32::MAX), MAX_TIMEOUT_MS),
            (Some(1), 1),
        ] {
            assert_eq!(hef_timeout_ms(requested), expected);
        }
    }

    #[test]
    fn wd_wire_legacy_request_deserializes_to_defaults() {
        let request: InferWdRequest =
            serde_json::from_value(json!({"path": "image.png", "model_id": "model"})).unwrap();
        assert_eq!(request.backend, None);
        assert_eq!(request.backend.unwrap_or_default(), WdBackend::Onnx);
        assert!(request.hef_path.is_none());
        assert_eq!(request.timeout_ms, None);
        assert_eq!(request.general_thr, 0.35);
        assert_eq!(request.character_thr, 0.85);
        assert!(request.profile.is_none());
        assert!(request.model_subdir.is_none());
        assert!(request.hef_options().unwrap().is_none());
    }

    #[test]
    fn wd_wire_hef_request_parses_and_clamps_timeout() {
        for timeout in [None, Some(120_000), Some(u32::MAX)] {
            let request: InferWdRequest = serde_json::from_value(json!({
                "path": "image.png", "model_id": "model", "backend": "hef",
                "hef_path": "model.hef", "timeout_ms": timeout, "hef_sha256": "0".repeat(64)
            }))
            .unwrap();
            assert_eq!(request.hef_options().unwrap(), Some(("model.hef", 120_000)));
        }
    }
}
