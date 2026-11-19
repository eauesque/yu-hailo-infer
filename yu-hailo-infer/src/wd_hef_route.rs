//! HEF-only WD execution and closed, path-free wire errors.
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use infer_core::{profile::TagSourceSpec, tags::load_tags, WdProfileSpec};

use crate::{
    hailort::{wd::WdError, wd_hef::WdHefError, wd_validate, HailoRtError},
    router::{
        check_model_id, check_scan_roots, run_hailort_task, run_media_preprocessing,
        scan_roots_forbidden, AppState, HailoTaskError, MediaPreprocessError,
        MAX_IMAGE_PREPROCESSING_RESERVATION_BYTES,
    },
    wd_wire::{hef_error, hef_timeout_ms, InferWdRequest, InferWdResponse, WdBackend},
};

pub(crate) async fn infer(state: AppState, body: InferWdRequest) -> Response {
    let deadline =
        Instant::now() + Duration::from_millis(u64::from(hef_timeout_ms(body.timeout_ms)));
    let result = tokio::time::timeout_at(deadline.into(), execute(state, body, deadline))
        .await
        .unwrap_or(Err("hef_timeout"));
    match result {
        Ok(response) => Json(serde_json::json!(response)).into_response(),
        Err("path_outside_roots") => scan_roots_forbidden(),
        Err(code) => {
            tracing::warn!(code);
            let status = if code == "image_invalid" {
                StatusCode::UNPROCESSABLE_ENTITY
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            };
            hef_error(status, code)
        }
    }
}

fn profile_matches(spec: &WdProfileSpec) -> bool {
    let p = &spec.preprocess_spec;
    spec.validate().is_ok()
        && p.input_size == 448
        && p.resize_strategy == "longest_side_pad"
        && p.pad_color == [255, 255, 255]
        && p.channel_order == "BGR"
        && p.layout == "NHWC"
        && p.scale == 1.0
        && p.mean.is_none()
        && p.std.is_none()
        && matches!(&spec.tag_source, TagSourceSpec::Csv { file, .. } if file == "selected_tags.csv")
}

fn model_error(error: WdHefError) -> &'static str {
    match error {
        WdHefError::NotFound => "hef_not_found",
        WdHefError::Invalid
        | WdHefError::Runtime(WdError::InvalidHef)
        | WdHefError::Runtime(WdError::HailoRt(HailoRtError::InvalidMetadata(_))) => "hef_invalid",
        WdHefError::Runtime(WdError::Timeout) => "hef_timeout",
        WdHefError::Runtime(_) => "device_error",
    }
}

fn worker_error(error: HailoTaskError) -> &'static str {
    match error {
        HailoTaskError::QueueFull => "device_busy",
        _ => "device_error",
    }
}

pub(crate) async fn execute(
    state: AppState,
    body: InferWdRequest,
    deadline: Instant,
) -> Result<InferWdResponse, &'static str> {
    if crate::hailort::session::is_lost() {
        return Err("device_error");
    }
    let digest = body.hef_digest().ok_or("hef_invalid")?;
    let profile_applied = body.profile.is_some();
    let builtin =
        infer_core::builtin_wd_profile("model.onnx", body.general_thr, body.character_thr);
    let spec = body.profile.unwrap_or_else(|| builtin.clone());
    if !profile_matches(&spec) || !check_model_id(&body.model_id) {
        return Err("hef_invalid");
    }
    let model_dir = infer_core::resolve_model_dir(
        &state.wd_cache_dir,
        &body.model_id,
        body.model_subdir.as_deref(),
    )
    .ok_or("hef_invalid")?;
    let hef_path = PathBuf::from(body.hef_path.ok_or("hef_invalid")?);
    let recipe = spec.clone();
    let prepared = run_media_preprocessing(MAX_IMAGE_PREPROCESSING_RESERVATION_BYTES, move || {
        // The ONNX fallback files remain required even when a profile names different weights.
        if !infer_core::is_profile_model_ready(&model_dir, &recipe)
            || !infer_core::is_profile_model_ready(&model_dir, &builtin)
        {
            return Err("model_not_downloaded");
        }
        let tag_meta = load_tags(
            &model_dir,
            &recipe.tag_source,
            &recipe.categories_mode,
            &recipe.supports_categories,
        )
        .map_err(|_| "hef_invalid")?;
        let real_path = std::fs::canonicalize(&body.path).map_err(|_| "image_invalid")?;
        if !check_scan_roots(&state, &real_path) {
            return Err("path_outside_roots");
        }
        let pixels =
            infer_core::wd_recipe::preprocess_u8(&real_path).map_err(|_| "image_invalid")?;
        Ok((tag_meta, real_path, pixels))
    })
    .await;
    let (tag_meta, image_path, pixels) = match prepared {
        Ok(prepared) => prepared,
        Err(MediaPreprocessError::Busy) => return Err("device_busy"),
        Err(MediaPreprocessError::Task(code)) => return Err(code),
        Err(MediaPreprocessError::Join(_)) => return Err("device_error"),
    };

    let result = run_hailort_task(move |ctx| {
        if Instant::now() >= deadline {
            return Ok(Err(WdError::Timeout.into()));
        }
        let validate = |inputs: &[_], outputs: &[_]| {
            wd_validate::metadata(inputs, outputs, tag_meta.tags.len())?;
            if spec
                .output_spec
                .output_key
                .as_ref()
                .is_some_and(|key| key != &outputs[0].name)
            {
                return Err(WdHefError::Invalid);
            }
            Ok(())
        };
        Ok(
            ctx.with_wd_hef_metadata(&hef_path, digest, validate, |model, inputs, outputs| {
                // The vocabulary/profile may change while the HEF file and resident model stay the same.
                validate(inputs, outputs)?;
                let remaining = deadline
                    .saturating_duration_since(Instant::now())
                    .as_millis() as u32;
                if remaining == 0 {
                    return Err(WdError::Timeout.into());
                }
                let output = &outputs[0];
                let mut bytes = vec![0u8; output.frame_size];
                model.run(&pixels, &mut [(&output.name, &mut bytes)], remaining)?;
                let scores: Vec<f32> = bytes
                    .into_iter()
                    .map(|q| {
                        let score = (f32::from(q) - output.quant.zero_point) * output.quant.scale;
                        if spec.output_spec.wants_sigmoid() {
                            (1.0_f64 / (1.0 + (-(score as f64)).exp())) as f32
                        } else {
                            score
                        }
                    })
                    .collect();
                Ok(infer_core::wd_recipe::build_result(
                    &image_path,
                    &scores,
                    &tag_meta,
                    &body.model_id,
                    &spec,
                ))
            }),
        )
    })
    .await
    .map_err(worker_error)?
    .map_err(model_error)?;
    Ok(InferWdResponse {
        backend_applied: WdBackend::Hef,
        data: result,
        ok: true,
        profile_applied,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revfix_busy_is_not_inferred_from_message() {
        assert_eq!(worker_error(HailoTaskError::QueueFull), "device_busy");
        assert_eq!(
            worker_error(HailoTaskError::Unavailable("HailoRT device queue is full")),
            "device_error"
        );
    }

    #[test]
    fn stopped_and_panicked_workers_are_device_errors() {
        assert_eq!(
            worker_error(HailoTaskError::Unavailable("HailoRT device thread stopped")),
            "device_error"
        );
        assert_eq!(worker_error(HailoTaskError::Panicked), "device_error");
    }
}
