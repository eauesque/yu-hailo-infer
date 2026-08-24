//! Profile-driven inference specification.
//!
//! yu-server owns the tagger profile registry and its parser. Rather than
//! duplicate that parser here — a second reading of one rule in a second
//! language, in a second repository — the server sends the inference-relevant
//! fields on the wire and this module deserializes them verbatim, under their
//! original names.
//!
//! Field names and defaults mirror
//! `extensions/builtin_wd_tagger/core_impl/profiles/*.json`. Where a default
//! is spelled out below it is the value the Python side falls back to, so a
//! request that omits the field behaves identically on both sides.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::InferError;

/// How to turn an image file into the model's input tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessSpec {
    pub input_size: u32,
    #[serde(default = "default_resize_strategy")]
    pub resize_strategy: String,
    #[serde(default = "default_pad_color")]
    pub pad_color: [u8; 3],
    #[serde(default = "default_channel_order")]
    pub channel_order: String,
    #[serde(default = "default_scale")]
    pub scale: f32,
    #[serde(default)]
    pub mean: Option<[f32; 3]>,
    #[serde(default)]
    pub std: Option<[f32; 3]>,
    #[serde(default = "default_layout")]
    pub layout: String,
}

fn default_resize_strategy() -> String {
    "longest_side_pad".to_string()
}
fn default_pad_color() -> [u8; 3] {
    [255, 255, 255]
}
fn default_channel_order() -> String {
    "RGB".to_string()
}
fn default_scale() -> f32 {
    1.0
}
fn default_layout() -> String {
    "NHWC".to_string()
}

/// Where the ordered `(tag, category)` list comes from.
///
/// Only the two shapes the shipped profiles use are accepted. `json_list` and
/// `composite` exist in the Python schema but have no shipped consumer; an
/// unknown variant is rejected rather than approximated, so the caller falls
/// back to the Python implementation instead of receiving wrong tags.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TagSourceSpec {
    Csv {
        file: String,
        #[serde(default = "default_delimiter")]
        delimiter: String,
        #[serde(default = "default_name_col")]
        name_col: String,
        #[serde(default = "default_category_col")]
        category_col: String,
        #[serde(default)]
        category_map: BTreeMap<String, String>,
    },
    JsonDict {
        file: String,
        #[serde(default)]
        container_key: Option<String>,
        idx_to_tag_key: String,
        tag_to_category_key: String,
    },
}

fn default_delimiter() -> String {
    ",".to_string()
}
fn default_name_col() -> String {
    "name".to_string()
}
fn default_category_col() -> String {
    "category".to_string()
}

/// Which output head to read, and whether it still needs an activation.
///
/// Defaults reproduce the historical behaviour: first output, verbatim. Every
/// v1 WD profile relies on it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutputSpec {
    #[serde(default)]
    pub output_key: Option<String>,
    #[serde(default)]
    pub activation: Option<String>,
}

impl OutputSpec {
    pub fn wants_sigmoid(&self) -> bool {
        self.activation.as_deref() == Some("sigmoid")
    }
}

/// The full inference contract for one model, as sent by yu-server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WdProfileSpec {
    /// Basename of the ONNX file inside the model directory.
    pub model_file: String,
    pub preprocess_spec: PreprocessSpec,
    pub tag_source: TagSourceSpec,
    #[serde(default)]
    pub output_spec: OutputSpec,
    #[serde(default)]
    pub default_thresholds: BTreeMap<String, f32>,
    #[serde(default = "default_categories_mode")]
    pub categories_mode: String,
    #[serde(default)]
    pub supports_categories: Vec<String>,
}

fn default_categories_mode() -> String {
    "from_tag_source".to_string()
}

/// Upper bound on a single metadata file, matching the Python reader's
/// `_METADATA_MAX_BYTES`.
pub const METADATA_MAX_BYTES: u64 = 32 * 1024 * 1024;
/// Upper bound on the tag vocabulary, matching the Python reader's `_MAX_TAGS`.
pub const MAX_TAGS: usize = 100_000;
/// Upper bound on one tag string, matching the Python reader's `_MAX_TAG_LEN`.
pub const MAX_TAG_LEN: usize = 256;

impl WdProfileSpec {
    /// Reject anything this implementation cannot reproduce exactly.
    ///
    /// Silently approximating an unsupported strategy is the failure this
    /// whole mechanism exists to prevent: it yields plausible tags rather
    /// than an error, so nothing downstream can tell that the model ran with
    /// the wrong recipe.
    pub fn validate(&self) -> Result<(), InferError> {
        if self.preprocess_spec.resize_strategy != "longest_side_pad" {
            return Err(InferError::UnsupportedProfile(format!(
                "resize_strategy={} not supported",
                self.preprocess_spec.resize_strategy
            )));
        }
        match self.preprocess_spec.channel_order.as_str() {
            "RGB" | "BGR" => {}
            other => {
                return Err(InferError::UnsupportedProfile(format!(
                    "channel_order={other} not supported"
                )))
            }
        }
        match self.preprocess_spec.layout.as_str() {
            "NHWC" | "NCHW" => {}
            other => {
                return Err(InferError::UnsupportedProfile(format!(
                    "layout={other} not supported"
                )))
            }
        }
        if let Some(activation) = self.output_spec.activation.as_deref() {
            if activation != "none" && activation != "sigmoid" {
                return Err(InferError::UnsupportedProfile(format!(
                    "activation={activation} not supported"
                )));
            }
        }
        if let Some(std) = self.preprocess_spec.std {
            if std.contains(&0.0) {
                return Err(InferError::UnsupportedProfile(
                    "preprocess_spec.std contains zero".to_string(),
                ));
            }
        }
        if self.preprocess_spec.input_size < 32 || self.preprocess_spec.input_size > 2048 {
            return Err(InferError::UnsupportedProfile(format!(
                "input_size={} out of range [32,2048]",
                self.preprocess_spec.input_size
            )));
        }
        if !matches!(
            self.categories_mode.as_str(),
            "from_tag_source" | "all_general"
        ) {
            return Err(InferError::UnsupportedProfile(format!(
                "categories_mode={} not supported",
                self.categories_mode
            )));
        }
        validate_basename(&self.model_file, "model_file")?;
        match &self.tag_source {
            TagSourceSpec::Csv { file, .. } | TagSourceSpec::JsonDict { file, .. } => {
                validate_basename(file, "tag_source.file")?;
            }
        }
        Ok(())
    }

    /// Per-category threshold, mirroring Python's `ThresholdTable.for_tag` in
    /// `global_per_category` mode: the category's own value, else `general`,
    /// else 0.35.
    pub fn threshold_for(&self, category: &str) -> f32 {
        self.default_thresholds
            .get(category)
            .or_else(|| self.default_thresholds.get("general"))
            .copied()
            .unwrap_or(0.35)
    }
}

/// Reject anything that is not a plain file name.
///
/// These names arrive over the wire and are joined onto the model directory,
/// so a separator or `..` would reach outside it.
fn validate_basename(name: &str, field: &str) -> Result<(), InferError> {
    let bad = name.is_empty()
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || name.starts_with('.')
        || name
            .chars()
            .any(|c| c.is_control() || c == ':' || c == '\0');
    if bad {
        return Err(InferError::UnsupportedProfile(format!(
            "{field}={name:?} is not a plain file name"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(json: serde_json::Value) -> WdProfileSpec {
        serde_json::from_value(json).expect("profile deserializes")
    }

    fn minimal() -> serde_json::Value {
        serde_json::json!({
            "model_file": "model.onnx",
            "preprocess_spec": {"input_size": 448},
            "tag_source": {"type": "csv", "file": "selected_tags.csv"},
        })
    }

    #[test]
    fn omitted_preprocess_fields_fall_back_to_the_python_defaults() {
        let s = spec(minimal());
        assert_eq!(s.preprocess_spec.resize_strategy, "longest_side_pad");
        assert_eq!(s.preprocess_spec.pad_color, [255, 255, 255]);
        assert_eq!(s.preprocess_spec.channel_order, "RGB");
        assert_eq!(s.preprocess_spec.scale, 1.0);
        assert_eq!(s.preprocess_spec.mean, None);
        assert_eq!(s.preprocess_spec.std, None);
        assert_eq!(s.preprocess_spec.layout, "NHWC");
        assert_eq!(s.categories_mode, "from_tag_source");
        assert!(!s.output_spec.wants_sigmoid());
        assert_eq!(s.output_spec.output_key, None);
    }

    #[test]
    fn the_shipped_wd_profile_shape_round_trips() {
        let s = spec(serde_json::json!({
            "model_file": "model.onnx",
            "preprocess_spec": {
                "input_size": 448, "resize_strategy": "longest_side_pad",
                "pad_color": [255, 255, 255], "channel_order": "BGR",
                "dtype": "float32", "scale": 1.0, "mean": null, "std": null,
                "layout": "NHWC"
            },
            "tag_source": {
                "type": "csv", "file": "selected_tags.csv", "delimiter": ",",
                "name_col": "name", "category_col": "category",
                "category_map": {"0": "general", "4": "character", "9": "rating"}
            },
            "default_thresholds": {"general": 0.35, "character": 0.85, "rating": 0.0},
            "supports_categories": ["general", "character", "rating"]
        }));
        s.validate().expect("shipped WD profile is supported");
        assert_eq!(s.preprocess_spec.channel_order, "BGR");
        assert_eq!(s.threshold_for("character"), 0.85);
        assert_eq!(s.threshold_for("general"), 0.35);
        match &s.tag_source {
            TagSourceSpec::Csv { category_map, .. } => {
                assert_eq!(category_map.get("4").map(String::as_str), Some("character"));
            }
            other => panic!("expected csv tag source, got {other:?}"),
        }
    }

    #[test]
    fn the_shipped_camie_profile_shape_round_trips() {
        let s = spec(serde_json::json!({
            "model_file": "camie-tagger-v2.onnx",
            "preprocess_spec": {
                "input_size": 512, "resize_strategy": "longest_side_pad",
                "pad_color": [255, 255, 255], "channel_order": "RGB",
                "dtype": "float32", "scale": 0.00392156862745098,
                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                "layout": "NCHW"
            },
            "tag_source": {
                "type": "json_dict", "file": "camie-tagger-v2-metadata.json",
                "container_key": "dataset_info.tag_mapping",
                "idx_to_tag_key": "idx_to_tag",
                "tag_to_category_key": "tag_to_category"
            },
            "output_spec": {"output_key": "refined_predictions", "activation": "sigmoid"},
            "default_thresholds": {
                "general": 0.55, "character": 0.85, "rating": 0.0, "year": 0.7,
                "meta": 0.7, "artist": 0.7, "copyright": 0.7
            },
            "supports_categories": [
                "general", "rating", "meta", "year", "character", "artist", "copyright"
            ]
        }));
        s.validate().expect("shipped camie profile is supported");
        assert_eq!(s.preprocess_spec.layout, "NCHW");
        assert!(s.output_spec.wants_sigmoid());
        assert_eq!(
            s.output_spec.output_key.as_deref(),
            Some("refined_predictions")
        );
        assert_eq!(s.threshold_for("meta"), 0.7);
        // An unlisted category falls back to `general`, as Python's
        // ThresholdTable does — not to a hardcoded constant.
        assert_eq!(s.threshold_for("no_such_category"), 0.55);
    }

    #[test]
    fn threshold_falls_back_to_0_35_only_when_general_is_absent_too() {
        let mut s = spec(minimal());
        s.default_thresholds.clear();
        assert_eq!(s.threshold_for("general"), 0.35);
        s.default_thresholds.insert("general".into(), 0.5);
        assert_eq!(s.threshold_for("anything"), 0.5);
    }

    #[test]
    fn unsupported_recipes_are_rejected_rather_than_approximated() {
        let cases: Vec<(&str, serde_json::Value)> = vec![
            (
                "resize_strategy",
                serde_json::json!({"preprocess_spec": {"input_size": 448, "resize_strategy": "stretch"}}),
            ),
            (
                "channel_order",
                serde_json::json!({"preprocess_spec": {"input_size": 448, "channel_order": "YUV"}}),
            ),
            (
                "layout",
                serde_json::json!({"preprocess_spec": {"input_size": 448, "layout": "NWHC"}}),
            ),
            (
                "activation",
                serde_json::json!({"output_spec": {"activation": "softmax"}}),
            ),
            (
                "std",
                serde_json::json!({"preprocess_spec": {"input_size": 448, "std": [1.0, 0.0, 1.0]}}),
            ),
            (
                "input_size",
                serde_json::json!({"preprocess_spec": {"input_size": 16}}),
            ),
            (
                "categories_mode",
                serde_json::json!({"categories_mode": "invented"}),
            ),
        ];
        for (label, overlay) in cases {
            let mut base = minimal();
            for (k, v) in overlay.as_object().unwrap() {
                base[k] = v.clone();
            }
            let err = spec(base)
                .validate()
                .expect_err(&format!("{label} must be rejected"));
            assert!(
                matches!(err, InferError::UnsupportedProfile(_)),
                "{label}: wrong error {err:?}"
            );
        }
    }

    #[test]
    fn file_names_that_escape_the_model_directory_are_rejected() {
        for bad in [
            "../model.onnx",
            "sub/model.onnx",
            "sub\\model.onnx",
            "",
            ".hidden",
            "a\nb.onnx",
        ] {
            let mut base = minimal();
            base["model_file"] = serde_json::json!(bad);
            let err = spec(base)
                .validate()
                .unwrap_err_or_panic(&format!("model_file={bad:?} must be rejected"));
            assert!(matches!(err, InferError::UnsupportedProfile(_)));
        }
        for bad in ["../selected_tags.csv", "d/selected_tags.csv"] {
            let mut base = minimal();
            base["tag_source"] = serde_json::json!({"type": "csv", "file": bad});
            let err = spec(base)
                .validate()
                .unwrap_err_or_panic(&format!("tag_source.file={bad:?} must be rejected"));
            assert!(matches!(err, InferError::UnsupportedProfile(_)));
        }
        // The names the shipped profiles actually use must survive.
        for good in ["model.onnx", "selected_tags.csv", "camie-tagger-v2.onnx"] {
            let mut base = minimal();
            base["model_file"] = serde_json::json!(good);
            spec(base).validate().expect("shipped names stay valid");
        }
    }

    /// Small helper so the failure message names the input that slipped past.
    trait UnwrapErrOrPanic<T, E> {
        fn unwrap_err_or_panic(self, msg: &str) -> E;
    }

    impl<T: std::fmt::Debug, E> UnwrapErrOrPanic<T, E> for Result<T, E> {
        fn unwrap_err_or_panic(self, msg: &str) -> E {
            match self {
                Ok(value) => panic!("{msg}; got Ok({value:?})"),
                Err(err) => err,
            }
        }
    }

    #[test]
    fn an_unknown_tag_source_type_fails_to_deserialize() {
        let mut base = minimal();
        base["tag_source"] = serde_json::json!({"type": "json_list", "file": "tags.json"});
        let parsed: Result<WdProfileSpec, _> = serde_json::from_value(base);
        assert!(
            parsed.is_err(),
            "json_list has no shipped consumer and must not be silently accepted"
        );
    }
}
