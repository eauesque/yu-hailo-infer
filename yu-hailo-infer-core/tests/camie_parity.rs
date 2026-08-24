//! Cross-implementation check for the camie tagger.
//!
//! Runs the real 752 MB export through this crate's engine and compares the
//! result against tags produced by the Python adapter on the same fixtures.
//! Skipped unless the three paths are supplied, because neither the model nor
//! the fixtures live in this repository:
//!
//! ```text
//! YU_CAMIE_MODEL_DIR=<dir holding camie-tagger-v2.onnx + metadata>
//! YU_CAMIE_FIXTURE_DIR=<dir holding sample_*.jpg>
//! YU_CAMIE_REFERENCE=<path to camie_reference_tags.json>
//! cargo test -p yu-hailo-infer-core --test camie_parity -- --ignored --nocapture
//! ```
//!
//! Confidences are compared with a tolerance rather than for equality, but the
//! tolerance is small on purpose. The measured worst-case difference across the
//! fixtures is 0.00005 — exactly the granularity of Python's
//! `round(conf, 4)` — which says the resampling and normalisation agree to the
//! last digit either side reports. A loose tolerance here would hide a wrong
//! output head or a skipped activation, since those still yield confident,
//! well-formed tags.

use std::{collections::BTreeMap, path::PathBuf};

use yu_hailo_infer_core::WdInferEngine;

/// Half of Python's `round(conf, 4)` granularity, plus float slack. Measured:
/// the observed worst case is 0.00005. Anything larger means the two
/// implementations are computing different numbers, not rounding differently.
const CONFIDENCE_TOLERANCE: f32 = 1e-4;

fn camie_spec() -> yu_hailo_infer_core::WdProfileSpec {
    serde_json::from_value(serde_json::json!({
        "model_file": "camie-tagger-v2.onnx",
        "preprocess_spec": {
            "input_size": 512,
            "resize_strategy": "longest_side_pad",
            "pad_color": [255, 255, 255],
            "channel_order": "RGB",
            "dtype": "float32",
            "scale": 0.00392156862745098,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "layout": "NCHW"
        },
        "tag_source": {
            "type": "json_dict",
            "file": "camie-tagger-v2-metadata.json",
            "container_key": "dataset_info.tag_mapping",
            "idx_to_tag_key": "idx_to_tag",
            "tag_to_category_key": "tag_to_category"
        },
        "output_spec": {"output_key": "refined_predictions", "activation": "sigmoid"},
        "default_thresholds": {
            "general": 0.55, "character": 0.85, "rating": 0.0, "year": 0.7,
            "meta": 0.7, "artist": 0.7, "copyright": 0.7
        },
        "categories_mode": "from_tag_source",
        "supports_categories": [
            "general", "rating", "meta", "year", "character", "artist", "copyright"
        ]
    }))
    .expect("camie spec is well formed")
}

fn env_dir(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from)
}

#[test]
#[ignore = "needs the 752 MB camie export; see the module comment"]
fn camie_matches_the_python_reference_tags() {
    let (Some(model_dir), Some(fixture_dir), Some(reference)) = (
        env_dir("YU_CAMIE_MODEL_DIR"),
        env_dir("YU_CAMIE_FIXTURE_DIR"),
        env_dir("YU_CAMIE_REFERENCE"),
    ) else {
        panic!(
            "set YU_CAMIE_MODEL_DIR, YU_CAMIE_FIXTURE_DIR and YU_CAMIE_REFERENCE; \
             running this test without them would report a pass having compared nothing"
        );
    };

    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&reference).expect("reference readable"))
            .expect("reference is JSON");
    let reference = reference.as_object().expect("reference is an object");
    assert!(
        !reference.is_empty(),
        "reference holds no fixtures; an empty comparison passes vacuously"
    );

    let engine = WdInferEngine::new(&model_dir, camie_spec()).expect("engine builds");

    let mut failures: Vec<String> = Vec::new();
    let mut worst_delta = 0.0_f32;
    for (name, expected) in reference {
        let path = fixture_dir.join(name);
        let got = engine.run(&path).expect("inference runs");

        let expected_rating = expected["rating"].as_str().unwrap_or_default();
        if got.rating != expected_rating {
            failures.push(format!(
                "{name}: rating {:?} != python {:?}",
                got.rating, expected_rating
            ));
        }

        let mut py: BTreeMap<String, (f32, String)> = BTreeMap::new();
        for entry in expected["tags"].as_array().expect("tags is an array") {
            let tag = entry[0].as_str().expect("tag name").to_string();
            let conf = entry[1].as_f64().expect("confidence") as f32;
            let cat = entry[2].as_str().expect("category").to_string();
            py.insert(tag, (conf, cat));
        }
        let rs: BTreeMap<String, (f32, String)> = got
            .tags
            .iter()
            .map(|t| (t.tag.clone(), (t.confidence, t.category.clone())))
            .collect();

        for (tag, (py_conf, py_cat)) in &py {
            match rs.get(tag) {
                None => failures.push(format!(
                    "{name}: python has {tag:?} at {py_conf:.4}, rust does not"
                )),
                Some((rs_conf, rs_cat)) => {
                    if rs_cat != py_cat {
                        failures.push(format!(
                            "{name}: {tag:?} category {rs_cat:?} != python {py_cat:?}"
                        ));
                    }
                    let delta = (rs_conf - py_conf).abs();
                    worst_delta = worst_delta.max(delta);
                    if delta > CONFIDENCE_TOLERANCE {
                        failures.push(format!(
                            "{name}: {tag:?} confidence {rs_conf:.4} vs python {py_conf:.4} \
                             (delta {delta:.4} over {CONFIDENCE_TOLERANCE})"
                        ));
                    }
                }
            }
        }
        for tag in rs.keys() {
            if !py.contains_key(tag) {
                failures.push(format!(
                    "{name}: rust has {tag:?} at {:.4}, python does not",
                    rs[tag].0
                ));
            }
        }
        eprintln!(
            "{name}: rust {} tags, python {} tags, rating {}",
            rs.len(),
            py.len(),
            got.rating
        );
    }

    eprintln!("worst confidence delta across all fixtures: {worst_delta:.5}");
    assert!(
        failures.is_empty(),
        "{} discrepancies:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
