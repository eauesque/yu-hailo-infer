//! Ordered `(tag, category)` vocabulary loading, driven by the profile.
//!
//! Mirrors `extensions/builtin_wd_tagger/core_impl/adapters/tag_source.py`.
//! The two shapes below are the two the shipped profiles use; the caller
//! rejects the rest before reaching here.
//!
//! This module deliberately has no "find a JSON in the model directory and
//! hope it is a profile" fallback. That heuristic used to live here and was
//! unsafe in two ways: `read_dir` order is unspecified, and for camie the
//! first JSON is the 7 MB tag metadata, not a profile.

use std::{
    collections::{BTreeMap, HashSet},
    fmt,
    path::Path,
};

use serde::{
    de::{MapAccess, Visitor},
    Deserialize, Deserializer,
};

use crate::{
    profile::{TagSourceSpec, MAX_TAGS, MAX_TAG_LEN, METADATA_MAX_BYTES},
    InferError,
};

#[derive(Debug)]
pub struct TagMeta {
    pub tags: Vec<(String, String)>,
    pub rating_indices: Vec<usize>,
}

/// Read a metadata JSON under the same size cap and BOM tolerance as Python's
/// `_read_json_bounded`, rejecting duplicate keys at every level.
fn read_json_bounded(path: &Path) -> Result<serde_json::Value, InferError> {
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > METADATA_MAX_BYTES {
        return Err(InferError::TagMetadata(format!(
            "{} exceeds {METADATA_MAX_BYTES} bytes",
            path.display()
        )));
    }
    let raw = std::fs::read(path)?;
    let text = std::str::from_utf8(&raw)
        .map_err(|e| InferError::TagMetadata(format!("{} not UTF-8: {e}", path.display())))?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    // Parse once through the dedup-aware map so a repeated key is an error at
    // every nesting level, then hand back a plain Value for navigation.
    let checked: DedupValue = serde_json::from_str(text)
        .map_err(|e| InferError::TagMetadata(format!("{} invalid JSON: {e}", path.display())))?;
    Ok(checked.0)
}

/// `serde_json::Value` whose object nodes reject duplicate keys.
struct DedupValue(serde_json::Value);

impl<'de> Deserialize<'de> for DedupValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = DedupValue;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value with unique object keys")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<DedupValue, A::Error> {
                let mut out = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    let value = map.next_value::<DedupValue>()?;
                    if out.contains_key(&key) {
                        return Err(serde::de::Error::custom(format!("duplicate key {key:?}")));
                    }
                    out.insert(key, value.0);
                }
                Ok(DedupValue(serde_json::Value::Object(out)))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<DedupValue, A::Error> {
                let mut out = Vec::new();
                while let Some(item) = seq.next_element::<DedupValue>()? {
                    out.push(item.0);
                }
                Ok(DedupValue(serde_json::Value::Array(out)))
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<DedupValue, E> {
                Ok(DedupValue(serde_json::Value::String(v.to_string())))
            }
            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<DedupValue, E> {
                Ok(DedupValue(serde_json::Value::String(v)))
            }
            fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<DedupValue, E> {
                Ok(DedupValue(serde_json::Value::Bool(v)))
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<DedupValue, E> {
                Ok(DedupValue(serde_json::Value::Null))
            }
            fn visit_none<E: serde::de::Error>(self) -> Result<DedupValue, E> {
                Ok(DedupValue(serde_json::Value::Null))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<DedupValue, E> {
                Ok(DedupValue(serde_json::Value::from(v)))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<DedupValue, E> {
                Ok(DedupValue(serde_json::Value::from(v)))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<DedupValue, E> {
                Ok(DedupValue(serde_json::Value::from(v)))
            }
        }
        deserializer.deserialize_any(V)
    }
}

fn validate_caps(pairs: &[(String, String)], ctx: &str) -> Result<(), InferError> {
    if pairs.len() > MAX_TAGS {
        return Err(InferError::TagMetadata(format!(
            "{ctx} produced {} tags exceeding cap {MAX_TAGS}",
            pairs.len()
        )));
    }
    if let Some((tag, _)) = pairs.iter().find(|(t, _)| t.len() > MAX_TAG_LEN) {
        return Err(InferError::TagMetadata(format!(
            "{ctx} contains tag of length {} exceeding {MAX_TAG_LEN}",
            tag.len()
        )));
    }
    Ok(())
}

/// Load the ordered vocabulary described by `spec`.
///
/// `categories_mode == "all_general"` rewrites every category, matching
/// Python's `_apply_categories_mode`.
pub fn load_tags(
    model_dir: &Path,
    spec: &TagSourceSpec,
    categories_mode: &str,
    supports_categories: &[String],
) -> Result<TagMeta, InferError> {
    let mut pairs = match spec {
        TagSourceSpec::Csv {
            file,
            delimiter,
            name_col,
            category_col,
            category_map,
        } => load_csv(
            &model_dir.join(file),
            delimiter,
            name_col,
            category_col,
            category_map,
        )?,
        TagSourceSpec::JsonDict {
            file,
            container_key,
            idx_to_tag_key,
            tag_to_category_key,
        } => load_json_dict(
            &model_dir.join(file),
            container_key.as_deref(),
            idx_to_tag_key,
            tag_to_category_key,
            supports_categories,
        )?,
    };

    if categories_mode == "all_general" {
        for entry in &mut pairs {
            entry.1 = "general".to_string();
        }
    }

    let rating_indices = pairs
        .iter()
        .enumerate()
        .filter(|(_, (_, cat))| cat == "rating")
        .map(|(i, _)| i)
        .collect();

    Ok(TagMeta {
        tags: pairs,
        rating_indices,
    })
}

fn load_csv(
    path: &Path,
    delimiter: &str,
    name_col: &str,
    category_col: &str,
    category_map: &BTreeMap<String, String>,
) -> Result<Vec<(String, String)>, InferError> {
    let size = std::fs::metadata(path)?.len();
    if size > METADATA_MAX_BYTES {
        return Err(InferError::TagMetadata(format!(
            "{} exceeds {METADATA_MAX_BYTES} bytes ({size})",
            path.display()
        )));
    }
    let delim_bytes = delimiter.as_bytes();
    if delim_bytes.len() != 1 {
        return Err(InferError::TagMetadata(format!(
            "tag_source.delimiter must be one byte, got {delimiter:?}"
        )));
    }
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delim_bytes[0])
        .has_headers(true)
        // Python's csv.DictReader tolerates a row with fewer fields than the
        // header and yields "" for the rest. The csv crate rejects it by
        // default, which would turn a ragged vocabulary file into a hard
        // error on one side and a loaded model on the other.
        .flexible(true)
        .from_path(path)?;
    let headers = reader.headers()?.clone();
    let name_idx = headers.iter().position(|h| h == name_col);
    let cat_idx = headers.iter().position(|h| h == category_col);

    let mut pairs = Vec::new();
    for record in reader.records() {
        let record = record?;
        // Python's csv.DictReader yields "" for a column that is not present,
        // and the category map falls through to the raw value — not to
        // "general". Reproduce both.
        let name = name_idx
            .and_then(|i| record.get(i))
            .unwrap_or("")
            .to_string();
        let raw_cat = cat_idx.and_then(|i| record.get(i)).unwrap_or("");
        let category = category_map
            .get(raw_cat)
            .cloned()
            .unwrap_or_else(|| raw_cat.to_string());
        pairs.push((name, category));
    }
    validate_caps(&pairs, "tag_source.type=csv")?;
    Ok(pairs)
}

fn load_json_dict(
    path: &Path,
    container_key: Option<&str>,
    idx_to_tag_key: &str,
    tag_to_category_key: &str,
    supports_categories: &[String],
) -> Result<Vec<(String, String)>, InferError> {
    let data = read_json_bounded(path)?;
    let mut container = &data;
    if let Some(ck) = container_key.filter(|c| !c.is_empty()) {
        for part in ck.split('.') {
            container = container.get(part).ok_or_else(|| {
                InferError::TagMetadata(format!(
                    "tag_source.container_key={ck:?}: missing segment {part:?}"
                ))
            })?;
        }
    }
    let idx_to_tag = container
        .get(idx_to_tag_key)
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            InferError::TagMetadata(format!("{idx_to_tag_key:?} must resolve to an object"))
        })?;
    let tag_to_cat = container
        .get(tag_to_category_key)
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            InferError::TagMetadata(format!("{tag_to_category_key:?} must resolve to an object"))
        })?;

    // Keys are stringified ints; order by integer value, as Python does.
    let mut ordered: Vec<(i64, &serde_json::Value)> = Vec::with_capacity(idx_to_tag.len());
    for (key, value) in idx_to_tag {
        let index = key.parse::<i64>().map_err(|_| {
            InferError::TagMetadata(format!(
                "{idx_to_tag_key:?}: keys must be int-castable strings, got {key:?}"
            ))
        })?;
        ordered.push((index, value));
    }
    ordered.sort_by_key(|(index, _)| *index);

    let allowed: HashSet<&str> = supports_categories.iter().map(String::as_str).collect();
    let mut pairs = Vec::with_capacity(ordered.len());
    for (_, value) in ordered {
        let tag = value.as_str().map(str::to_string).unwrap_or_else(|| {
            // Python's str(tag_name) stringifies non-strings; match it.
            value.to_string()
        });
        let category = tag_to_cat
            .get(&tag)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                InferError::TagMetadata(format!("tag_to_category missing entry for tag {tag:?}"))
            })?
            .to_string();
        if !allowed.is_empty() && !allowed.contains(category.as_str()) {
            return Err(InferError::TagMetadata(format!(
                "tag_to_category value {category:?} not in supports_categories {supports_categories:?}"
            )));
        }
        pairs.push((tag, category));
    }
    validate_caps(&pairs, "tag_source.type=json_dict(mapping)")?;
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn csv_spec(category_map: &[(&str, &str)]) -> TagSourceSpec {
        TagSourceSpec::Csv {
            file: "selected_tags.csv".to_string(),
            delimiter: ",".to_string(),
            name_col: "name".to_string(),
            category_col: "category".to_string(),
            category_map: category_map
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    fn json_spec() -> TagSourceSpec {
        TagSourceSpec::JsonDict {
            file: "meta.json".to_string(),
            container_key: Some("dataset_info.tag_mapping".to_string()),
            idx_to_tag_key: "idx_to_tag".to_string(),
            tag_to_category_key: "tag_to_category".to_string(),
        }
    }

    #[test]
    fn csv_maps_categories_and_records_rating_positions() {
        // The rows are deliberately shorter than the header: WD's shipped
        // selected_tags.csv declares a `count` column, and Python's DictReader
        // tolerates its absence. Rejecting the row would load nothing here and
        // everything on the Python side.
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("selected_tags.csv"),
            "tag_id,name,category,count\n0,a,0\n1,r,9\n2,c,4\n",
        )
        .unwrap();
        let meta = load_tags(
            dir.path(),
            &csv_spec(&[("0", "general"), ("4", "character"), ("9", "rating")]),
            "from_tag_source",
            &[],
        )
        .unwrap();
        assert_eq!(
            meta.tags,
            vec![
                ("a".to_string(), "general".to_string()),
                ("r".to_string(), "rating".to_string()),
                ("c".to_string(), "character".to_string()),
            ]
        );
        assert_eq!(meta.rating_indices, vec![1]);
    }

    #[test]
    fn csv_leaves_an_unmapped_category_as_its_raw_value() {
        // Python's parse_tags_csv_with_spec falls through to the raw string.
        // Defaulting to "general" here would mislabel every tag in a category
        // the profile forgot to map, and no test of the mapped categories
        // would notice.
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("selected_tags.csv"),
            "tag_id,name,category\n0,a,0\n1,b,7\n",
        )
        .unwrap();
        let meta = load_tags(
            dir.path(),
            &csv_spec(&[("0", "general")]),
            "from_tag_source",
            &[],
        )
        .unwrap();
        assert_eq!(meta.tags[1], ("b".to_string(), "7".to_string()));
    }

    #[test]
    fn csv_honours_non_default_column_names() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("selected_tags.csv"),
            "label;kind\nfoo;0\nbar;9\n",
        )
        .unwrap();
        let spec = TagSourceSpec::Csv {
            file: "selected_tags.csv".to_string(),
            delimiter: ";".to_string(),
            name_col: "label".to_string(),
            category_col: "kind".to_string(),
            category_map: [("0", "general"), ("9", "rating")]
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        };
        let meta = load_tags(dir.path(), &spec, "from_tag_source", &[]).unwrap();
        assert_eq!(meta.tags[0].0, "foo");
        assert_eq!(meta.rating_indices, vec![1]);
    }

    fn write_json(dir: &Path, body: &str) {
        fs::write(dir.join("meta.json"), body).unwrap();
    }

    #[test]
    fn json_dict_orders_by_integer_key_not_lexicographically() {
        // "10" sorts before "2" as a string. Getting this wrong misaligns
        // every tag past index 9 against the score vector, which produces a
        // full set of confident, wrong tags rather than an error.
        let dir = TempDir::new().unwrap();
        write_json(
            dir.path(),
            r#"{"dataset_info":{"tag_mapping":{
                "idx_to_tag":{"0":"t0","2":"t2","10":"t10"},
                "tag_to_category":{"t0":"general","t2":"general","t10":"rating"}
            }}}"#,
        );
        let meta = load_tags(dir.path(), &json_spec(), "from_tag_source", &[]).unwrap();
        assert_eq!(
            meta.tags
                .iter()
                .map(|(t, _)| t.as_str())
                .collect::<Vec<_>>(),
            vec!["t0", "t2", "t10"]
        );
        assert_eq!(meta.rating_indices, vec![2]);
    }

    #[test]
    fn json_dict_rejects_a_category_outside_supports_categories() {
        let dir = TempDir::new().unwrap();
        write_json(
            dir.path(),
            r#"{"dataset_info":{"tag_mapping":{
                "idx_to_tag":{"0":"t0"},
                "tag_to_category":{"t0":"invented"}
            }}}"#,
        );
        let err = load_tags(
            dir.path(),
            &json_spec(),
            "from_tag_source",
            &["general".to_string()],
        )
        .unwrap_err();
        assert!(matches!(err, InferError::TagMetadata(_)), "{err:?}");
    }

    #[test]
    fn json_dict_rejects_a_tag_with_no_category_entry() {
        let dir = TempDir::new().unwrap();
        write_json(
            dir.path(),
            r#"{"dataset_info":{"tag_mapping":{
                "idx_to_tag":{"0":"t0","1":"orphan"},
                "tag_to_category":{"t0":"general"}
            }}}"#,
        );
        let err = load_tags(dir.path(), &json_spec(), "from_tag_source", &[]).unwrap_err();
        assert!(matches!(err, InferError::TagMetadata(_)), "{err:?}");
    }

    #[test]
    fn json_dict_rejects_a_missing_container_segment() {
        let dir = TempDir::new().unwrap();
        write_json(dir.path(), r#"{"dataset_info":{}}"#);
        let err = load_tags(dir.path(), &json_spec(), "from_tag_source", &[]).unwrap_err();
        assert!(matches!(err, InferError::TagMetadata(_)), "{err:?}");
    }

    #[test]
    fn duplicate_json_keys_are_rejected_as_python_rejects_them() {
        let dir = TempDir::new().unwrap();
        write_json(
            dir.path(),
            r#"{"dataset_info":{"tag_mapping":{
                "idx_to_tag":{"0":"t0","0":"shadowed"},
                "tag_to_category":{"t0":"general","shadowed":"general"}
            }}}"#,
        );
        let err = load_tags(dir.path(), &json_spec(), "from_tag_source", &[]).unwrap_err();
        assert!(matches!(err, InferError::TagMetadata(_)), "{err:?}");
    }

    #[test]
    fn all_general_mode_rewrites_every_category_and_clears_ratings() {
        let dir = TempDir::new().unwrap();
        write_json(
            dir.path(),
            r#"{"dataset_info":{"tag_mapping":{
                "idx_to_tag":{"0":"t0","1":"t1"},
                "tag_to_category":{"t0":"general","t1":"rating"}
            }}}"#,
        );
        let meta = load_tags(dir.path(), &json_spec(), "all_general", &[]).unwrap();
        assert!(meta.tags.iter().all(|(_, c)| c == "general"));
        assert!(meta.rating_indices.is_empty());
    }

    #[test]
    fn a_utf8_bom_does_not_break_the_json_reader() {
        let dir = TempDir::new().unwrap();
        write_json(
            dir.path(),
            "\u{feff}{\"dataset_info\":{\"tag_mapping\":{\
                \"idx_to_tag\":{\"0\":\"t0\"},\
                \"tag_to_category\":{\"t0\":\"general\"}}}}",
        );
        let meta = load_tags(dir.path(), &json_spec(), "from_tag_source", &[]).unwrap();
        assert_eq!(meta.tags.len(), 1);
    }
}
