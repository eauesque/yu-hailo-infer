use std::{collections::HashMap, path::Path};

use serde::Deserialize;

use crate::InferError;

#[derive(Deserialize)]
struct CsvRow {
    #[allow(dead_code)]
    tag_id: i64,
    name: String,
    category: i64,
    #[allow(dead_code)]
    count: Option<i64>,
}

pub struct TagMeta {
    pub tags: Vec<(String, String)>,
    pub rating_indices: Vec<usize>,
}

fn default_category_map() -> HashMap<i64, String> {
    [
        (0, "general"),
        (1, "artist"),
        (3, "copyright"),
        (4, "character"),
        (5, "meta"),
        (9, "rating"),
    ]
    .into_iter()
    .map(|(k, v)| (k, v.to_string()))
    .collect()
}

fn load_category_map_from_profile(model_dir: &Path) -> Option<HashMap<i64, String>> {
    let json_path = std::fs::read_dir(model_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))?
        .path();
    let json: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(json_path).ok()?).ok()?;
    let map = json.get("category_map")?.as_object()?;
    Some(
        map.iter()
            .filter_map(|(k, v)| Some((k.parse::<i64>().ok()?, v.as_str()?.to_string())))
            .collect(),
    )
}

pub fn load_tags(model_dir: &Path) -> Result<TagMeta, InferError> {
    let path = model_dir.join("selected_tags.csv");
    let mut reader = csv::Reader::from_path(&path)?;
    let cat_map = load_category_map_from_profile(model_dir).unwrap_or_else(default_category_map);
    let mut tags = Vec::new();
    let mut rating_indices = Vec::new();
    for (i, record) in reader.deserialize::<CsvRow>().enumerate() {
        let row = record?;
        let cat_str = cat_map
            .get(&row.category)
            .cloned()
            .unwrap_or_else(|| "general".to_string());
        if cat_str == "rating" {
            rating_indices.push(i);
        }
        tags.push((row.name, cat_str));
    }
    Ok(TagMeta {
        tags,
        rating_indices,
    })
}
