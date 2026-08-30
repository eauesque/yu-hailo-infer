//! Native Hailo YOLO output decoding, NMS, and coordinate conversion.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::yolo_labels::get_label;

/// Metadata needed to decode a Hailo output tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantMeta {
    pub name: String,
    pub scale: f32,
    pub zero_point: f32,
    pub is_float32: bool,
    pub shape: [usize; 3],
    /// HailoRT format type: UINT8=1, UINT16=2, FLOAT32=3.
    pub format_type: u64,
}

/// A Hailo YOLO output tensor together with the metadata needed to decode it.
#[derive(Debug, Clone, PartialEq)]
pub struct YoloOutputBuffer {
    pub data: Vec<u8>,
    pub meta: QuantMeta,
}

/// Letterbox mapping values supplied by the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleInfo {
    pub orig_w: u32,
    pub orig_h: u32,
    pub scale: f64,
    pub pad_x: f64,
    pub pad_y: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f64,
    pub bbox: [f64; 4],
}

/// HailoRT format types this decoder understands: UINT8=1 and FLOAT32=3.
const SUPPORTED_FORMAT_TYPES: [u64; 2] = [1, 3];

/// Preserves the sidecar-boundary validation formerly performed by yu-server.
pub fn validate_yolo_outputs(outputs: &[YoloOutputBuffer]) -> Result<(), String> {
    if outputs.is_empty() {
        return Err("yu-infer response contained no output tensors".to_string());
    }

    let is_single_nms_output =
        outputs.len() == 1 && outputs[0].meta.name.to_ascii_lowercase().contains("nms");

    for output in outputs {
        let meta = &output.meta;
        if !SUPPORTED_FORMAT_TYPES.contains(&meta.format_type) {
            return Err(format!(
                "yu-infer output '{}' has unsupported format_type {}",
                meta.name, meta.format_type
            ));
        }
        if is_single_nms_output {
            const NMS_ROW_BYTES: usize = 6 * 4;
            if !meta.is_float32 || output.data.len() % NMS_ROW_BYTES != 0 {
                return Err(format!(
                    "yu-infer NMS output '{}' data length {} is not a multiple of {NMS_ROW_BYTES} f32 bytes",
                    meta.name,
                    output.data.len()
                ));
            }
        } else {
            let element_size: usize = if meta.is_float32 { 4 } else { 1 };
            let expected_len = meta.shape.iter().product::<usize>() * element_size;
            if output.data.len() != expected_len {
                return Err(format!(
                    "yu-infer output '{}' data length {} does not match shape {:?} (expected {expected_len} bytes)",
                    meta.name,
                    output.data.len(),
                    meta.shape
                ));
            }
        }
    }
    Ok(())
}

pub fn dequantize(data: &[u8], scale: f32, zero_point: f32, is_float32: bool) -> Vec<f32> {
    if is_float32 {
        data.as_chunks::<4>()
            .0
            .iter()
            .map(|&bytes| f32::from_le_bytes(bytes))
            .collect()
    } else {
        data.iter()
            .map(|&value| (value as f32 - zero_point) * scale)
            .collect()
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn xywh_to_xyxy([cx, cy, width, height]: [f64; 4]) -> [f64; 4] {
    [
        cx - width / 2.0,
        cy - height / 2.0,
        cx + width / 2.0,
        cy + height / 2.0,
    ]
}

fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    intersection / (area_a + area_b - intersection + 1e-6)
}

/// Python `nms_numpy` equivalent: score-descending, greedy IoU suppression.
pub fn nms(boxes_xyxy: &[[f64; 4]], scores: &[f64], iou_threshold: f64) -> Vec<usize> {
    let count = boxes_xyxy.len().min(scores.len());
    if count == 0 {
        return Vec::new();
    }

    let mut remaining: Vec<usize> = (0..count).collect();
    remaining.sort_by(|&left, &right| {
        scores[right]
            .total_cmp(&scores[left])
            .then_with(|| right.cmp(&left))
    });

    let mut keep = Vec::new();
    while let Some(&index) = remaining.first() {
        keep.push(index);
        remaining = remaining[1..]
            .iter()
            .copied()
            .filter(|&candidate| iou(boxes_xyxy[index], boxes_xyxy[candidate]) <= iou_threshold)
            .collect();
    }
    keep
}

pub fn is_nms_output(outputs: &[YoloOutputBuffer]) -> bool {
    outputs.len() == 1
        && outputs[0].meta.is_float32
        && outputs[0].meta.name.to_ascii_lowercase().contains("nms")
}

/// Parses Hailo's embedded-NMS tensor with rows `[y1, x1, y2, x2, score, class_id]`.
pub fn parse_nms_output(buf: &YoloOutputBuffer) -> (Vec<[f64; 4]>, Vec<f64>, Vec<u32>) {
    let values = dequantize(
        &buf.data,
        buf.meta.scale,
        buf.meta.zero_point,
        buf.meta.is_float32,
    );
    let mut boxes = Vec::new();
    let mut scores = Vec::new();
    let mut class_ids = Vec::new();

    for row in values.as_chunks::<6>().0 {
        let score = row[4] as f64;
        if score > 0.0 {
            boxes.push([row[1] as f64, row[0] as f64, row[3] as f64, row[2] as f64]);
            scores.push(score);
            class_ids.push(row[5] as u32);
        }
    }
    (boxes, scores, class_ids)
}

/// Decodes the grid/stride-relative Hailo YOLO output tensors used by the Python backend.
pub fn decode_hailo_yolo_outputs(
    outputs: &[YoloOutputBuffer],
    num_classes: usize,
    input_size: u32,
) -> (Vec<[f64; 4]>, Vec<f64>, Vec<u32>) {
    let mut all_boxes = Vec::new();
    let mut all_scores = Vec::new();
    let mut all_class_ids = Vec::new();

    for output in outputs {
        let [grid_h, grid_w, channels] = output.meta.shape;
        if grid_h == 0 || grid_w == 0 || channels < 4 + num_classes {
            continue;
        }

        let values = dequantize(
            &output.data,
            output.meta.scale,
            output.meta.zero_point,
            output.meta.is_float32,
        );
        let required = match grid_h
            .checked_mul(grid_w)
            .and_then(|cells| cells.checked_mul(channels))
        {
            Some(required) => required,
            None => continue,
        };
        if values.len() != required {
            continue;
        }

        let stride = input_size as f64 / grid_h as f64;
        for grid_y in 0..grid_h {
            for grid_x in 0..grid_w {
                let offset = (grid_y * grid_w + grid_x) * channels;
                let cx = (sigmoid(values[offset] as f64) + grid_x as f64) * stride;
                let cy = (sigmoid(values[offset + 1] as f64) + grid_y as f64) * stride;
                let width = (values[offset + 2] as f64).exp() * stride;
                let height = (values[offset + 3] as f64).exp() * stride;

                let mut best_class = 0usize;
                let mut best_score = f64::NEG_INFINITY;
                for class_index in 0..num_classes {
                    let score = sigmoid(values[offset + 4 + class_index] as f64);
                    if score > best_score {
                        best_score = score;
                        best_class = class_index;
                    }
                }

                all_boxes.push(xywh_to_xyxy([cx, cy, width, height]));
                all_scores.push(best_score);
                all_class_ids.push(best_class as u32);
            }
        }
    }
    (all_boxes, all_scores, all_class_ids)
}

pub fn postprocess_yolo_outputs(
    outputs: &[YoloOutputBuffer],
    conf_threshold: f64,
    iou_threshold: f64,
    num_classes: usize,
    input_size: u32,
    scale_info: &ScaleInfo,
) -> Result<Vec<Detection>, String> {
    validate_yolo_outputs(outputs)?;
    let embedded_nms = is_nms_output(outputs);
    let (boxes, scores, class_ids) = if embedded_nms {
        parse_nms_output(&outputs[0])
    } else {
        decode_hailo_yolo_outputs(outputs, num_classes, input_size)
    };

    let mut candidates: Vec<([f64; 4], f64, u32)> = boxes
        .into_iter()
        .zip(scores)
        .zip(class_ids)
        .filter_map(|((bbox, confidence), class_id)| {
            (confidence >= conf_threshold).then_some((bbox, confidence, class_id))
        })
        .collect();

    if !embedded_nms {
        let class_ids: BTreeSet<u32> = candidates.iter().map(|candidate| candidate.2).collect();
        let mut kept = Vec::new();
        for class_id in class_ids {
            let indices: Vec<usize> = candidates
                .iter()
                .enumerate()
                .filter_map(|(index, candidate)| (candidate.2 == class_id).then_some(index))
                .collect();
            let class_boxes: Vec<[f64; 4]> =
                indices.iter().map(|&index| candidates[index].0).collect();
            let class_scores: Vec<f64> = indices.iter().map(|&index| candidates[index].1).collect();
            kept.extend(
                nms(&class_boxes, &class_scores, iou_threshold)
                    .into_iter()
                    .map(|index| candidates[indices[index]]),
            );
        }
        candidates = kept;
    }

    let input_size = input_size as f64;
    let mut detections: Vec<Detection> = candidates
        .into_iter()
        .map(|(bbox, confidence, class_id)| {
            let bbox = if scale_info.scale > 0.0 {
                [
                    (bbox[0] - scale_info.pad_x) / scale_info.scale / scale_info.orig_w as f64,
                    (bbox[1] - scale_info.pad_y) / scale_info.scale / scale_info.orig_h as f64,
                    (bbox[2] - scale_info.pad_x) / scale_info.scale / scale_info.orig_w as f64,
                    (bbox[3] - scale_info.pad_y) / scale_info.scale / scale_info.orig_h as f64,
                ]
            } else {
                bbox.map(|value| value / input_size)
            };
            Detection {
                class_id,
                class_name: get_label(class_id),
                confidence: round_to_four(confidence),
                bbox: bbox.map(|value| round_to_four(value.clamp(0.0, 1.0))),
            }
        })
        .collect();
    detections.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    Ok(detections)
}

pub fn round_to_four(value: f64) -> f64 {
    (value * 10_000.0).round_ties_even() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float_output(name: &str, shape: [usize; 3], values: &[f32]) -> YoloOutputBuffer {
        YoloOutputBuffer {
            data: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            meta: QuantMeta {
                name: name.to_owned(),
                scale: 1.0,
                zero_point: 0.0,
                is_float32: true,
                shape,
                format_type: 3,
            },
        }
    }

    fn unit_scale() -> ScaleInfo {
        ScaleInfo {
            orig_w: 640,
            orig_h: 640,
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        }
    }

    #[test]
    fn nms_removes_overlapping_lower_score_box() {
        let boxes = vec![[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]];
        let scores = vec![0.9, 0.5];
        assert_eq!(nms(&boxes, &scores, 0.45), vec![0]);
    }

    #[test]
    fn nms_keeps_non_overlapping_boxes() {
        let boxes = vec![[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]];
        let scores = vec![0.9, 0.8];
        assert_eq!(nms(&boxes, &scores, 0.45).len(), 2);
    }

    #[test]
    fn nms_empty_input_returns_empty() {
        assert!(nms(&[], &[], 0.45).is_empty());
    }

    #[test]
    fn dequantize_uint8_applies_scale_and_zero_point() {
        let out = dequantize(&[128, 0, 255], 0.5, 128.0, false);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] + 64.0).abs() < 1e-6);
        assert!((out[2] - 63.5).abs() < 1e-6);
    }

    #[test]
    fn dequantize_float32_reinterprets_bytes() {
        let out = dequantize(&1.5f32.to_le_bytes(), 1.0, 0.0, true);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn decode_hailo_yolo_outputs_applies_grid_stride_and_sigmoid() {
        let output = float_output("head", [1, 1, 5], &[0.0; 5]);
        let (boxes, scores, class_ids) = decode_hailo_yolo_outputs(&[output], 1, 640);
        assert_eq!(boxes, vec![[0.0, 0.0, 640.0, 640.0]]);
        assert!((scores[0] - 0.5).abs() < 1e-6);
        assert_eq!(class_ids, vec![0]);
    }

    #[test]
    fn decode_hailo_yolo_outputs_offsets_by_nonzero_grid_cell_and_applies_exp_to_wh() {
        let mut values = vec![0.0f32; 2 * 2 * 5];
        let cell_offset = (2 + 1) * 5;
        values[cell_offset..cell_offset + 5].copy_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        let output = float_output("head", [2, 2, 5], &values);
        let (boxes, _scores, _class_ids) = decode_hailo_yolo_outputs(&[output], 1, 640);
        assert_eq!(boxes[cell_offset / 5], [320.0, 320.0, 640.0, 640.0]);
    }

    #[test]
    fn decode_hailo_yolo_outputs_picks_argmax_class_across_multiple_classes() {
        let values = [0.0, 0.0, 0.0, 0.0, -5.0, 5.0, -5.0];
        let output = float_output("head", [1, 1, 7], &values);
        let (_boxes, scores, class_ids) = decode_hailo_yolo_outputs(&[output], 3, 640);
        assert_eq!(class_ids, vec![1]);
        assert!(scores[0] > 0.99);
    }

    #[test]
    fn decode_hailo_yolo_outputs_concatenates_multiple_heads() {
        let head_a = float_output("head_a", [1, 1, 5], &[0.0; 5]);
        let head_b = float_output("head_b", [1, 1, 5], &[0.0; 5]);
        let (boxes, _scores, _class_ids) = decode_hailo_yolo_outputs(&[head_a, head_b], 1, 640);
        assert_eq!(boxes.len(), 2);
    }

    #[test]
    fn decode_hailo_yolo_outputs_dequantizes_uint8_input() {
        let output = YoloOutputBuffer {
            data: vec![128u8; 5],
            meta: QuantMeta {
                name: "head".to_owned(),
                scale: 1.0,
                zero_point: 128.0,
                is_float32: false,
                shape: [1, 1, 5],
                format_type: 1,
            },
        };
        let (boxes, scores, class_ids) = decode_hailo_yolo_outputs(&[output], 1, 640);
        assert_eq!(boxes, vec![[0.0, 0.0, 640.0, 640.0]]);
        assert!((scores[0] - 0.5).abs() < 1e-6);
        assert_eq!(class_ids, vec![0]);
    }

    #[test]
    fn parse_nms_output_filters_zero_score_and_reorders_columns() {
        let output = float_output(
            "nms",
            [2, 1, 6],
            &[
                10.0, 20.0, 30.0, 40.0, 0.9, 3.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0,
            ],
        );
        let (boxes, scores, class_ids) = parse_nms_output(&output);
        assert_eq!(boxes, vec![[20.0, 10.0, 40.0, 30.0]]);
        assert_eq!(scores.len(), 1);
        assert!((scores[0] - 0.9).abs() < 1e-6);
        assert_eq!(class_ids, vec![3]);
    }

    #[test]
    fn is_nms_output_true_only_for_single_float32_nms_named_output() {
        let output = float_output("postprocess_nms", [1, 1, 6], &[0.0; 6]);
        assert!(is_nms_output(std::slice::from_ref(&output)));
        assert!(!is_nms_output(&[output.clone(), output.clone()]));
        let mut integer = output;
        integer.meta.is_float32 = false;
        assert!(!is_nms_output(&[integer]));
    }

    #[test]
    fn postprocess_yolo_outputs_filters_by_confidence_and_sorts_by_confidence_desc() {
        let output = float_output(
            "nms",
            [3, 1, 6],
            &[
                0.0, 0.0, 320.0, 320.0, 0.8, 1.0, 320.0, 320.0, 640.0, 640.0, 0.9, 2.0, 0.0, 0.0,
                10.0, 10.0, 0.2, 3.0,
            ],
        );
        let detections =
            postprocess_yolo_outputs(&[output], 0.25, 0.45, 80, 640, &unit_scale()).unwrap();
        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0].class_id, 2);
        assert_eq!(detections[1].class_id, 1);
        assert!(detections[0].confidence > detections[1].confidence);
    }

    #[test]
    fn postprocess_yolo_outputs_non_embedded_nms_path_applies_class_nms_and_scale_info() {
        let output = float_output("head", [1, 1, 5], &[0.0, 0.0, 0.0, 0.0, 10.0]);
        let scale_info = ScaleInfo {
            orig_w: 1280,
            orig_h: 1280,
            scale: 0.5,
            pad_x: 0.0,
            pad_y: 0.0,
        };
        let detections =
            postprocess_yolo_outputs(&[output], 0.25, 0.45, 1, 640, &scale_info).unwrap();
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].class_id, 0);
        assert_eq!(detections[0].bbox, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn postprocess_migration_parity_covers_embedded_nms_and_grid_decode() {
        let nms = float_output("nms", [1, 1, 6], &[10.0, 20.0, 30.0, 40.0, 0.9, 3.0]);
        assert_eq!(
            postprocess_yolo_outputs(&[nms], 0.25, 0.45, 80, 640, &unit_scale()).unwrap(),
            vec![Detection {
                class_id: 3,
                class_name: "motorcycle".to_owned(),
                confidence: 0.9,
                bbox: [0.0312, 0.0156, 0.0625, 0.0469]
            }]
        );
        // Grid path requires channels >= 4 (bbox) + num_classes; use the real
        // 80-class head width so this matches yu-server's former hardcoded
        // num_classes=80 call exactly, instead of the smaller fixtures used
        // by the other (num_classes=1) tests above.
        let mut values = vec![0.0f32; 84];
        values[4] = 10.0; // class 0 ("person") logit, high confidence
        let grid = float_output("head", [1, 1, 84], &values);
        assert_eq!(
            postprocess_yolo_outputs(&[grid], 0.25, 0.45, 80, 640, &unit_scale()).unwrap(),
            vec![Detection {
                class_id: 0,
                class_name: "person".to_owned(),
                confidence: 1.0,
                bbox: [0.0, 0.0, 1.0, 1.0]
            }]
        );
    }

    #[test]
    fn validation_rejects_unsupported_format_type() {
        let mut output = float_output("head", [1, 1, 5], &[0.0; 5]);
        output.meta.format_type = 2;
        let error =
            postprocess_yolo_outputs(&[output], 0.25, 0.45, 80, 640, &unit_scale()).unwrap_err();
        assert!(error.contains("unsupported format_type"), "{error}");
    }

    #[test]
    fn validation_rejects_grid_length_mismatch() {
        let output = YoloOutputBuffer {
            data: vec![0; 2],
            meta: QuantMeta {
                name: "head".to_owned(),
                scale: 1.0,
                zero_point: 0.0,
                is_float32: false,
                shape: [1, 1, 5],
                format_type: 1,
            },
        };
        let error =
            postprocess_yolo_outputs(&[output], 0.25, 0.45, 80, 640, &unit_scale()).unwrap_err();
        assert!(error.contains("does not match shape"), "{error}");
    }

    #[test]
    fn validation_accepts_nms_with_bogus_shape_and_rejects_partial_row() {
        let output = float_output("nms", [80, 100, 8000], &[0.0; 6]);
        assert!(validate_yolo_outputs(&[output]).is_ok());
        let output = YoloOutputBuffer {
            data: vec![0; 23],
            meta: QuantMeta {
                name: "nms".to_owned(),
                scale: 1.0,
                zero_point: 0.0,
                is_float32: true,
                shape: [80, 100, 8000],
                format_type: 3,
            },
        };
        assert!(validate_yolo_outputs(&[output])
            .unwrap_err()
            .contains("multiple of 24"));
    }
}
