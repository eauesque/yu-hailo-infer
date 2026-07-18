use super::{shim::ShimYolo, HailoRtError, HailoRtResult, Hef, VStreamDirection, VStreamInfo};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct YoloModelMetadata {
    pub(crate) inputs: Vec<VStreamInfo>,
    pub(crate) outputs: Vec<VStreamInfo>,
}

impl YoloModelMetadata {
    #[cfg(test)]
    pub(crate) fn single_input_for_test(input_frame_size: usize) -> Self {
        Self {
            inputs: vec![VStreamInfo {
                name: "input".to_string(),
                direction: VStreamDirection::Input,
                shape: super::TensorShape::new([1, 1, input_frame_size]),
                quant: super::QuantInfo {
                    zero_point: 0.0,
                    scale: 1.0,
                },
                format_type: super::ffi::HAILO_FORMAT_TYPE_UINT8,
                frame_size: input_frame_size,
            }],
            outputs: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct YoloOutputBuffer {
    pub(crate) name: String,
    pub(crate) data: Vec<u8>,
    pub(crate) info: VStreamInfo,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct YoloInferenceResult {
    pub(crate) outputs: Vec<YoloOutputBuffer>,
}

pub(crate) fn load_yolo_metadata(hef_path: &str) -> HailoRtResult<YoloModelMetadata> {
    let hef = Hef::from_path(hef_path)?;
    let infos = hef.vstream_infos()?;
    let (inputs, outputs): (Vec<_>, Vec<_>) = infos
        .into_iter()
        .partition(|info| info.direction == VStreamDirection::Input);
    if inputs.len() != 1 {
        return Err(HailoRtError::InvalidMetadata(
            "YOLO HEF must expose exactly one input vstream",
        ));
    }
    if outputs.is_empty() {
        return Err(HailoRtError::InvalidMetadata(
            "YOLO HEF must expose at least one output vstream",
        ));
    }
    Ok(YoloModelMetadata { inputs, outputs })
}

pub(crate) fn validate_input_len(
    metadata: &YoloModelMetadata,
    input_len: usize,
) -> HailoRtResult<()> {
    let expected = metadata
        .inputs
        .first()
        .ok_or(HailoRtError::InvalidMetadata("missing YOLO input"))?
        .frame_size;
    if input_len != expected {
        return Err(HailoRtError::InvalidMetadata("input buffer size mismatch"));
    }
    Ok(())
}

pub(crate) fn run_yolo_once(hef_path: &str, input: &[u8]) -> HailoRtResult<YoloInferenceResult> {
    let mut yolo = ShimYolo::create(hef_path)?;
    let (inputs, outputs) = yolo.metadata()?;
    let metadata = YoloModelMetadata { inputs, outputs };
    validate_input_len(&metadata, input.len())?;

    let mut output_storage: Vec<Vec<u8>> = metadata
        .outputs
        .iter()
        .map(|info| vec![0u8; info.frame_size])
        .collect();
    let mut output_views: Vec<_> = metadata
        .outputs
        .iter()
        .zip(output_storage.iter_mut())
        .map(|(info, data)| (info.name.as_str(), data.as_mut_slice()))
        .collect();
    yolo.run(input, &mut output_views, 10_000)?;

    Ok(YoloInferenceResult {
        outputs: metadata
            .outputs
            .into_iter()
            .zip(output_storage)
            .map(|(info, data)| YoloOutputBuffer {
                name: info.name.clone(),
                data,
                info,
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yolo_input_rejects_wrong_size() {
        let meta = YoloModelMetadata::single_input_for_test(1_228_800);
        let err = validate_input_len(&meta, 3).unwrap_err();
        assert!(err.to_string().contains("input buffer size"));
    }

    #[test]
    #[ignore = "requires /dev/hailo0 and HAILO_YOLO_HEF"]
    fn smoke_yolo_hef_metadata_loads() {
        let hef = std::env::var("HAILO_YOLO_HEF").expect("HAILO_YOLO_HEF");
        let metadata = load_yolo_metadata(&hef).expect("metadata");
        assert_eq!(metadata.inputs.len(), 1);
        assert!(!metadata.outputs.is_empty());
    }

    #[test]
    #[ignore = "requires /dev/hailo0 and HAILO_YOLO_HEF"]
    fn smoke_yolo_one_zero_frame_runs() {
        let hef = std::env::var("HAILO_YOLO_HEF").expect("HAILO_YOLO_HEF");
        let metadata = load_yolo_metadata(&hef).expect("metadata");
        let input_len = metadata.inputs[0].frame_size;
        let result = run_yolo_once(&hef, &vec![0u8; input_len]).expect("one-frame inference");
        assert_eq!(result.outputs.len(), metadata.outputs.len());
        for output in result.outputs {
            assert_eq!(output.data.len(), output.info.frame_size);
        }
    }
}
