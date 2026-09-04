use crate::InferError;

pub(crate) const CLIP_EMBEDDING_DIMENSION: usize = 512;

pub(crate) fn validate_and_normalize(vector: &mut [f32]) -> Result<(), InferError> {
    if vector.len() != CLIP_EMBEDDING_DIMENSION {
        return Err(InferError::InvalidModelOutput(format!(
            "CLIP embedding must contain {CLIP_EMBEDDING_DIMENSION} values, got {}",
            vector.len()
        )));
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm >= 1e-12 {
        for value in vector {
            *value /= norm;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_and_normalization_match_clip_zero_norm_guard() {
        let mut vector = vec![0.0; CLIP_EMBEDDING_DIMENSION];
        vector[0] = 3.0;
        vector[1] = 4.0;
        validate_and_normalize(&mut vector).unwrap();
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);

        let mut zero = vec![0.0; CLIP_EMBEDDING_DIMENSION];
        validate_and_normalize(&mut zero).unwrap();
        assert!(zero.iter().all(|&value| value == 0.0));
    }
}
