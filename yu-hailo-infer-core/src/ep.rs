//! Execution-provider selection shared by every ONNX session in this crate.
//!
//! Which providers are compiled in is a build-time decision made by cargo
//! features (`cuda`, `rocm`, `coreml`, `directml`, `openvino`). Each one is
//! registered with ORT's default non-fatal behaviour, so a provider whose
//! hardware, driver or dynamic library is absent at runtime is skipped with a
//! warning instead of failing the session. CPU is always appended last, so
//! every build -- including one with no GPU feature at all -- still runs.

use std::path::Path;

use ort::{ep::ExecutionProviderDispatch, session::Session};

use crate::InferError;

/// Build an ORT session that tries every compiled-in GPU provider in order and
/// falls back to CPU.
pub(crate) fn build_session(model_path: &Path) -> Result<Session, InferError> {
    let builder = Session::builder()?;
    let providers = gpu_providers();
    let mut builder = if providers.is_empty() {
        // No GPU feature compiled in: ORT's own default list is CPU-only, and
        // leaving it alone keeps this identical to the pre-feature behaviour.
        builder
    } else {
        let mut providers = providers;
        // `with_execution_providers` replaces the default list rather than
        // prepending to it, so CPU has to be named explicitly -- otherwise a
        // build whose GPU provider fails to register has nothing left to run.
        providers.push(ort::ep::CPU::default().build());
        // `with_execution_providers` reports a *recoverable* error, which keeps
        // the builder inside the error value and so has a different type from
        // the rest of ort's API. Drop the recovery payload to get back to the
        // plain `ort::Error` that InferError knows how to absorb.
        builder
            .with_execution_providers(providers)
            .map_err(ort::Error::from)?
    };
    Ok(builder.commit_from_file(model_path)?)
}

/// The GPU providers this binary was compiled with, in registration order.
///
/// Order is priority: ORT tries each in turn and uses the first that accepts a
/// given node. The features are mutually exclusive in practice (each build
/// targets one accelerator) but nothing here depends on that.
fn gpu_providers() -> Vec<ExecutionProviderDispatch> {
    #[allow(unused_mut)]
    let mut providers: Vec<ExecutionProviderDispatch> = Vec::new();

    #[cfg(feature = "cuda")]
    {
        tracing::info!("Requesting CUDA execution provider");
        providers.push(ort::ep::CUDA::default().build());
    }
    #[cfg(feature = "rocm")]
    {
        tracing::info!("Requesting ROCm execution provider");
        providers.push(ort::ep::ROCm::default().build());
    }
    #[cfg(feature = "coreml")]
    {
        tracing::info!("Requesting CoreML execution provider");
        providers.push(ort::ep::CoreML::default().build());
    }
    #[cfg(feature = "directml")]
    {
        tracing::info!("Requesting DirectML execution provider");
        providers.push(ort::ep::DirectML::default().build());
    }
    #[cfg(feature = "openvino")]
    {
        tracing::info!("Requesting OpenVINO execution provider");
        providers.push(ort::ep::OpenVINO::default().build());
    }

    providers
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The provider list must match the features this build enabled: a `#[cfg]`
    /// that stops matching its feature (a rename, a typo, a feature dropped
    /// from Cargo.toml) silently returns the binary to CPU-only inference,
    /// which is exactly the failure this module exists to end. Counting the
    /// features here fails the build instead of degrading it quietly.
    #[test]
    fn provider_count_matches_enabled_features() {
        let expected = usize::from(cfg!(feature = "cuda"))
            + usize::from(cfg!(feature = "rocm"))
            + usize::from(cfg!(feature = "coreml"))
            + usize::from(cfg!(feature = "directml"))
            + usize::from(cfg!(feature = "openvino"));
        assert_eq!(gpu_providers().len(), expected);
    }
}
