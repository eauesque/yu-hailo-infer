//! Checked WD HEF loading and the device thread's single resident WD model.
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{self, Read};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;
use std::sync::Mutex;

use super::wd::{ConfiguredWd, OpenedWd, WdError};
use super::VStreamInfo;

const LIMIT: u64 = 268_435_456;
#[cfg(target_os = "linux")]
pub(crate) const OPEN_FLAGS: i32 =
    libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NOCTTY;

#[derive(Debug, thiserror::Error)]
pub(crate) enum WdHefError {
    #[error("WD HEF unavailable")]
    NotFound,
    #[error("WD HEF invalid")]
    Invalid,
    #[error(transparent)]
    Runtime(#[from] WdError),
}

#[derive(Debug, PartialEq, Eq)]
struct FileKey {
    dev: u64,
    ino: u64,
    size: u64,
    mtime: i64,
    mtime_nsec: i64,
}

struct CheckedHef {
    file: File,
    key: FileKey,
}

fn open_error(error: io::Error) -> WdHefError {
    match error.raw_os_error() {
        // Linux ENXIO/ENODEV (special files) and ELOOP (terminal symlink).
        Some(libc::ENXIO | libc::ENODEV | libc::ELOOP) => WdHefError::Invalid,
        _ if error.kind() == io::ErrorKind::InvalidInput => WdHefError::Invalid,
        _ => WdHefError::NotFound,
    }
}

impl CheckedHef {
    fn open(path: &Path) -> Result<Self, WdHefError> {
        let before = std::fs::symlink_metadata(path).map_err(open_error)?;
        if !before.is_file() {
            return Err(WdHefError::Invalid);
        }
        // read(true) supplies O_RDONLY; metadata() is fstat on this same fd.
        #[cfg(test)]
        tests::before_open();
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(OPEN_FLAGS)
            .open(path)
            .map_err(open_error)?;
        let stat = file.metadata().map_err(|_| WdHefError::NotFound)?;
        if !stat.is_file()
            || stat.len() == 0
            || stat.len() > LIMIT
            || stat.dev() != before.dev()
            || stat.ino() != before.ino()
        {
            return Err(WdHefError::Invalid);
        }
        Ok(Self {
            file,
            key: FileKey {
                dev: stat.dev(),
                ino: stat.ino(),
                size: stat.len(),
                mtime: stat.mtime(),
                mtime_nsec: stat.mtime_nsec(),
            },
        })
    }
}

fn read_bounded(reader: impl Read, limit: u64, size: u64) -> Result<Vec<u8>, WdHefError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size.min(limit) as usize)
        .map_err(|_| WdHefError::Invalid)?;
    let mut reader = reader.take(limit + 1);
    let mut chunk = [0; 64 * 1024];
    loop {
        let count = match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(WdHefError::NotFound),
        };
        if bytes.len() as u64 + count as u64 > limit {
            return Err(WdHefError::Invalid);
        }
        // Exact growth avoids doubling a near-limit allocation after fstat.
        bytes
            .try_reserve_exact(count)
            .map_err(|_| WdHefError::Invalid)?;
        bytes.extend_from_slice(&chunk[..count]);
    }
    Ok(bytes)
}

struct CachedWd {
    key: FileKey,
    digest: [u8; 32],
    model: ConfiguredWd,
    inputs: Vec<VStreamInfo>,
    outputs: Vec<VStreamInfo>,
}

#[derive(Default)]
pub(crate) struct WdHefCache {
    slot: Mutex<Option<CachedWd>>,
}

impl WdHefCache {
    pub(crate) fn with_model<T>(
        &self,
        path: &Path,
        digest: [u8; 32],
        validate: impl FnOnce(&[VStreamInfo], &[VStreamInfo]) -> Result<(), WdHefError>,
        task: impl FnOnce(&mut ConfiguredWd) -> Result<T, WdHefError>,
    ) -> Result<T, WdHefError> {
        self.with_metadata(path, digest, validate, |model, _, _| task(model))
    }

    pub(crate) fn with_metadata<T>(
        &self,
        path: &Path,
        digest: [u8; 32],
        validate: impl FnOnce(&[VStreamInfo], &[VStreamInfo]) -> Result<(), WdHefError>,
        task: impl FnOnce(&mut ConfiguredWd, &[VStreamInfo], &[VStreamInfo]) -> Result<T, WdHefError>,
    ) -> Result<T, WdHefError> {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        let checked = match CheckedHef::open(path) {
            Ok(checked) => checked,
            Err(error) => {
                *slot = None;
                return Err(error);
            }
        };
        if slot
            .as_ref()
            .is_none_or(|cached| cached.key != checked.key || cached.digest != digest)
        {
            *slot = None; // Release CMA before reading or opening the replacement.
            #[cfg(test)]
            tests::before_read();
            let bytes = read_bounded(checked.file, LIMIT, checked.key.size)?;
            let verified: [u8; 32] = Sha256::digest(&bytes).into();
            if verified != digest {
                return Err(WdHefError::Invalid);
            }
            let opened = OpenedWd::open(&bytes);
            drop(bytes); // HailoRT copies during open, including its failure path.
            let opened = opened?;
            let (inputs, outputs) = opened.metadata()?;
            validate(&inputs, &outputs)?;
            let model = opened.configure()?;
            *slot = Some(CachedWd {
                key: checked.key,
                digest: verified,
                model,
                inputs,
                outputs,
            });
        }
        let cached = slot.as_mut().unwrap();
        let result = task(&mut cached.model, &cached.inputs, &cached.outputs);
        if matches!(&result, Err(WdHefError::Runtime(WdError::Timeout))) {
            *slot = None;
        }
        result
    }

    pub(crate) fn evict(&self) {
        *self.slot.lock().unwrap_or_else(|e| e.into_inner()) = None;
        self.slot.clear_poison();
    }
}

#[cfg(test)]
#[path = "wd_hef_tests.rs"]
pub(crate) mod tests;
