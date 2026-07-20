use std::ffi::CString;
use std::ptr;

use super::{check_status, ffi, HailoRtError, HailoRtResult, VStreamInfo};

pub(crate) struct VDevice {
    raw: ffi::HailoVDevice,
}

impl VDevice {
    pub(crate) fn new_default() -> HailoRtResult<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: Passing NULL params requests HailoRT defaults; raw out pointer is valid.
        let status = unsafe { ffi::hailo_create_vdevice(ptr::null_mut(), &mut raw) };
        check_status("create_vdevice", status)?;
        if raw.is_null() {
            return Err(HailoRtError::InvalidMetadata(
                "create_vdevice returned null",
            ));
        }
        Ok(Self { raw })
    }

    pub(crate) fn configure<'a>(
        &'a self,
        hef: &'a Hef,
    ) -> HailoRtResult<ConfiguredNetworkGroup<'a>> {
        let mut groups = [ptr::null_mut(); ffi::HAILO_MAX_NETWORK_GROUPS];
        let mut count = groups.len();
        // SAFETY: Handles are valid RAII-owned HailoRT handles; params NULL requests defaults.
        let status = unsafe {
            ffi::hailo_configure_vdevice(
                self.raw,
                hef.raw,
                ptr::null_mut(),
                groups.as_mut_ptr(),
                &mut count,
            )
        };
        check_status("configure_vdevice", status)?;
        if count == 0 || groups[0].is_null() {
            return Err(HailoRtError::InvalidMetadata(
                "configure_vdevice returned no network groups",
            ));
        }
        Ok(ConfiguredNetworkGroup {
            raw: groups[0],
            _device: self,
        })
    }
}

impl Drop for VDevice {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: raw is owned by this RAII wrapper and released once here.
            let _ = unsafe { ffi::hailo_release_vdevice(self.raw) };
        }
    }
}

pub(crate) struct Hef {
    pub(crate) raw: ffi::HailoHef,
}

impl Hef {
    pub(crate) fn from_path(path: &str) -> HailoRtResult<Self> {
        let path = CString::new(path)?;
        let mut raw = ptr::null_mut();
        // SAFETY: path is a nul-terminated C string; raw out pointer is valid.
        let status = unsafe { ffi::hailo_create_hef_file(&mut raw, path.as_ptr()) };
        check_status("create_hef_file", status)?;
        if raw.is_null() {
            return Err(HailoRtError::InvalidMetadata(
                "create_hef_file returned null",
            ));
        }
        Ok(Self { raw })
    }

    pub(crate) fn vstream_infos(&self) -> HailoRtResult<Vec<VStreamInfo>> {
        let mut raw_infos = vec![ffi::HailoVStreamInfo::default(); ffi::HAILO_MAX_STREAMS_COUNT];
        let mut actual = raw_infos.len();
        // SAFETY: raw_infos has capacity for HailoRT's documented maximum stream count.
        let status = unsafe {
            ffi::hailo_hef_get_all_vstream_infos(
                self.raw,
                ptr::null(),
                raw_infos.as_mut_ptr(),
                &mut actual,
            )
        };
        check_status("hef_get_all_vstream_infos", status)?;
        if actual == 0 {
            return Err(HailoRtError::InvalidMetadata("HEF has no vstreams"));
        }
        raw_infos.truncate(actual);
        raw_infos
            .into_iter()
            .map(VStreamInfo::from_raw)
            .collect::<HailoRtResult<Vec<_>>>()
    }
}

impl Drop for Hef {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: raw is owned by this RAII wrapper and released once here.
            let _ = unsafe { ffi::hailo_release_hef(self.raw) };
        }
    }
}

pub(crate) struct ConfiguredNetworkGroup<'a> {
    pub(crate) raw: ffi::HailoConfiguredNetworkGroup,
    _device: &'a VDevice,
}

impl ConfiguredNetworkGroup<'_> {
    pub(crate) fn activate(&self) -> HailoRtResult<ActivatedNetworkGroup<'_>> {
        let mut raw = ptr::null_mut();
        // SAFETY: raw network group is valid while VDevice is alive; NULL params requests defaults.
        let status =
            unsafe { ffi::hailo_activate_network_group(self.raw, ptr::null_mut(), &mut raw) };
        check_status("activate_network_group", status)?;
        if raw.is_null() {
            return Err(HailoRtError::InvalidMetadata(
                "activate_network_group returned null",
            ));
        }
        Ok(ActivatedNetworkGroup { raw, _group: self })
    }
}

impl Drop for ConfiguredNetworkGroup<'_> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: Shutdown is optional and cancels ongoing operations before VDevice drop.
            let _ = unsafe { ffi::hailo_shutdown_network_group(self.raw) };
        }
    }
}

pub(crate) struct ActivatedNetworkGroup<'a> {
    raw: ffi::HailoActivatedNetworkGroup,
    _group: &'a ConfiguredNetworkGroup<'a>,
}

impl Drop for ActivatedNetworkGroup<'_> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: raw is owned by this activation wrapper and released once here.
            let _ = unsafe { ffi::hailo_deactivate_network_group(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hef_rejects_paths_with_nul_bytes() {
        let err = match Hef::from_path("bad\0path") {
            Ok(_) => panic!("expected nul byte error"),
            Err(err) => err,
        };
        assert!(matches!(err, HailoRtError::Nul(_)));
    }
}
