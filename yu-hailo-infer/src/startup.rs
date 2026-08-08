use std::io::Read;
use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct RawStartupContract {
    instance_id: String,
    scan_roots: Vec<String>,
    auth_token: String,
    vdevice_group_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StartupContract {
    pub instance_id: String,
    pub scan_roots: Vec<PathBuf>,
    pub auth_token: String,
    pub vdevice_group_id: String,
}

#[derive(Debug, thiserror::Error)]
pub enum StartupError {
    #[error("failed to read stdin: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse startup contract JSON: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("auth_token must not be empty")]
    EmptyToken,
}

pub fn read_startup_contract(mut reader: impl Read) -> Result<StartupContract, StartupError> {
    let mut buf = String::new();
    reader.read_to_string(&mut buf)?;
    let raw: RawStartupContract = serde_json::from_str(&buf)?;
    if raw.auth_token.is_empty() {
        return Err(StartupError::EmptyToken);
    }
    Ok(StartupContract {
        instance_id: raw.instance_id,
        scan_roots: raw.scan_roots.into_iter().map(PathBuf::from).collect(),
        auth_token: raw.auth_token,
        vdevice_group_id: raw
            .vdevice_group_id
            .or_else(|| std::env::var("HAILO_VDEVICE_GROUP_ID").ok())
            .unwrap_or_else(|| "YU_SHARED".to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn read_with_group_env(input: &str, value: Option<&str>) -> StartupContract {
        let _guard = ENV_LOCK.lock().unwrap();
        let previous = std::env::var_os("HAILO_VDEVICE_GROUP_ID");
        match value {
            Some(value) => std::env::set_var("HAILO_VDEVICE_GROUP_ID", value),
            None => std::env::remove_var("HAILO_VDEVICE_GROUP_ID"),
        }
        let contract = read_startup_contract(input.as_bytes()).unwrap();
        match previous {
            Some(value) => std::env::set_var("HAILO_VDEVICE_GROUP_ID", value),
            None => std::env::remove_var("HAILO_VDEVICE_GROUP_ID"),
        }
        contract
    }

    #[test]
    fn parses_valid_contract() {
        let input = r#"{"instance_id": "inst-1", "scan_roots": ["/data/images", "/data/more"], "auth_token": "abc123", "vdevice_group_id": "contract-group"}"#;
        let contract = read_startup_contract(input.as_bytes()).unwrap();
        assert_eq!(contract.instance_id, "inst-1");
        assert_eq!(
            contract.scan_roots,
            vec![PathBuf::from("/data/images"), PathBuf::from("/data/more")]
        );
        assert_eq!(contract.auth_token, "abc123");
        assert_eq!(contract.vdevice_group_id, "contract-group");
    }

    #[test]
    fn uses_env_group_id_when_contract_field_is_absent() {
        let input = r#"{"instance_id": "inst-1", "scan_roots": [], "auth_token": "abc123"}"#;
        let contract = read_with_group_env(input, Some("env-group"));
        assert_eq!(contract.vdevice_group_id, "env-group");
    }

    #[test]
    fn uses_shared_default_when_group_id_sources_are_absent() {
        let input = r#"{"instance_id": "inst-1", "scan_roots": [], "auth_token": "abc123"}"#;
        let contract = read_with_group_env(input, None);
        assert_eq!(contract.vdevice_group_id, "YU_SHARED");
    }

    #[test]
    fn rejects_empty_auth_token() {
        let input = r#"{"instance_id": "inst-1", "scan_roots": [], "auth_token": ""}"#;
        assert!(matches!(
            read_startup_contract(input.as_bytes()),
            Err(StartupError::EmptyToken)
        ));
    }

    #[test]
    fn rejects_malformed_json() {
        let input = "not json";
        assert!(matches!(
            read_startup_contract(input.as_bytes()),
            Err(StartupError::Parse(_))
        ));
    }
}
