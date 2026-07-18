use std::io::Read;
use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct RawStartupContract {
    instance_id: String,
    scan_roots: Vec<String>,
    auth_token: String,
}

#[derive(Debug, Clone)]
pub struct StartupContract {
    pub instance_id: String,
    pub scan_roots: Vec<PathBuf>,
    pub auth_token: String,
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_contract() {
        let input = r#"{"instance_id": "inst-1", "scan_roots": ["/data/images", "/data/more"], "auth_token": "abc123"}"#;
        let contract = read_startup_contract(input.as_bytes()).unwrap();
        assert_eq!(contract.instance_id, "inst-1");
        assert_eq!(
            contract.scan_roots,
            vec![PathBuf::from("/data/images"), PathBuf::from("/data/more")]
        );
        assert_eq!(contract.auth_token, "abc123");
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
