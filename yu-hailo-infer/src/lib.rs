//! yu-infer sidecar library.
//!
//! The whole service lives here so that embedders can provide their own
//! `yu-infer` binary without duplicating the implementation. The bundled
//! binary (`src/main.rs`) is a thin shim over [`run`], and downstream
//! consumers are expected to ship an equivalent shim rather than
//! re-implementing startup.

pub mod hailort;
pub mod router;
pub mod speech2text_route;
pub mod startup;

use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

#[derive(Parser)]
pub struct Cli {
    #[arg(long, default_value_t = 18771)]
    pub port: u16,
    #[arg(long)]
    pub wd_cache_dir: PathBuf,
}

/// Parses the CLI, reads the startup contract from stdin, and serves until
/// shutdown. Initialising tracing is the caller's responsibility so that an
/// embedder can install its own subscriber first.
pub async fn run() {
    let cli = Cli::parse();

    let contract = startup::read_startup_contract(std::io::stdin())
        .expect("failed to read startup contract from stdin");
    hailort::set_vdevice_group_id(&contract.vdevice_group_id)
        .expect("failed to set HailoRT VDevice group ID");
    tracing::info!(
        scan_root_count = contract.scan_roots.len(),
        "yu-infer startup contract received"
    );

    let state = router::AppState {
        started_at: std::time::Instant::now(),
        instance_id: contract.instance_id,
        scan_roots: Arc::new(RwLock::new(contract.scan_roots)),
        auth_token: contract.auth_token,
        wd_cache_dir: cli.wd_cache_dir,
        wd_infer: Arc::new(RwLock::new(HashMap::new())),
        clip_text: Arc::new(RwLock::new(HashMap::new())),
    };
    let app = router::build_router(state);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], cli.port));
    tracing::info!("yu-infer listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind yu-infer listener");
    axum::serve(listener, app)
        .await
        .expect("yu-infer server error");
}
