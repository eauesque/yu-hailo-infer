mod hailort;
mod router;
mod startup;

use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value_t = 18771)]
    port: u16,
    #[arg(long)]
    wd_cache_dir: PathBuf,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let contract = startup::read_startup_contract(std::io::stdin())
        .expect("failed to read startup contract from stdin");
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
