//! Thin shim over the `yu_infer` library. Embedders that need the binary
//! next to their own executable should vendor a copy of this file rather
//! than duplicating the service implementation.

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    yu_infer::run().await;
}
