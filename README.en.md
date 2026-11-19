# yu-hailo-infer
English | [日本語](README.md)

Rust native inference microservice for Hailo-10H. Exposes CLIP embeddings, WD-Tagger tag inference, LLM generation, VLM text generation, and YOLO object detection via HTTP.

Release history: [CHANGELOG.en.md](CHANGELOG.en.md). Unofficial — not affiliated with Hailo.

This is the service itself, not a `-sys` binding crate. It targets the Hailo-10H and reaches HailoRT through the C++ `InferModel` API via a small in-tree shim, because the C `VStreams` path returns `HAILO_NOT_IMPLEMENTED` on that device. If you want raw C-API bindings — for the Hailo-8 / AI HAT+ on Raspberry Pi, for instance — see [`hailort-sys`](https://crates.io/crates/hailort-sys) instead; it is an unrelated project and sits a layer below this one.

## Supported Features

| Feature | Endpoint | Status |
|---|---|---|
| CLIP image embedding | `POST /v1/infer/clip-image` | Supported |
| EmbeddingGemma 2 text embedding | `POST /v1/infer/text-embed` | Experimental host FP32, 256 dimensions |
| CLIP text embedding | `POST /v1/infer/clip-text` | Supported |
| WD-Tagger tag inference | `POST /v1/infer/wd` | ONNX supported; HEF awaits device acceptance |
| LLM text generation | `POST /v1/infer/llm/generate` (+`/stream`) | Supported |
| LLM tokenize | `POST /v1/infer/llm/tokenize` | Supported |
| VLM text generation | `POST /v1/infer/vlm/generate` (+`/stream`) | Supported (text-only; image-attached chat not supported) |
| YOLO object detection | `POST /v1/infer/yolo/detect` | Supported (NMS/decoding included, v0.2+) |
| Speech-to-text transcription | `POST /v1/infer/speech2text/transcribe` (base64 WAV input, transcribe/translate supported), `POST /v1/infer/speech2text/tokenize` | Supported |

## Supported Devices

- **Hailo-10H**: Existing features verified on hardware; WD HEF inference remains device-unverified until acceptance
- Hailo-8 / Hailo-8L and other models: **Not verified** (may work at HailoRT SDK level, but no support guaranteed)

## Dependencies

- Rust: **1.88** or newer. `[workspace.package]` declares `rust-version = "1.88"` and every crate inherits it. The number was measured, not assumed (2026-09-03): the workspace builds on 1.88.0 and cargo refuses 1.85.0. The floor comes from dependencies rather than from this code — cargo reports that `ort`/`ort-sys` 2.0.0-rc.12 require `rustc 1.88` — so it moves when they do. Re-measure with `cargo build --workspace` under an older toolchain before lowering it.
- HailoRT SDK (version to be documented) — requires `hailort` shared library and headers
- `ort` (ONNX Runtime bindings) — used for CLIP text encoder and WD-Tagger inference
- `tokenizers` (Apache-2.0) — used for BPE tokenization of CLIP/LLM
- This repository provides **inference only**. Vector search, index construction (usearch, etc.), and database persistence are out of scope (responsibility of the consuming application)

## Quick Start

Enable existing Cargo features `cuda`, `rocm`, `directml`, `coreml` or `openvino` for GPU providers; the default build is CPU-only. For example: `cargo build --release -p yu-hailo-infer --features cuda`. ROCm, DirectML and OpenVINO require a matching ONNX Runtime via `ORT_DYLIB_PATH`. EmbeddingGemma 2 uses FP32 CUDA (TF32 disabled), ROCm or DirectML; EG2 CoreML/OpenVINO use CPU until precision is validated. Existing CLIP provider support is preserved.

`/healthz` reports requested and registered providers under `text_embed.onnx_execution`. Registration is unknown before loading or while the inference lock is held, and does not guarantee every node executes on GPU. Hardware GPU numerical parity and throughput remain unverified.

```bash
cargo build --release -p yu-hailo-infer

# auth_token / scan_roots / instance_id are not CLI arguments, but passed to stdin at startup
# as JSON ("startup contract"). --port defaults to 18771; --wd-cache-dir is required.
echo '{"instance_id":"local-dev","scan_roots":["/data/images"],"auth_token":"<token>"}' \
  | ./target/release/yu-hailo-infer --port 8100 --wd-cache-dir /var/cache/yu-infer-wd
```

```bash
curl -X POST http://127.0.0.1:8100/v1/infer/clip-text \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"text": "a photo of a cat"}'
```

## WD-Tagger HEF backend (device-unverified)

`POST /v1/infer/wd` accepts `backend` (default `"onnx"`, or `"hef"`), `hef_path`, `hef_sha256`, and `timeout_ms`, alongside the existing `path`, `model_id`, thresholds, and optional `profile` / `model_subdir`. HEF requires `hef_path` and `hef_sha256` (exactly 64 lowercase hexadecimal characters); the sidecar has no default HEF path. Missing, malformed, or mismatched hashes return 503 `hef_invalid`. ONNX ignores `hef_sha256` entirely and never logs it. `timeout_ms` applies only to HEF and defaults to, and is capped at, 120,000 ms. The sidecar does not automatically fall back to ONNX. Success has the form `{"ok":true,"backend_applied":"hef","profile_applied":false,"data":{...}}`. ONNX success returns `backend_applied:"onnx"`; supplying a profile returns `profile_applied:true`.

The additive `GET /healthz` field `wd_hef` is `true` only in real HailoRT builds on Linux with the HEF route compiled. Stub builds always return `false`; `hailo_stub` keeps its existing meaning. Clients must treat an absent `wd_hef` in older responses as unsupported. Health does not open a HEF, access WD caches, or create a VDevice. The field advertises compiled capability, not model presence or device readiness.

This supports ViT UINT8 HEFs: one 448×448×3 BGR/NHWC input (602,112 bytes) and one UINT8 output with exactly the vocabulary length, manually dequantized as `(q - zero_point) * scale`. Preprocessing shares ONNX's white padding and Lanczos3 resize; incompatible profiles are rejected. HEF still requires `model.onnx` and `selected_tags.csv` in the model directory under `--wd-cache-dir`. HEF files are opened with `O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC`, checked with fstat on that same fd for regular-file type and a 256 MiB maximum, then read with a bound and passed as a verified buffer to MemoryView. Terminal symlinks and special files are rejected.

One process-static VDevice is shared across model handles; Hailo work is serialized on the common device thread. A single WD HEF cache slot is resident. Each request opens/fstats the HEF; a change in dev, inode, size, mtime (including nanoseconds), or requested digest releases the old handle before rereading. SHA256 of the bounded bytes from that same checked fd must match all 32 requested bytes before HailoRT open, metadata, or configure. Cache hits require both the file key and verified digest to match.

`HAILO_COMMUNICATION_CLOSED` marks the process-wide Hailo session lost. New Hailo requests return 503 without calling HailoRT (WD HEF: `{"error":"device_error"}`); queued work is rejected too. CPU ONNX routes remain available while draining. `/healthz` always includes `hailo_session_lost`: normally `false`, then `true` with `ok:false`. The sidecar starts graceful shutdown after a short delay, drains HTTP responses, and exits with **70 (`EX_SOFTWARE`)** so yu-server can respawn it. A fresh process is required to recover. Embedders should use the library's `run()` to retain this exit behavior.

HEF route errors contain only `{"error":"<code>"}`, without paths, detail, or message.

| HTTP | error | Condition |
|---|---|---|
| 503 | `hef_invalid` | Missing `hef_path` or missing/malformed/mismatched `hef_sha256` with `backend:"hef"` |
| 403 | `path outside allowed roots` | Image outside scan roots; identical to ONNX |
| 422 | `image_invalid` | Image canonicalization, decode, or preprocessing failure |
| 503 | `hef_not_found` | Missing or unreadable HEF |
| 503 | `hef_invalid` | Invalid HEF type, size, metadata, or profile |
| 503 | `device_busy` | Device queue or preprocessing budget full |
| 503 | `device_error` | HailoRT failure, session lost, degraded state, or stopped device thread |
| 503 | `hef_timeout` | Sidecar response deadline expired |
| 503 | `model_not_downloaded` | Missing vocabulary or model.onnx |

Existing authentication errors return 401; JSON extraction/type failures use axum responses. These changes still require device acceptance testing. This revision does not establish accuracy, fps, concurrent residency behavior, or memory usage.

## Instructing LLMs

This repository provides `docs-index.yaml` (a pointer index to code files) for AI/LLM coding agents. Instead of long-form documentation for humans, agents are expected to follow instructions like this:

There is also `docs/ai-reference.yaml`, an AI-facing reference covering configuration, endpoint specs, and known quirks (e.g. the auth-failure response shape differing from other error responses, the single-device serialization behavior). Point an agent there first for requests like "explain how to use/configure this service."

> "Read `docs-index.yaml`, then read the code at the `path` listed in the relevant entry, and explain <your question>."

Examples:
> "Check `docs-index.yaml`, then explain how the YOLO detection NMS/decoding implementation works (`yu-hailo-infer-core/src/yolo_postprocess.rs`)"
> "Look at `docs-index.yaml`, then tell me where the CLIP image embedding dequantize processing is"

## License

MIT

## EmbeddingGemma 2

`POST /v1/infer/text-embed` accepts `{"text":"桜の花","task":"query","dim":256}` with the existing bearer authentication. For documents use `task:"document"` and an optional `title`; the server adds the search prefix. The success envelope contains `data.vector`, `dim`, `model`, `revision`, `precision`, and `preprocessing`.

Only pinned FP32 revision `daa72c51243991dfcaf9f9137d2c573d8f7790c0` is supported. Place `tokenizer.json`, `onnx/model.onnx`, and `onnx/model.onnx_data` under `~/.cache/yu_ai_manager/embeddinggemma2/<revision>` or an absolute `model_dir`. Artifact sizes and SHA-256 must match the manifest in `embeddinggemma2.rs`; no automatic download occurs here. Loading uses a private verified snapshot (~1.1 GB additional disk space), removed on graceful cache destruction; forced termination may leave residue. Input text/title is limited to16 KiB combined, then8192 tokens. One active request and four FIFO waiters are allowed; overflow returns503. Pi5, GPU behavior, and production throughput are unverified.
