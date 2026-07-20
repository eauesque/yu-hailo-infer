# yu-hailo-infer
English | [日本語](README.md)

Rust native inference microservice for Hailo-10H. Exposes CLIP embeddings, WD-Tagger tag inference, LLM generation, VLM text generation, and YOLO object detection via HTTP.

Release history: [CHANGELOG.en.md](CHANGELOG.en.md). Unofficial — not affiliated with Hailo.

## Supported Features

| Feature | Endpoint | Status |
|---|---|---|
| CLIP image embedding | `POST /v1/infer/clip-image` | Supported |
| CLIP text embedding | `POST /v1/infer/clip-text` | Supported |
| WD-Tagger tag inference | `POST /v1/infer/wd` | Supported |
| LLM text generation | `POST /v1/infer/llm/generate` (+`/stream`) | Supported |
| LLM tokenize | `POST /v1/infer/llm/tokenize` | Supported |
| VLM text generation | `POST /v1/infer/vlm/generate` (+`/stream`) | Supported (text-only; image-attached chat not supported) |
| YOLO object detection | `POST /v1/infer/yolo/detect` | Supported (NMS/decoding included, v0.2+) |
| Speech-to-text transcription | `POST /v1/infer/speech2text/transcribe` (base64 WAV input, transcribe/translate supported), `POST /v1/infer/speech2text/tokenize` | Supported |

## Supported Devices

- **Hailo-10H**: Verified on hardware
- Hailo-8 / Hailo-8L and other models: **Not verified** (may work at HailoRT SDK level, but no support guaranteed)

## Dependencies

- Rust 1.88+ (to be verified; see `rust-version` in Cargo.toml for the actual toolchain version)
- HailoRT SDK (version to be documented) — requires `hailort` shared library and headers
- `ort` (ONNX Runtime bindings) — used for CLIP text encoder and WD-Tagger inference
- `tokenizers` (Apache-2.0) — used for BPE tokenization of CLIP/LLM
- This repository provides **inference only**. Vector search, index construction (usearch, etc.), and database persistence are out of scope (responsibility of the consuming application)

## Quick Start

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

## Instructing LLMs

This repository provides `docs-index.yaml` (a pointer index to code files) for AI/LLM coding agents. Instead of long-form documentation for humans, agents are expected to follow instructions like this:

There is also `docs/ai-reference.yaml`, an AI-facing reference covering configuration, endpoint specs, and known quirks (e.g. the auth-failure response shape differing from other error responses, the single-device serialization behavior). Point an agent there first for requests like "explain how to use/configure this service."

> "Read `docs-index.yaml`, then read the code at the `path` listed in the relevant entry, and explain <your question>."

Examples:
> "Check `docs-index.yaml`, then explain how the YOLO detection NMS/decoding implementation works (`yu-hailo-infer-core/src/yolo_postprocess.rs`)"
> "Look at `docs-index.yaml`, then tell me where the CLIP image embedding dequantize processing is"

## License

MIT
