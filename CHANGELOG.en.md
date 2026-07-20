# Changelog

English | [日本語](CHANGELOG.md)

All notable changes to this project are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-07-20

First public release. Extracted from the [yu_ai_manager](https://github.com/eauesque/yu_ai_manager)
project, where this inference stack was originally developed, and published as a
standalone service so that other projects can use the Hailo-10H NPU without
depending on that application.

### Added

- **HTTP inference sidecar for the Hailo-10H NPU** (`yu-hailo-infer`), covering:
  - CLIP image embedding — `POST /v1/infer/clip-image`
  - CLIP text embedding — `POST /v1/infer/clip-text`
  - WD-Tagger tag inference — `POST /v1/infer/wd`
  - LLM text generation and tokenization — `POST /v1/infer/llm/generate` (+ `/stream`), `POST /v1/infer/llm/tokenize`
  - VLM text generation — `POST /v1/infer/vlm/generate` (+ `/stream`); text-only, image-attached chat is out of scope
  - YOLO object detection — `POST /v1/infer/yolo/detect`, returning final detections with NMS and decoding applied
  - Speech-to-text transcription and tokenization — `POST /v1/infer/speech2text/transcribe` (base64 WAV, transcribe and translate modes), `POST /v1/infer/speech2text/tokenize`
- **Shared inference engines** (`yu-hailo-infer-core`): ONNX Runtime backed WD-Tagger
  and CLIP text encoders, plus YOLO post-processing (dequantization, grid/stride
  relative decoding, embedded-NMS output parsing, and NMS).
- **Bearer token authentication** (`yu-hailo-auth`) with constant-time comparison.
  The token is delivered over the startup contract on stdin rather than as a CLI
  argument, so it is not exposed through `/proc/<pid>/cmdline`.
- **Scan-root containment**: filesystem paths supplied by callers are rejected
  unless they resolve inside the roots declared in the startup contract.
- **Builds without the HailoRT SDK**: when the SDK headers are absent the build
  falls back to a stub shim, so the crate compiles (and documents) on machines
  with no Hailo hardware.
- **AI-agent oriented documentation**: `docs-index.yaml` (a pointer index into the
  code) and `docs/ai-reference.yaml` (configuration, endpoint specs, and known
  quirks), intended to be read by coding agents instead of long-form prose.

### Notes

- **Unofficial.** This project is not affiliated with, endorsed by, or supported by
  Hailo. It links against the HailoRT SDK but vendors none of it.
- **Device support**: verified on Hailo-10H hardware. Hailo-8 / Hailo-8L and other
  models are **unverified** — they may work at the HailoRT SDK level, but no support
  is implied.
- **Inference only.** Vector search, index construction, and database persistence
  are deliberately out of scope and remain the consuming application's concern.
- Image-attached VLM chat and web-search RAG integration are out of scope: both are
  application-level concerns rather than general-purpose inference.

[0.1.0]: https://github.com/eauesque/yu-hailo-infer/releases/tag/v0.1.0
