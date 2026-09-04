# Changelog

English | [日本語](CHANGELOG.md)

All notable changes to this project are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed

- **Measured the MSRV as `1.88` and declared `rust-version` in
  `[workspace.package]`**, with every crate inheriting it via
  `rust-version.workspace = true`. The READMEs previously said only "built and
  tested with 1.96.0, MSRV undetermined" and the crates shipped no
  `rust-version` at all, so a consumer could not tell whether their toolchain
  would work without trying it.
  - The number is measured, not assumed: the workspace builds on 1.88.0 and
    cargo refuses 1.85.0.
  - **The floor comes from dependencies, not from this code**: cargo reports
    that `ort`/`ort-sys` 2.0.0-rc.12 require `rustc 1.88`. It moves when they
    do, so re-measure before lowering it.
  - Verified the declaration is live by injection: raising it to `1.99`
    temporarily makes cargo refuse and name *our* packages
    (`yu-hailo-auth@0.3.1 requires rustc 1.99`) rather than a dependency.
    Without that check, a failure of key inheritance would have left the MSRV
    a promise with nothing enforcing it.

## [0.3.1] - 2026-08-30

### Fixed

- **A stale `/_internal/scan-roots-changed` notification could overwrite newer
  scan roots.** yu-server assigns a monotonic `generation` in config-write order,
  but the receiver never read it and overwrote its roots on every call. yu-server's
  own send lock orders only the sends it starts; one it gave up on (5s timeout)
  can still land afterwards and clobber a newer one. The handler now remembers the
  highest generation it applied and drops anything not strictly newer, answering
  `{"ok":true,"applied":false,"stale":true}`. The generation lives under the same
  `RwLock` as the roots so the compare-and-apply cannot interleave with another
  request. A request carrying no `generation` (a yu-server predating the field) is
  still applied unconditionally and leaves the stored generation alone.

---

## [0.3.0] - 2026-08-16

Verified on Hailo-10H hardware. Wires native tool-call support (HailoRT genai
`LLMGenerator::write(messages, tools)`) into `/v1/infer/llm/generate/stream`.

### Added

- **`tools` field on the LLM streaming generation request.** Accepts an array of
  OpenAI-style tool definitions (`{"name":...,"description":...,"parameters":...}`)
  in the request JSON. Added `tools_json`/`tools_count` arguments through the C
  ABI and Rust binding (`shim.cpp`/`shim.h`/`shim_stub.cpp`/`llm.rs`), forwarding
  to the HailoRT SDK's `write(prompt_json_strings, tools_json_strings)`. Existing
  callers keep working unchanged by passing an empty array (backward compatible).
  Added validation for `MAX_LLM_TOOLS` (64) and a combined byte-size cap (same
  bound as `MAX_PROMPT_BYTES`).
- Verified on hardware (Qwen3-1.7B-Instruct.hef) that passing tool definitions
  makes the model respond in `<tool_call>\n{"name": "...", "arguments": {...}}\n
  </tool_call>` form — Qwen's own native function-calling syntax.

## [0.2.0] - 2026-08-08

Verified on Hailo-10H hardware. The main change is model residency, which is not
a performance optimisation but the only shape in which the service works at all:
HailoRT 5.3.0 does not return CMA when a model is released, so the previous
create-and-drop-per-request design permanently lost roughly 59 MiB per request
and, against a 512 MiB pool, allowed about one request per boot.

### Added

- **A single process-lifetime `VDevice`.** Previously `VDevice::create_shared()`
  was called in four places, once per model kind. The Hailo-10H has exactly one
  physical device, and creating two `VDevice` instances in one process fails with
  `HAILO_OUT_OF_PHYSICAL_DEVICES(74)`. Measured further: **even with the same
  group_id, models created on separate instances fail at `InferModel.run()`.**
  The device is never released — `VDevice.release()` does not reclaim CMA, so
  releasing accomplishes nothing.
- **`vdevice_group_id` in the startup contract.** When absent it falls back to the
  `HAILO_VDEVICE_GROUP_ID` environment variable, then to `"YU_SHARED"`. This lets
  the sidecar share the device with another process using the same group id — for
  example the Python extension in yu_ai_manager. Confirmed on hardware: the
  sidecar's CLIP encoder runs while the Python side holds an LLM.
- **Resident model handles**, keyed by the whole create-argument tuple.
  InferModel-class handles (YOLO, CLIP) may coexist with different HEFs;
  GenAI-class handles (LLM, VLM) are limited to one at a time, and a request for a
  different GenAI HEF returns 409 naming the HEF currently loaded. `Speech2Text`
  is not made resident because it has no `clear_context`.
- **A dedicated device thread.** Model handles are `!Send`, so the global mutex was
  replaced by a thread that owns them. Only closures and results cross the thread
  boundary, so no `unsafe impl Send` was introduced anywhere. Each work item runs
  under `catch_unwind`.

### Fixed

- **Residency would have broken chat on the second turn.** HailoRT accepts a
  `system` role message only while the context is empty, and the caller sends one
  every turn, so reusing a handle fails with `System role messages can only be
  provided on the first prompt`. ⟹ **`clear_context()` is now called before every
  generation.** Confirmed on hardware by differential measurement: removing that
  one line makes turn 2 fail with `HAILO_INVALID_OPERATION(6)`; restoring it makes
  the turn pass.
- **Bounded the resources media preprocessing may hold.** Image and audio decoding
  now reserve their worst-case working set up front and reject requests that
  exceed it rather than queueing them.
- **The no-HailoRT build did not pass its own gate**; fixed, and CI added.

### Changed

- **The two shim implementations now share one declaration header
  (`src/hailort/shim.h`).** `build.rs` compiles `shim.cpp` or `shim_stub.cpp`
  depending on whether the SDK is present, and because `extern "C"` encodes
  neither arity nor types in the symbol name, changing one without the other
  compiled, linked and passed tests on both machines while being silent undefined
  behaviour at runtime. **Note that this closes the C++-to-C++ face only — Rust's
  `ffi.rs` still mirrors these declarations by hand.**

### Known limitations

- **A continuous CMA leak of roughly 14 MB/min occurs during inference**, on a path
  independent of load/unload. Residency does not remove it; sessions longer than
  about 30 minutes need a Pi reboot to stay stable.
- CMA is reclaimed only by rebooting the Pi. Process exit does not return it —
  measured.

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
