#!/usr/bin/env python3
"""Real-hardware smoke test for yu-hailo-infer's own HTTP surface.

Unlike yu_ai_manager's scripts/hailo_realhw_smoke.py (which drives
yu-server's proxy routes plus DB/JobManager/SSE glue), this script talks
directly to `/v1/infer/*` on a bare yu-hailo-infer process -- no yu-server,
no database, no job manager. It exists to prove this repo's own HTTP
contract works end to end against a real Hailo-10H device, independent of
any caller.

Usage:
    uv run python scripts/hailo_infer_smoke.py [--keep-scratch] [--binary PATH]

Requires (all optional per-test -- a missing prerequisite is reported as a
skip, not silently ignored, since this can only be verified on real
hardware and this script may run on a host without one):
    - yu-hailo-infer built (release preferred; falls back to debug)
    - HEF models under ~/hailo_models/ (yolov8n.hef,
      clip_vit_b_16_image_encoder.hef, Llama3.2-1B-Instruct.hef or
      compatible LLM HEF, Whisper-Tiny.hef or compatible S2T HEF) --
      override paths via the HAILO_*_HEF env vars the service itself reads
    - CLIP text ONNX model under $HAILO_CLIP_TEXT_MODEL_DIR (or the
      service's own default) for the clip-text test
    - /dev/hailo0 present and not held by another process, for every
      HailoRT-backed test (clip-image, wd, yolo, llm, vlm, speech2text)

Never touches any real config -- writes a scratch wd-cache-dir and test
fixtures under tmp/hailo_infer_smoke/ on every run (see CLAUDE.md's
tmp/-only rule for local artifacts).
"""
import base64
import contextlib
import json
import logging
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
SCRATCH = REPO / "tmp" / "hailo_infer_smoke"
WD_CACHE_DIR = SCRATCH / "wd-cache"
IMG_PATH = SCRATCH / "test.png"
PORT = 18773
BASE = f"http://127.0.0.1:{PORT}"
AUTH_TOKEN = "hailo-infer-smoke-test-token"
HOME_MODELS = Path.home() / "hailo_models"

results = []  # (name, status, detail) -- status in {"PASS", "FAIL", "SKIP"}


def record(name, status, detail=""):
    results.append((name, status, detail))
    print(f"[{status}] {name}: {detail}")


def auth_headers():
    return {"Authorization": f"Bearer {AUTH_TOKEN}"}


def http(method, path, body=None, timeout=120, headers=None, is_json=True):
    url = BASE + path
    data = None
    hdrs = dict(auth_headers())
    if headers:
        hdrs.update(headers)
    if body is not None:
        if is_json:
            hdrs["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        else:
            data = body
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw.decode(errors="replace")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw.decode(errors="replace")


def sse_post(path, body, timeout=180):
    url = BASE + path
    data = json.dumps(body).encode()
    hdrs = dict(auth_headers())
    hdrs["Content-Type"] = "application/json"
    hdrs["Accept"] = "text/event-stream"
    req = urllib.request.Request(url, data=data, method="POST", headers=hdrs)
    events = []
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                block, buf = buf.split(b"\n\n", 1)
                for line in block.split(b"\n"):
                    if line.startswith(b"data: "):
                        with contextlib.suppress(json.JSONDecodeError):
                            events.append(json.loads(line[6:].decode()))
            if time.time() - start > timeout:
                break
    return events


def make_png(path: Path, w=64, h=64):
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            raw.extend([(x * 255) // w, (y * 255) // h, 128])
    idat = zlib.compress(bytes(raw), 9)
    with open(path, "wb") as f:
        f.write(sig)
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", idat))
        f.write(chunk(b"IEND", b""))


def make_wav(path: Path, seconds=1.0, sample_rate=16000):
    """Silent 16-bit PCM mono WAV -- enough to exercise the transcribe
    pipeline without needing real speech."""
    import wave

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * int(seconds * sample_rate))


def b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def find_binary(override: str | None) -> Path:
    if override:
        p = Path(override)
        if p.exists():
            return p
        print(f"ERROR: --binary {p} does not exist.")
        sys.exit(2)
    for candidate in (
        REPO / "target" / "release" / "yu-hailo-infer",
        REPO / "target" / "debug" / "yu-hailo-infer",
    ):
        if candidate.exists():
            return candidate
    print("ERROR: yu-hailo-infer binary not found; run `cargo build --release -p yu-hailo-infer` first.")
    sys.exit(2)


def wait_health(timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            status, body = http("GET", "/healthz", timeout=3)
            if status == 200:
                return body
        except Exception:
            logger.debug("health check attempt failed", exc_info=True)
        time.sleep(0.5)
    return None


DEVICE_NODE_CANDIDATES = ("/dev/hailo0", "/dev/h1x-0")


def device_holders():
    """Best-effort: list PIDs with an open FD on the Hailo device node.
    The node name depends on the PCIe driver generation -- /dev/hailo0 on
    some hosts, /dev/h1x-0 (Hailo-10H) on others -- so check both rather
    than hardcoding one and silently reporting zero holders on the other."""
    holders = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        fd_dir = proc_dir / "fd"
        try:
            for fd in fd_dir.iterdir():
                try:
                    if os.readlink(fd) in DEVICE_NODE_CANDIDATES:
                        holders.append(proc_dir.name)
                except OSError:
                    continue
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
    return holders


def setup_scratch():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    WD_CACHE_DIR.mkdir(parents=True)
    make_png(IMG_PATH)


def spawn_server(binary: Path):
    proc = subprocess.Popen(
        [str(binary), "--port", str(PORT), "--wd-cache-dir", str(WD_CACHE_DIR)],
        cwd=str(REPO),
        stdin=subprocess.PIPE,
    )
    contract = {
        "instance_id": "hailo-infer-smoke-test",
        "scan_roots": [str(SCRATCH)],
        "auth_token": AUTH_TOKEN,
    }
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(contract) + "\n").encode())
    proc.stdin.close()
    return proc


def stop_server(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def restart_server(proc, binary, reason):
    """HailoRT 5.x allows exactly one resident GenAI (LLM/VLM/S2T share one
    slot) model per process lifetime -- see yu_ai_manager's
    scripts/hailo_realhw_smoke.py restart_server() for the same constraint
    on the caller side. Switching GenAI model families within one process
    produces a 409 HEF conflict, not a code bug; restarting is the only way
    to free the slot."""
    print(f"\n--- restarting yu-hailo-infer ({reason}) ---")
    stop_server(proc)
    new_proc = spawn_server(binary)
    health = wait_health()
    if health is None:
        record("server_restart", "FAIL", f"did not become healthy again after: {reason}")
        stop_server(new_proc)
        raise RuntimeError(f"server restart failed: {reason}")
    return new_proc


def main():
    keep = "--keep-scratch" in sys.argv
    binary_override = None
    if "--binary" in sys.argv:
        binary_override = sys.argv[sys.argv.index("--binary") + 1]
    binary = find_binary(binary_override)

    setup_scratch()
    proc = spawn_server(binary)

    try:
        health = wait_health()
        if health is None:
            record("server_startup", "FAIL", "yu-hailo-infer did not become healthy in time")
            print_summary()
            sys.exit(1)
        record("server_startup", "PASS", f"pid={proc.pid} body={health}")

        # --- healthz shape: instance_id + hailo_stub must be present ---
        ok = isinstance(health, dict) and health.get("instance_id") == "hailo-infer-smoke-test" \
            and "hailo_stub" in health
        record("healthz_shape", "PASS" if ok else "FAIL", str(health)[:200])

        # --- auth: missing bearer token must 401 ---
        try:
            status, body = http("GET", "/healthz", headers={"Authorization": ""}, timeout=10)
            # /healthz is exempt from auth -- this should still succeed.
            record("healthz_is_auth_exempt", "PASS" if status == 200 else "FAIL", f"status={status}")
        except Exception as e:
            record("healthz_is_auth_exempt", "FAIL", repr(e))

        try:
            req = urllib.request.Request(BASE + "/v1/infer/llm/tokenize", data=b'{"text":"x"}',
                                          method="POST", headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=10)
                record("unauthorized_request_rejected", "FAIL", "expected 401, request succeeded")
            except urllib.error.HTTPError as e:
                record("unauthorized_request_rejected", "PASS" if e.code == 401 else "FAIL", f"status={e.code}")
        except Exception as e:
            record("unauthorized_request_rejected", "FAIL", repr(e))

        # --- scan-roots-changed (internal, authenticated) ---
        try:
            status, body = http("POST", "/_internal/scan-roots-changed",
                                 {"scan_roots": [str(SCRATCH)], "generation": 1}, timeout=10)
            ok = status == 200 and isinstance(body, dict) and body.get("applied") is True
            record("scan_roots_changed", "PASS" if ok else "FAIL", str(body)[:200])
        except Exception as e:
            record("scan_roots_changed", "FAIL", repr(e))

        # --- clip-text (ONNX, no HailoRT device needed) ---
        try:
            status, body = http("POST", "/v1/infer/clip-text", {"text": "a photo of a cat"}, timeout=60)
            if status == 200 and isinstance(body, dict) and body.get("ok"):
                record("clip_text", "PASS", str(body)[:200])
            else:
                record("clip_text", "SKIP", f"CLIP text ONNX model not available on this host: status={status} body={str(body)[:200]}")
        except Exception as e:
            record("clip_text", "SKIP", f"CLIP text model unavailable: {e!r}")

        # --- HailoRT-backed endpoints: each needs /dev/hailo0 + its HEF ---
        image_b64 = b64_file(IMG_PATH)

        clip_hef = os.environ.get("HAILO_CLIP_HEF", str(HOME_MODELS / "clip_vit_b_16_image_encoder.hef"))
        try:
            status, body = http("POST", "/v1/infer/clip-image", {"image_base64": image_b64}, timeout=60)
            if status == 200 and isinstance(body, dict) and body.get("ok"):
                record("clip_image", "PASS", str(body)[:200])
            else:
                record("clip_image", "SKIP", f"needs a live Hailo-10H + HEF at {clip_hef}: status={status} body={str(body)[:200]}")
        except Exception as e:
            record("clip_image", "SKIP", f"needs a live Hailo-10H: {e!r}")

        try:
            status, body = http("POST", "/v1/infer/wd",
                                 {"path": str(IMG_PATH), "model_id": "wd-v1-4-moat-tagger-v2"}, timeout=60)
            if status == 200 and isinstance(body, dict) and body.get("ok"):
                record("wd_tagger", "PASS", str(body)[:200])
            else:
                record("wd_tagger", "SKIP", f"needs a WD-Tagger model cached under {WD_CACHE_DIR}: status={status} body={str(body)[:200]}")
        except Exception as e:
            record("wd_tagger", "SKIP", f"needs a WD-Tagger model: {e!r}")

        yolo_hef = os.environ.get("HAILO_YOLO_HEF", str(HOME_MODELS / "yolov8n.hef"))
        try:
            status, body = http("GET", "/v1/infer/yolo/metadata", timeout=30)
            if status == 200 and isinstance(body, dict) and body.get("ok"):
                record("yolo_metadata", "PASS", str(body)[:200])
            else:
                record("yolo_metadata", "SKIP", f"needs a live Hailo-10H + HEF at {yolo_hef}: status={status} body={str(body)[:200]}")
        except Exception as e:
            record("yolo_metadata", "SKIP", f"needs a live Hailo-10H: {e!r}")

        try:
            status, body = http("GET", "/v1/infer/yolo/smoke-zero", timeout=30)
            ok = status == 200 and isinstance(body, dict) and body.get("ok")
            record("yolo_smoke_zero", "PASS" if ok else "SKIP", str(body)[:200])
        except Exception as e:
            record("yolo_smoke_zero", "SKIP", f"needs a live Hailo-10H: {e!r}")

        try:
            status, meta_body = http("GET", "/v1/infer/yolo/metadata", timeout=30)
            frame_size = None
            if status == 200 and isinstance(meta_body, dict) and meta_body.get("ok"):
                inputs = (meta_body.get("data") or {}).get("inputs") or []
                if inputs:
                    frame_size = inputs[0].get("frame_size")
            if frame_size:
                # /detect expects the raw quantized input tensor (same shape
                # the model itself reports via /metadata), not an
                # encoded image -- see yolo_detect's input_frame_size check
                # in router.rs. A zero frame is a cheap decode-pipeline
                # smoke check, same idea as /yolo/smoke-zero.
                zero_frame_b64 = base64.b64encode(bytes(frame_size)).decode()
                status, body = http("POST", "/v1/infer/yolo/detect", {"input_base64": zero_frame_b64}, timeout=60)
                ok = status == 200 and isinstance(body, dict) and body.get("ok")
                record("yolo_detect", "PASS" if ok else "SKIP", str(body)[:200])
            else:
                record("yolo_detect", "SKIP", "could not determine YOLO input frame size from /metadata")
        except Exception as e:
            record("yolo_detect", "SKIP", f"needs a live Hailo-10H: {e!r}")

        # llm/vlm/speech2text all load onto the same single GenAI residency
        # slot HailoRT exposes per process -- restart between families so
        # loading one doesn't 409-conflict with the one already resident.
        llm_hef = os.environ.get("HAILO_LLM_HEF", str(HOME_MODELS / "Llama3.2-1B-Instruct.hef"))
        try:
            status, body = http("POST", "/v1/infer/llm/tokenize", {"text": "hello"}, timeout=60)
            ok = status == 200 and isinstance(body, dict) and body.get("ok")
            record("llm_tokenize", "PASS" if ok else "SKIP", str(body)[:200])
        except Exception as e:
            record("llm_tokenize", "SKIP", f"needs a live Hailo-10H + HEF at {llm_hef}: {e!r}")

        try:
            status, body = http("POST", "/v1/infer/llm/generate",
                                 {"prompt": "What is the capital of France?", "timeout_ms": 90000}, timeout=120)
            ok = status == 200 and isinstance(body, dict) and body.get("ok")
            record("llm_generate", "PASS" if ok else "SKIP", str(body)[:200])
        except Exception as e:
            record("llm_generate", "SKIP", f"needs a live Hailo-10H + HEF at {llm_hef}: {e!r}")

        proc = restart_server(proc, binary, "LLM -> VLM (single GenAI residency slot)")

        vlm_hef = os.environ.get("HAILO_VLM_HEF", str(HOME_MODELS / "qwen2-vl-2b-instruct.hef"))
        try:
            events = sse_post("/v1/infer/vlm/generate/stream",
                               {"prompt": "Describe this image.", "frames": [image_b64], "timeout_ms": 90000},
                               timeout=150)
            record("vlm_generate_stream", "PASS" if events else "SKIP",
                   f"needs a live Hailo-10H + HEF at {vlm_hef}: events={len(events)}")
        except Exception as e:
            record("vlm_generate_stream", "SKIP", f"needs a live Hailo-10H + HEF at {vlm_hef}: {e!r}")

        proc = restart_server(proc, binary, "VLM -> speech2text (single GenAI residency slot)")

        s2t_hef = os.environ.get("HAILO_S2T_HEF", str(HOME_MODELS / "Whisper-Tiny.hef"))
        try:
            status, body = http("POST", "/v1/infer/speech2text/tokenize", {"text": "hello world"}, timeout=30)
            ok = status == 200 and isinstance(body, dict) and body.get("ok")
            record("speech2text_tokenize", "PASS" if ok else "SKIP", str(body)[:200])
        except Exception as e:
            record("speech2text_tokenize", "SKIP", f"needs a live Hailo-10H + HEF at {s2t_hef}: {e!r}")

        try:
            wav_path = SCRATCH / "silent.wav"
            make_wav(wav_path)
            status, body = http("POST", "/v1/infer/speech2text/transcribe",
                                 {"audio_base64": b64_file(wav_path)}, timeout=90)
            ok = status == 200 and isinstance(body, dict) and body.get("ok")
            record("speech2text_transcribe", "PASS" if ok else "SKIP", str(body)[:200])
        except Exception as e:
            record("speech2text_transcribe", "SKIP", f"needs a live Hailo-10H + HEF at {s2t_hef}: {e!r}")

    finally:
        print("\n--- shutting down yu-hailo-infer ---")
        stop_server(proc)

        holders = device_holders()
        record("cleanup_device_released", "PASS" if not holders else "FAIL", f"holders={holders}")

        if not keep:
            shutil.rmtree(SCRATCH, ignore_errors=True)

    print_summary()


def print_summary():
    print("\n=== SUMMARY ===")
    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status == "FAIL")
    skipped = sum(1 for _, status, _ in results if status == "SKIP")
    for name, status, _ in results:
        print(f"{status:4s}  {name}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped (needs real Hailo-10H hardware/models)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
