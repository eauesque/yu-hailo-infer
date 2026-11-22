// Stub shim — compiled on hosts without hailo/hailort.hpp (WSL2, CI, non-Pi).
// All functions return a sentinel error; the real shim is used on Hailo hardware.
#include "shim.h"

#include <cstring>
#include <string>
#include <array>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <new>

struct YuHailortYolo {
    bool metadata_enabled = false;
};
struct YuHailortWd {
    std::vector<uint8_t> hef;
    bool metadata_ready = false;
    bool configured = false;
};
struct YuHailortSpeech2Text {};
struct YuHailortLlm {
    std::string context;
};
struct YuHailortLlmStream {};
struct YuHailortVlm {};
struct YuHailortVlmStream {};

extern "C" {

static const int HAILO_STUB_ERR = 1; // HAILO_NOT_INITIALIZED equivalent

static std::string &vdevice_group_id()
{
    static std::string group_id = "YU_SHARED";
    return group_id;
}

static size_t vdevice_create_count = 0;
static size_t yolo_create_count = 0;
static size_t yolo_release_count = 0;
static size_t s2t_create_count = 0;
static size_t s2t_release_count = 0;
static size_t llm_create_count = 0;
static size_t llm_release_count = 0;
static size_t llm_clear_context_count = 0;
static size_t vlm_create_count = 0;
static size_t vlm_release_count = 0;

// Synthetic stub statuses; these are deliberately not SDK integer values.
enum WdStubStatus { HAILO_INVALID_HEF = -101, HAILO_HEF_FILE_CORRUPTED,
    HAILO_HEF_NOT_SUPPORTED, HAILO_HEF_NOT_COMPATIBLE_WITH_DEVICE, HAILO_INVALID_HEF_USE };
static thread_local int wd_open_status = 0;
void yu_hailort_stub_wd_open_status(int status) { wd_open_status = status; }
int yu_hailort_stub_wd_invalid_hef_status(size_t index) {
    static const int statuses[] = {2, HAILO_INVALID_HEF, HAILO_HEF_FILE_CORRUPTED,
        HAILO_HEF_NOT_SUPPORTED, HAILO_HEF_NOT_COMPATIBLE_WITH_DEVICE, HAILO_INVALID_HEF_USE};
    return statuses[index];
}
static thread_local int wd_failure = 0;
enum { HAILO_COMMUNICATION_CLOSED = -201 }; // Synthetic status, deliberately not the SDK value.
static thread_local bool yolo_communication_closed = false;
void yu_hailort_stub_yolo_communication_closed(bool closed) { yolo_communication_closed = closed; }
bool yu_hailort_is_communication_closed(int status) { return status == HAILO_COMMUNICATION_CLOSED; }
static thread_local std::array<size_t, 7> wd_counts {};
static thread_local std::string wd_calls;
static thread_local YuHailortYoloMetadata wd_metadata_override {};
static thread_local bool wd_has_metadata_override = false;
static thread_local std::vector<uint8_t> wd_output;
static thread_local uint32_t wd_delay_ms = 0;
static thread_local uint32_t wd_timeout_ms = 0;
static thread_local std::vector<uint8_t> wd_input;
static bool fail_initial_vdevice = false;

static void wd_record(size_t event, const char *name)
{
    ++wd_counts[event];
    if (!wd_calls.empty()) wd_calls += ",";
    wd_calls += name;
}

void yu_hailort_stub_wd_fail(int stage) { wd_failure = stage; }
void yu_hailort_stub_wd_metadata(int field, double value)
{
    wd_has_metadata_override = true;
    auto &m = wd_metadata_override;
    auto &i = m.inputs[0];
    auto &o = m.outputs[0];
    switch (field) {
    case -1:
        wd_timeout_ms = 0;
        wd_input.clear();
        m = {};
        m.inputs_count = m.outputs_count = 1;
        i = {"input", 448, 448, 3, 1, 0.0f, 1.0f, 602112, 1};
        o = {"output", 1, 1, 3, 1, 10.0f, 0.1f, 3, 1};
        break;
    case 0: m.inputs_count = value; break;
    case 1: m.outputs_count = value; break;
    case 2: i.height = value; break;
    case 3: i.width = value; break;
    case 4: i.features = value; break;
    case 5: i.format_type = value; break;
    case 6: i.format_order = value; break;
    case 7: i.frame_size = value; break;
    case 8: o.height = value; break;
    case 9: o.width = value; break;
    case 10: o.features = value; break;
    case 11: o.format_type = value; break;
    case 12: o.frame_size = value; break;
    case 13: o.qp_scale = value; break;
    case 14: o.qp_zp = value; break;
    case 15: i.qp_scale = value; break;
    case 16: i.qp_zp = value; break;
    }
}
void yu_hailort_stub_wd_output(const uint8_t *bytes, size_t size, uint32_t delay_ms)
{
    wd_output.clear();
    if (bytes && size) wd_output.assign(bytes, bytes + size);
    wd_delay_ms = delay_ms;
}
uint32_t yu_hailort_stub_wd_timeout(void) { return wd_timeout_ms; }
size_t yu_hailort_stub_wd_input(uint8_t *bytes, size_t size)
{
    if (bytes && size == wd_input.size()) std::memcpy(bytes, wd_input.data(), size);
    return wd_input.size();
}
size_t yu_hailort_stub_wd_count(size_t event) { return wd_counts.at(event); }
const char *yu_hailort_stub_wd_calls(void) { return wd_calls.c_str(); }
void yu_hailort_stub_vdevice_fail_initial(void) { fail_initial_vdevice = true; }

static int ensure_vdevice_created();

int yu_hailort_wd_open(const uint8_t *hef, size_t size, YuHailortWd **out)
try {
    if (!out) return 2;
    *out = nullptr;
    wd_record(1, "open");
    if (!hef || !size) return 2;
    const auto status = ensure_vdevice_created();
    if (status) return status;
    if (wd_open_status) return wd_open_status;
    if (wd_failure == 20) throw std::bad_alloc();
    if (wd_failure == 1) return HAILO_STUB_ERR;
    auto ctx = std::make_unique<YuHailortWd>();
    ctx->hef.assign(hef, hef + size);
    *out = ctx.release();
    ++wd_counts[6]; // Successfully created handles, distinct from open attempts.
    return 0;
} catch (...) {
    return HAILO_STUB_ERR;
}
int yu_hailort_wd_metadata(YuHailortWd *ctx, YuHailortYoloMetadata *out)
try {
    wd_record(2, "metadata");
    if (!ctx || !out || ctx->configured) return 2;
    if (wd_failure == 21) throw std::bad_alloc();
    ctx->metadata_ready = false;
    if (wd_failure == 2) return HAILO_STUB_ERR;
    *out = {};
    out->inputs_count = out->outputs_count = 1;
    out->inputs[0] = {"input", 448, 448, 3, 1, 0.0f, 1.0f, 602112,
        static_cast<uint32_t>(wd_failure == 4 ? 0 : 1)};
    out->outputs[0] = {"output", 1, 1, 10861, 1, 0.0f, 1.0f, 10861, 1};
    if (wd_has_metadata_override) *out = wd_metadata_override;
    ctx->metadata_ready = true;
    return 0;
} catch (...) {
    return HAILO_STUB_ERR;
}
void yu_hailort_wd_release(YuHailortWd *ctx) noexcept
try {
    if (!ctx) return;
    delete ctx;
    wd_record(4, "release");
    if (wd_failure == 24) throw std::bad_alloc();
} catch (...) {
}
int yu_hailort_wd_configure(YuHailortWd *opened, YuHailortWd **out)
try {
    std::unique_ptr<YuHailortWd, decltype(&yu_hailort_wd_release)> ctx(opened, yu_hailort_wd_release);
    wd_record(3, "configure");
    if (!out) return 2;
    *out = nullptr;
    if (!ctx || !ctx->metadata_ready || ctx->configured || wd_failure == 3) return 2;
    if (wd_failure == 22) throw std::bad_alloc();
    if (wd_failure == 5) return HAILO_STUB_ERR;
    if (wd_failure == 30) return HAILO_COMMUNICATION_CLOSED;
    ctx->configured = true;
    *out = ctx.release();
    return 0;
} catch (...) {
    return HAILO_STUB_ERR;
}
bool yu_hailort_wd_is_timeout(int status)
try { return status == -9; }
catch (...) { return false; }

bool yu_hailort_wd_is_invalid_hef(int status)
try {
    return status == 2 || status == HAILO_INVALID_HEF || status == HAILO_HEF_FILE_CORRUPTED ||
        status == HAILO_HEF_NOT_SUPPORTED || status == HAILO_HEF_NOT_COMPATIBLE_WITH_DEVICE ||
        status == HAILO_INVALID_HEF_USE;
} catch (...) {
    return false;
}

int yu_hailort_wd_run(YuHailortWd *ctx, const uint8_t *input, size_t size,
    YuHailortBuffer *outputs, size_t count, uint32_t timeout_ms)
try {
    wd_record(5, "run");
    if (wd_failure == 23) throw std::bad_alloc();
    wd_timeout_ms = timeout_ms;
    std::this_thread::sleep_for(std::chrono::milliseconds(wd_delay_ms));
    if (wd_failure == 6) return HAILO_STUB_ERR;
    if (wd_failure == 7) return -9; // Test-only timeout status, distinct from SDK values.
    const size_t output_size = wd_has_metadata_override ? wd_metadata_override.outputs[0].frame_size : 10861;
    if (!ctx || !ctx->configured || !input || size != 602112 || !outputs || count != 1 ||
        !outputs[0].data || outputs[0].size != output_size) return 2;
    wd_input.assign(input, input + size);
    if (wd_output.empty()) std::memset(outputs[0].data, input[0], outputs[0].size);
    else {
        if (wd_output.size() != outputs[0].size) return 2;
        std::memcpy(outputs[0].data, wd_output.data(), wd_output.size());
    }
    return 0;
} catch (...) {
    return HAILO_STUB_ERR;
}

static int ensure_vdevice_created()
{
    static const int creation_status = []() {
        ++vdevice_create_count;
        return fail_initial_vdevice ? HAILO_STUB_ERR : 0;
    }();
    return creation_status;
}

int yu_hailort_set_vdevice_group_id(const char *group_id)
{
    if (nullptr == group_id) {
        return HAILO_STUB_ERR;
    }
    vdevice_group_id() = group_id;
    return 0;
}

size_t yu_hailort_stub_vdevice_create_count(void) { return vdevice_create_count; }
size_t yu_hailort_stub_yolo_create_count(void) { return yolo_create_count; }
size_t yu_hailort_stub_yolo_release_count(void) { return yolo_release_count; }
size_t yu_hailort_stub_s2t_create_count(void) { return s2t_create_count; }
size_t yu_hailort_stub_s2t_release_count(void) { return s2t_release_count; }
size_t yu_hailort_stub_llm_create_count(void) { return llm_create_count; }
size_t yu_hailort_stub_llm_release_count(void) { return llm_release_count; }
size_t yu_hailort_stub_llm_clear_context_count(void) { return llm_clear_context_count; }
size_t yu_hailort_stub_vlm_create_count(void) { return vlm_create_count; }
size_t yu_hailort_stub_vlm_release_count(void) { return vlm_release_count; }

int yu_hailort_yolo_create(const char *, YuHailortYolo **out)
{
    if (nullptr == out) {
        return HAILO_STUB_ERR;
    }
    *out = nullptr;
    auto status = ensure_vdevice_created();
    if (status) return status;
    ++yolo_create_count;
    if (yolo_communication_closed) return HAILO_COMMUNICATION_CLOSED;
    *out = new YuHailortYolo();
    return 0;
}
void yu_hailort_yolo_release(YuHailortYolo *ctx)
{
    delete ctx;
    ++yolo_release_count;
}
void yu_hailort_stub_yolo_enable_metadata(YuHailortYolo *ctx)
{
    ctx->metadata_enabled = true;
}
int yu_hailort_yolo_metadata(const YuHailortYolo *ctx, YuHailortYoloMetadata *metadata)
{
    if ((nullptr == ctx) || (nullptr == metadata) || !ctx->metadata_enabled) {
        return HAILO_STUB_ERR;
    }
    static const uint32_t NHWC = 1; // Matches ffi.rs HAILO_FORMAT_ORDER_NHWC.
    *metadata = {};
    metadata->inputs_count = 1;
    metadata->outputs_count = 1;
    metadata->inputs[0] = {"input", 1, 1, 3, 1, 0.0f, 1.0f, 3, NHWC};
    metadata->outputs[0] = {"output", 1, 1, 512, 1, 0.0f, 1.0f, 512, NHWC};
    return 0;
}
int yu_hailort_yolo_run(YuHailortYolo *, const uint8_t *, size_t, YuHailortBuffer *, size_t, uint32_t) { return HAILO_STUB_ERR; }

void yu_hailort_string_free(char *value) { delete[] value; }

int yu_hailort_s2t_create(const char *, YuHailortSpeech2Text **out)
{
    if (nullptr == out) {
        return HAILO_STUB_ERR;
    }
    *out = nullptr;
    auto status = ensure_vdevice_created();
    if (status) return status;
    *out = new YuHailortSpeech2Text();
    ++s2t_create_count;
    return 0;
}
void yu_hailort_s2t_release(YuHailortSpeech2Text *ctx)
{
    delete ctx;
    ++s2t_release_count;
}
int yu_hailort_s2t_generate_text(YuHailortSpeech2Text *, const float *, size_t, int, const char *, float, uint32_t, char **) { return HAILO_STUB_ERR; }
int yu_hailort_s2t_generate_segments(YuHailortSpeech2Text *, const float *, size_t, int, const char *, float, uint32_t, char **) { return HAILO_STUB_ERR; }
int yu_hailort_s2t_tokenize(YuHailortSpeech2Text *, const char *, int *, size_t *) { return HAILO_STUB_ERR; }

int yu_hailort_llm_create(const char *, const char *, bool, YuHailortLlm **out)
{
    if (nullptr == out) {
        return HAILO_STUB_ERR;
    }
    *out = nullptr;
    auto status = ensure_vdevice_created();
    if (status) return status;
    *out = new YuHailortLlm();
    ++llm_create_count;
    return 0;
}
void yu_hailort_llm_release(YuHailortLlm *ctx)
{
    delete ctx;
    ++llm_release_count;
}
int yu_hailort_llm_generate_text(YuHailortLlm *ctx, const char *prompt, uint32_t, char **out_text)
{
    if ((nullptr == ctx) || (nullptr == prompt) || (nullptr == out_text)) {
        return HAILO_STUB_ERR;
    }
    ctx->context += prompt;
    *out_text = new char[ctx->context.size() + 1];
    std::memcpy(*out_text, ctx->context.c_str(), ctx->context.size() + 1);
    return 0;
}
int yu_hailort_llm_tokenize(YuHailortLlm *, const char *, int *, size_t *) { return HAILO_STUB_ERR; }
int yu_hailort_llm_context_usage(YuHailortLlm *, size_t *) { return HAILO_STUB_ERR; }
int yu_hailort_llm_max_context_capacity(YuHailortLlm *, size_t *) { return HAILO_STUB_ERR; }
int yu_hailort_llm_clear_context(YuHailortLlm *ctx)
{
    if (nullptr == ctx) {
        return HAILO_STUB_ERR;
    }
    ctx->context.clear();
    ++llm_clear_context_count;
    return 0;
}
int yu_hailort_llm_generate_stream_start(YuHailortLlm *, const char *const *, size_t, const char *const *, size_t, const float *, const float *, const uint32_t *, const float *, const uint32_t *, const bool *, const uint32_t *, YuHailortLlmStream **) { return HAILO_STUB_ERR; }
int yu_hailort_llm_stream_read(YuHailortLlmStream *, uint32_t, char **, int *) { return HAILO_STUB_ERR; }
void yu_hailort_llm_stream_release(YuHailortLlmStream *) {}

int yu_hailort_vlm_create(const char *, bool, YuHailortVlm **out)
{
    if (nullptr == out) {
        return HAILO_STUB_ERR;
    }
    *out = nullptr;
    auto status = ensure_vdevice_created();
    if (status) return status;
    *out = new YuHailortVlm();
    ++vlm_create_count;
    return 0;
}
void yu_hailort_vlm_release(YuHailortVlm *ctx)
{
    delete ctx;
    ++vlm_release_count;
}
int yu_hailort_vlm_generate_text(YuHailortVlm *, const char *, const uint8_t *const *, const size_t *, size_t, uint32_t, char **) { return HAILO_STUB_ERR; }
int yu_hailort_vlm_tokenize(YuHailortVlm *, const char *, int *, size_t *) { return HAILO_STUB_ERR; }
int yu_hailort_vlm_context_usage(YuHailortVlm *, size_t *) { return HAILO_STUB_ERR; }
int yu_hailort_vlm_max_context_capacity(YuHailortVlm *, size_t *) { return HAILO_STUB_ERR; }
int yu_hailort_vlm_clear_context(YuHailortVlm *) { return 0; }
int yu_hailort_vlm_input_frame_info(YuHailortVlm *, uint32_t *, uint32_t *, uint32_t *, uint32_t *, uint32_t *, uint32_t *) { return HAILO_STUB_ERR; }
int yu_hailort_vlm_generate_stream_start(YuHailortVlm *, const char *, const char *, const uint8_t *const *, const size_t *, size_t, const float *, const float *, const uint32_t *, const float *, const uint32_t *, const bool *, const uint32_t *, YuHailortVlmStream **) { return HAILO_STUB_ERR; }
int yu_hailort_vlm_stream_read(YuHailortVlmStream *, uint32_t, char **, int *) { return HAILO_STUB_ERR; }
void yu_hailort_vlm_stream_release(YuHailortVlmStream *) {}

// hailo_* C API stubs — required by safe.rs/metadata.rs RAII wrappers.
// None of the hailo hardware API is available without hailort.hpp; these
// return HAILO_STUB_ERR so the process can start and tests can run.
static thread_local bool c_api_communication_closed = false;
void yu_hailort_stub_c_api_communication_closed(bool closed) { c_api_communication_closed = closed; }
static int c_api_status() { return c_api_communication_closed ? HAILO_COMMUNICATION_CLOSED : HAILO_STUB_ERR; }
int hailo_create_vdevice(void *, void **) { return c_api_status(); }
int hailo_release_vdevice(void *) { return c_api_status(); }
int hailo_create_hef_file(void **, const char *) { return HAILO_STUB_ERR; }
int hailo_release_hef(void *) { return c_api_status(); }
int hailo_hef_get_all_vstream_infos(void *, const char *, void *, size_t *) { return HAILO_STUB_ERR; }
int hailo_configure_vdevice(void *, void *, void *, void **, size_t *) { return HAILO_STUB_ERR; }
int hailo_shutdown_network_group(void *) { return c_api_status(); }
int hailo_activate_network_group(void *, void *, void **) { return HAILO_STUB_ERR; }
int hailo_deactivate_network_group(void *) { return c_api_status(); }
int hailo_hef_make_input_vstream_params(void *, const char *, bool, int, void *, size_t *) { return HAILO_STUB_ERR; }
int hailo_hef_make_output_vstream_params(void *, const char *, bool, int, void *, size_t *) { return HAILO_STUB_ERR; }
int hailo_get_vstream_frame_size(void *, void *, size_t *) { return HAILO_STUB_ERR; }
int hailo_infer(void *, void *, void *, size_t, void *, void *, size_t, size_t) { return HAILO_STUB_ERR; }

}
