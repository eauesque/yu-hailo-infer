// Stub shim — compiled on hosts without hailo/hailort.hpp (WSL2, CI, non-Pi).
// All functions return a sentinel error; the real shim is used on Hailo hardware.
#include "shim.h"

#include <cstring>
#include <string>

struct YuHailortYolo {};
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

static void ensure_vdevice_created()
{
    static const bool created = []() {
        ++vdevice_create_count;
        return true;
    }();
    (void)created;
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
    ensure_vdevice_created();
    if (nullptr == out) {
        return HAILO_STUB_ERR;
    }
    *out = new YuHailortYolo();
    ++yolo_create_count;
    return 0;
}
void yu_hailort_yolo_release(YuHailortYolo *ctx)
{
    delete ctx;
    ++yolo_release_count;
}
int yu_hailort_yolo_metadata(const YuHailortYolo *, YuHailortYoloMetadata *) { return HAILO_STUB_ERR; }
int yu_hailort_yolo_run(YuHailortYolo *, const uint8_t *, size_t, YuHailortBuffer *, size_t, uint32_t) { return HAILO_STUB_ERR; }

void yu_hailort_string_free(char *value) { delete[] value; }

int yu_hailort_s2t_create(const char *, YuHailortSpeech2Text **out)
{
    ensure_vdevice_created();
    if (nullptr == out) {
        return HAILO_STUB_ERR;
    }
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
    ensure_vdevice_created();
    if (nullptr == out) {
        return HAILO_STUB_ERR;
    }
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
    ensure_vdevice_created();
    if (nullptr == out) {
        return HAILO_STUB_ERR;
    }
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
int hailo_create_vdevice(void *, void **) { return HAILO_STUB_ERR; }
int hailo_release_vdevice(void *) { return HAILO_STUB_ERR; }
int hailo_create_hef_file(void **, const char *) { return HAILO_STUB_ERR; }
int hailo_release_hef(void *) { return HAILO_STUB_ERR; }
int hailo_hef_get_all_vstream_infos(void *, const char *, void *, size_t *) { return HAILO_STUB_ERR; }
int hailo_configure_vdevice(void *, void *, void *, void **, size_t *) { return HAILO_STUB_ERR; }
int hailo_shutdown_network_group(void *) { return HAILO_STUB_ERR; }
int hailo_activate_network_group(void *, void *, void **) { return HAILO_STUB_ERR; }
int hailo_deactivate_network_group(void *) { return HAILO_STUB_ERR; }
int hailo_hef_make_input_vstream_params(void *, const char *, bool, int, void *, size_t *) { return HAILO_STUB_ERR; }
int hailo_hef_make_output_vstream_params(void *, const char *, bool, int, void *, size_t *) { return HAILO_STUB_ERR; }
int hailo_get_vstream_frame_size(void *, void *, size_t *) { return HAILO_STUB_ERR; }
int hailo_infer(void *, void *, void *, size_t, void *, void *, size_t, size_t) { return HAILO_STUB_ERR; }

}
