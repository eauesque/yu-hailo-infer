#pragma once

// Keeps the real and stub C++ implementations on the same C ABI.
// Tensor metadata layouts are checked here and in Rust's hand-written shim.rs.
// These layouts require 64-bit pointers and size_t.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

extern "C" {

struct YuHailortTensorInfo {
    char name[128];
    uint32_t height;
    uint32_t width;
    uint32_t features;
    uint32_t format_type;
    float qp_zp;
    float qp_scale;
    size_t frame_size;
    uint32_t format_order;
};

struct YuHailortYoloMetadata {
    size_t inputs_count;
    size_t outputs_count;
    YuHailortTensorInfo inputs[40];
    YuHailortTensorInfo outputs[40];
};

static_assert(sizeof(void *) == 8 && sizeof(size_t) == 8,
    "Tensor metadata ABI requires 64-bit pointers and size_t");
static_assert(sizeof(YuHailortTensorInfo) == 168);
static_assert(offsetof(YuHailortTensorInfo, name) == 0);
static_assert(offsetof(YuHailortTensorInfo, height) == 128);
static_assert(offsetof(YuHailortTensorInfo, width) == 132);
static_assert(offsetof(YuHailortTensorInfo, features) == 136);
static_assert(offsetof(YuHailortTensorInfo, format_type) == 140);
static_assert(offsetof(YuHailortTensorInfo, qp_zp) == 144);
static_assert(offsetof(YuHailortTensorInfo, qp_scale) == 148);
static_assert(offsetof(YuHailortTensorInfo, frame_size) == 152);
static_assert(offsetof(YuHailortTensorInfo, format_order) == 160);
static_assert(sizeof(YuHailortYoloMetadata) == 13456);
static_assert(offsetof(YuHailortYoloMetadata, inputs_count) == 0);
static_assert(offsetof(YuHailortYoloMetadata, outputs_count) == 8);
static_assert(offsetof(YuHailortYoloMetadata, inputs) == 16);
static_assert(offsetof(YuHailortYoloMetadata, outputs) == 6736);

struct YuHailortBuffer {
    const char *name;
    void *data;
    size_t size;
};

struct YuHailortYolo;
struct YuHailortWd;
struct YuHailortSpeech2Text;
struct YuHailortLlm;
struct YuHailortLlmStream;
struct YuHailortVlm;
struct YuHailortVlmStream;

int yu_hailort_set_vdevice_group_id(const char *group_id);

// HailoRT copies hef bytes during open; the caller may free them on return.
int yu_hailort_wd_open(const uint8_t *hef, size_t size, YuHailortWd **out);
int yu_hailort_wd_metadata(YuHailortWd *ctx, YuHailortYoloMetadata *metadata);
// Consumes ctx on success AND failure; only success returns an owned handle.
int yu_hailort_wd_configure(YuHailortWd *ctx, YuHailortWd **out);
void yu_hailort_wd_release(YuHailortWd *ctx) noexcept;
int yu_hailort_wd_run(YuHailortWd *ctx, const uint8_t *input, size_t input_size,
    YuHailortBuffer *outputs, size_t outputs_count, uint32_t timeout_ms);
bool yu_hailort_wd_is_timeout(int status);
bool yu_hailort_is_communication_closed(int status);
bool yu_hailort_wd_is_invalid_hef(int status);

int yu_hailort_yolo_create(const char *hef_path, YuHailortYolo **out);
void yu_hailort_yolo_release(YuHailortYolo *ctx);
int yu_hailort_yolo_metadata(const YuHailortYolo *ctx, YuHailortYoloMetadata *metadata);
int yu_hailort_yolo_run(
    YuHailortYolo *ctx,
    const uint8_t *input,
    size_t input_size,
    YuHailortBuffer *outputs,
    size_t outputs_count,
    uint32_t timeout_ms);

void yu_hailort_string_free(char *value);

int yu_hailort_s2t_create(const char *model_path, YuHailortSpeech2Text **out);
void yu_hailort_s2t_release(YuHailortSpeech2Text *ctx);
int yu_hailort_s2t_generate_text(
    YuHailortSpeech2Text *ctx,
    const float *audio,
    size_t audio_count,
    int task,
    const char *language,
    float repetition_penalty,
    uint32_t timeout_ms,
    char **out_text);
// Same generation as yu_hailort_s2t_generate_text, but returns the SDK's
// per-segment breakdown (start/end timestamps + text) as a JSON array
// string: [{"start_sec":F,"end_sec":F,"text":"..."},...]. Caller frees
// *out_json with yu_hailort_string_free, same as out_text above.
int yu_hailort_s2t_generate_segments(
    YuHailortSpeech2Text *ctx,
    const float *audio,
    size_t audio_count,
    int task,
    const char *language,
    float repetition_penalty,
    uint32_t timeout_ms,
    char **out_json);
int yu_hailort_s2t_tokenize(
    YuHailortSpeech2Text *ctx,
    const char *text,
    int *tokens,
    size_t *tokens_count);

int yu_hailort_llm_create(
    const char *model_path,
    const char *lora_name,
    bool optimize_memory_on_device,
    YuHailortLlm **out);
void yu_hailort_llm_release(YuHailortLlm *ctx);
int yu_hailort_llm_generate_text(
    YuHailortLlm *ctx,
    const char *prompt,
    uint32_t timeout_ms,
    char **out_text);
int yu_hailort_llm_tokenize(
    YuHailortLlm *ctx,
    const char *text,
    int *tokens,
    size_t *tokens_count);
int yu_hailort_llm_context_usage(YuHailortLlm *ctx, size_t *out);
int yu_hailort_llm_max_context_capacity(YuHailortLlm *ctx, size_t *out);
int yu_hailort_llm_clear_context(YuHailortLlm *ctx);
int yu_hailort_llm_generate_stream_start(
    YuHailortLlm *ctx,
    const char *const *messages_json,
    size_t messages_count,
    const char *const *tools_json,
    size_t tools_count,
    const float *temperature,
    const float *top_p,
    const uint32_t *top_k,
    const float *frequency_penalty,
    const uint32_t *max_generated_tokens,
    const bool *do_sample,
    const uint32_t *seed,
    YuHailortLlmStream **out);
int yu_hailort_llm_stream_read(
    YuHailortLlmStream *stream,
    uint32_t timeout_ms,
    char **out_token,
    int *out_status);
void yu_hailort_llm_stream_release(YuHailortLlmStream *stream);

int yu_hailort_vlm_create(
    const char *model_path,
    bool optimize_memory_on_device,
    YuHailortVlm **out);
void yu_hailort_vlm_release(YuHailortVlm *ctx);
int yu_hailort_vlm_generate_text(
    YuHailortVlm *ctx,
    const char *prompt,
    const uint8_t *const *frames,
    const size_t *frame_sizes,
    size_t frame_count,
    uint32_t timeout_ms,
    char **out_text);
int yu_hailort_vlm_tokenize(
    YuHailortVlm *ctx,
    const char *text,
    int *tokens,
    size_t *tokens_count);
int yu_hailort_vlm_context_usage(YuHailortVlm *ctx, size_t *out);
int yu_hailort_vlm_max_context_capacity(YuHailortVlm *ctx, size_t *out);
int yu_hailort_vlm_clear_context(YuHailortVlm *ctx);
int yu_hailort_vlm_input_frame_info(
    YuHailortVlm *ctx,
    uint32_t *frame_size,
    uint32_t *height,
    uint32_t *width,
    uint32_t *features,
    uint32_t *format_type,
    uint32_t *format_order);

int yu_hailort_vlm_generate_stream_start(
    YuHailortVlm *ctx,
    const char *prompt,
    const char *system_prompt,
    const uint8_t *const *frames,
    const size_t *frame_sizes,
    size_t frame_count,
    const float *temperature,
    const float *top_p,
    const uint32_t *top_k,
    const float *frequency_penalty,
    const uint32_t *max_generated_tokens,
    const bool *do_sample,
    const uint32_t *seed,
    YuHailortVlmStream **out);
int yu_hailort_vlm_stream_read(
    YuHailortVlmStream *stream,
    uint32_t timeout_ms,
    char **out_token,
    int *out_status);
void yu_hailort_vlm_stream_release(YuHailortVlmStream *stream);

size_t yu_hailort_stub_vdevice_create_count(void);
// Stub-only test seams; stage 1=open, 2=metadata error, 3=configure invalid,
// 4=rejected metadata, 5=configure device error. Events: 1..5=open/meta/config/release/run.
void yu_hailort_stub_wd_fail(int stage);
size_t yu_hailort_stub_wd_count(size_t event);
const char *yu_hailort_stub_wd_calls(void);
void yu_hailort_stub_vdevice_fail_initial(void);
size_t yu_hailort_stub_yolo_create_count(void);
size_t yu_hailort_stub_yolo_release_count(void);
size_t yu_hailort_stub_s2t_create_count(void);
size_t yu_hailort_stub_s2t_release_count(void);
size_t yu_hailort_stub_llm_create_count(void);
size_t yu_hailort_stub_llm_release_count(void);
size_t yu_hailort_stub_llm_clear_context_count(void);
size_t yu_hailort_stub_vlm_create_count(void);
size_t yu_hailort_stub_vlm_release_count(void);
}
