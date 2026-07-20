// Stub shim — compiled on hosts without hailo/hailort.hpp (WSL2, CI, non-Pi).
// All functions return a sentinel error; the real shim is used on Hailo hardware.
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

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
};

struct YuHailortYoloMetadata {
    size_t inputs_count;
    size_t outputs_count;
    YuHailortTensorInfo inputs[40];
    YuHailortTensorInfo outputs[40];
};

struct YuHailortBuffer {
    const char *name;
    void *data;
    size_t size;
};

struct YuHailortYolo {};
struct YuHailortSpeech2Text {};
struct YuHailortLlm {};
struct YuHailortLlmStream {};
struct YuHailortVlm {};
struct YuHailortVlmStream {};

static const int HAILO_STUB_ERR = 1; // HAILO_NOT_INITIALIZED equivalent

int yu_hailort_yolo_create(const char *, YuHailortYolo **) { return HAILO_STUB_ERR; }
void yu_hailort_yolo_release(YuHailortYolo *) {}
int yu_hailort_yolo_metadata(const YuHailortYolo *, YuHailortYoloMetadata *) { return HAILO_STUB_ERR; }
int yu_hailort_yolo_run(YuHailortYolo *, const uint8_t *, size_t, YuHailortBuffer *, size_t, uint32_t) { return HAILO_STUB_ERR; }

void yu_hailort_string_free(char *) {}

int yu_hailort_s2t_create(const char *, YuHailortSpeech2Text **) { return HAILO_STUB_ERR; }
void yu_hailort_s2t_release(YuHailortSpeech2Text *) {}
int yu_hailort_s2t_generate_text(YuHailortSpeech2Text *, const float *, size_t, int, const char *, float, uint32_t, char **) { return HAILO_STUB_ERR; }
int yu_hailort_s2t_tokenize(YuHailortSpeech2Text *, const char *, int *, size_t *) { return HAILO_STUB_ERR; }

int yu_hailort_llm_create(const char *, const char *, bool, YuHailortLlm **) { return HAILO_STUB_ERR; }
void yu_hailort_llm_release(YuHailortLlm *) {}
int yu_hailort_llm_generate_text(YuHailortLlm *, const char *, uint32_t, char **) { return HAILO_STUB_ERR; }
int yu_hailort_llm_tokenize(YuHailortLlm *, const char *, int *, size_t *) { return HAILO_STUB_ERR; }
int yu_hailort_llm_context_usage(YuHailortLlm *, size_t *) { return HAILO_STUB_ERR; }
int yu_hailort_llm_max_context_capacity(YuHailortLlm *, size_t *) { return HAILO_STUB_ERR; }
int yu_hailort_llm_clear_context(YuHailortLlm *) { return HAILO_STUB_ERR; }
int yu_hailort_llm_generate_stream_start(YuHailortLlm *, const char *const *, size_t, const float *, const float *, const uint32_t *, const float *, const uint32_t *, const bool *, const uint32_t *, YuHailortLlmStream **) { return HAILO_STUB_ERR; }
int yu_hailort_llm_stream_read(YuHailortLlmStream *, uint32_t, char **, int *) { return HAILO_STUB_ERR; }
void yu_hailort_llm_stream_release(YuHailortLlmStream *) {}

int yu_hailort_vlm_create(const char *, bool, YuHailortVlm **) { return HAILO_STUB_ERR; }
void yu_hailort_vlm_release(YuHailortVlm *) {}
int yu_hailort_vlm_generate_text(YuHailortVlm *, const char *, const uint8_t *const *, const size_t *, size_t, uint32_t, char **) { return HAILO_STUB_ERR; }
int yu_hailort_vlm_tokenize(YuHailortVlm *, const char *, int *, size_t *) { return HAILO_STUB_ERR; }
int yu_hailort_vlm_context_usage(YuHailortVlm *, size_t *) { return HAILO_STUB_ERR; }
int yu_hailort_vlm_max_context_capacity(YuHailortVlm *, size_t *) { return HAILO_STUB_ERR; }
int yu_hailort_vlm_clear_context(YuHailortVlm *) { return HAILO_STUB_ERR; }
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
