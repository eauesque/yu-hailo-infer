#include "hailo/hailort.hpp"
#include "shim.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct YuHailortYolo {
    std::shared_ptr<hailort::VDevice> vdevice;
    std::shared_ptr<hailort::InferModel> infer_model;
    std::unique_ptr<hailort::ConfiguredInferModel> configured;
    std::vector<YuHailortTensorInfo> inputs;
    std::vector<YuHailortTensorInfo> outputs;
};

struct YuHailortSpeech2Text {
    std::shared_ptr<hailort::VDevice> vdevice;
    std::unique_ptr<hailort::genai::Speech2Text> speech2text;
};

struct YuHailortLlm {
    std::shared_ptr<hailort::VDevice> vdevice;
    std::unique_ptr<hailort::genai::LLM> llm;
};

// Keeps the LLMGenerator alive alongside its LLMGeneratorCompletion, same
// requirement as YuHailortVlmStream.
struct YuHailortLlmStream {
    std::unique_ptr<hailort::genai::LLMGenerator> generator;
    std::unique_ptr<hailort::genai::LLMGeneratorCompletion> completion;
};

struct YuHailortVlm {
    std::shared_ptr<hailort::VDevice> vdevice;
    std::unique_ptr<hailort::genai::VLM> vlm;
};

// Keeps the VLMGenerator alive alongside its LLMGeneratorCompletion: the SDK
// requires the generator to outlive the completion object it produced.
struct YuHailortVlmStream {
    std::unique_ptr<hailort::genai::VLMGenerator> generator;
    std::unique_ptr<hailort::genai::LLMGeneratorCompletion> completion;
};

static std::string &vdevice_group_id()
{
    static std::string group_id = "YU_SHARED";
    return group_id;
}

int yu_hailort_set_vdevice_group_id(const char *group_id)
{
    if (nullptr == group_id) {
        return HAILO_INVALID_ARGUMENT;
    }
    vdevice_group_id() = group_id;
    return HAILO_SUCCESS;
}

static std::shared_ptr<hailort::VDevice> shared_vdevice(hailo_status &status)
{
    static hailo_status creation_status = HAILO_SUCCESS;
    static std::shared_ptr<hailort::VDevice> vdevice = []() {
        auto params = hailort::HailoRTDefaults::get_vdevice_params();
        params.group_id = vdevice_group_id().c_str();
        auto created = hailort::VDevice::create_shared(params);
        if (!created) {
            creation_status = created.status();
            return std::shared_ptr<hailort::VDevice>();
        }
        return created.release();
    }();
    status = creation_status;
    return vdevice;
}

static std::string escape_json_string(const std::string &input)
{
    std::string out;
    out.reserve(input.size() + 8);
    for (unsigned char c : input) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

static char *copy_string_to_c(const std::string &value)
{
    auto out = static_cast<char *>(std::malloc(value.size() + 1));
    if (nullptr == out) {
        return nullptr;
    }
    std::memcpy(out, value.data(), value.size());
    out[value.size()] = '\0';
    return out;
}

static void copy_name(char (&dst)[128], const std::string &name)
{
    std::memset(dst, 0, sizeof(dst));
    std::strncpy(dst, name.c_str(), sizeof(dst) - 1);
}

static YuHailortTensorInfo tensor_info_from_stream(const hailort::InferModel::InferStream &stream)
{
    YuHailortTensorInfo info {};
    copy_name(info.name, stream.name());
    auto shape = stream.shape();
    auto format = stream.format();
    auto quant_infos = stream.get_quant_infos();
    info.height = shape.height;
    info.width = shape.width;
    info.features = shape.features;
    info.format_type = static_cast<uint32_t>(format.type);
    info.frame_size = stream.get_frame_size();
    if (!quant_infos.empty()) {
        info.qp_zp = quant_infos[0].qp_zp;
        info.qp_scale = quant_infos[0].qp_scale;
    } else {
        info.qp_zp = 0.0f;
        info.qp_scale = 1.0f;
    }
    return info;
}

static hailo_status fill_metadata(YuHailortYolo *ctx)
{
    ctx->inputs.clear();
    ctx->outputs.clear();

    auto inputs = ctx->infer_model->inputs();
    auto outputs = ctx->infer_model->outputs();
    if (inputs.size() > 40 || outputs.size() > 40) {
        return HAILO_INVALID_ARGUMENT;
    }

    for (const auto &input : inputs) {
        ctx->inputs.push_back(tensor_info_from_stream(input));
    }
    for (const auto &output : outputs) {
        ctx->outputs.push_back(tensor_info_from_stream(output));
    }
    return HAILO_SUCCESS;
}

int yu_hailort_yolo_create(const char *hef_path, YuHailortYolo **out)
{
    if ((nullptr == hef_path) || (nullptr == out)) {
        return HAILO_INVALID_ARGUMENT;
    }
    *out = nullptr;

    hailo_status vdevice_status;
    auto vdevice = shared_vdevice(vdevice_status);
    if (!vdevice) {
        return vdevice_status;
    }

    auto infer_model = vdevice->create_infer_model(std::string(hef_path));
    if (!infer_model) {
        return infer_model.status();
    }

    auto configured = infer_model.value()->configure();
    if (!configured) {
        return configured.status();
    }

    auto ctx = std::make_unique<YuHailortYolo>();
    ctx->vdevice = vdevice;
    ctx->infer_model = infer_model.release();
    ctx->configured = std::make_unique<hailort::ConfiguredInferModel>(configured.release());
    auto status = fill_metadata(ctx.get());
    if (HAILO_SUCCESS != status) {
        return status;
    }

    *out = ctx.release();
    return HAILO_SUCCESS;
}

void yu_hailort_yolo_release(YuHailortYolo *ctx)
{
    delete ctx;
}

int yu_hailort_yolo_metadata(const YuHailortYolo *ctx, YuHailortYoloMetadata *metadata)
{
    if ((nullptr == ctx) || (nullptr == metadata)) {
        return HAILO_INVALID_ARGUMENT;
    }
    std::memset(metadata, 0, sizeof(*metadata));
    metadata->inputs_count = ctx->inputs.size();
    metadata->outputs_count = ctx->outputs.size();
    for (size_t i = 0; i < ctx->inputs.size(); ++i) {
        metadata->inputs[i] = ctx->inputs[i];
    }
    for (size_t i = 0; i < ctx->outputs.size(); ++i) {
        metadata->outputs[i] = ctx->outputs[i];
    }
    return HAILO_SUCCESS;
}

int yu_hailort_yolo_run(
    YuHailortYolo *ctx,
    const uint8_t *input,
    size_t input_size,
    YuHailortBuffer *outputs,
    size_t outputs_count,
    uint32_t timeout_ms)
{
    if ((nullptr == ctx) || (nullptr == input) || (nullptr == outputs)) {
        return HAILO_INVALID_ARGUMENT;
    }
    if ((ctx->inputs.size() != 1) || (outputs_count != ctx->outputs.size())) {
        return HAILO_INVALID_ARGUMENT;
    }
    if (input_size != ctx->inputs[0].frame_size) {
        return HAILO_INVALID_ARGUMENT;
    }

    std::map<std::string, hailort::MemoryView> buffers;
    buffers.emplace(
        std::string(ctx->inputs[0].name),
        hailort::MemoryView(const_cast<uint8_t *>(input), input_size));

    for (size_t i = 0; i < outputs_count; ++i) {
        if ((nullptr == outputs[i].data) || (outputs[i].size != ctx->outputs[i].frame_size)) {
            return HAILO_INVALID_ARGUMENT;
        }
        buffers.emplace(
            std::string(ctx->outputs[i].name),
            hailort::MemoryView(outputs[i].data, outputs[i].size));
    }

    auto bindings = ctx->configured->create_bindings(buffers);
    if (!bindings) {
        return bindings.status();
    }

    return ctx->configured->run(bindings.value(), std::chrono::milliseconds(timeout_ms));
}

void yu_hailort_string_free(char *value)
{
    std::free(value);
}

int yu_hailort_s2t_create(const char *model_path, YuHailortSpeech2Text **out)
{
    if ((nullptr == model_path) || (nullptr == out)) {
        return HAILO_INVALID_ARGUMENT;
    }
    *out = nullptr;

    hailo_status vdevice_status;
    auto vdevice = shared_vdevice(vdevice_status);
    if (!vdevice) {
        return vdevice_status;
    }

    auto params = hailort::genai::Speech2TextParams(std::string_view(model_path));
    auto speech2text = hailort::genai::Speech2Text::create(vdevice, params);
    if (!speech2text) {
        return speech2text.status();
    }

    auto ctx = std::make_unique<YuHailortSpeech2Text>();
    ctx->vdevice = vdevice;
    ctx->speech2text = std::make_unique<hailort::genai::Speech2Text>(speech2text.release());
    *out = ctx.release();
    return HAILO_SUCCESS;
}

void yu_hailort_s2t_release(YuHailortSpeech2Text *ctx)
{
    delete ctx;
}

int yu_hailort_s2t_generate_text(
    YuHailortSpeech2Text *ctx,
    const float *audio,
    size_t audio_count,
    int task,
    const char *language,
    float repetition_penalty,
    uint32_t timeout_ms,
    char **out_text)
{
    if ((nullptr == ctx) || (nullptr == audio) || (nullptr == out_text)) {
        return HAILO_INVALID_ARGUMENT;
    }
    *out_text = nullptr;

    auto params = ctx->speech2text->create_generator_params();
    if (!params) {
        return params.status();
    }
    auto generator_params = params.release();
    auto status = generator_params.set_task(static_cast<hailort::genai::Speech2TextTask>(task));
    if (HAILO_SUCCESS != status) {
        return status;
    }
    if ((nullptr != language) && ('\0' != language[0])) {
        status = generator_params.set_language(std::string_view(language));
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }
    status = generator_params.set_repetition_penalty(repetition_penalty);
    if (HAILO_SUCCESS != status) {
        return status;
    }

    auto audio_view = hailort::MemoryView(
        const_cast<float *>(audio),
        audio_count * sizeof(float));
    auto text = ctx->speech2text->generate_all_text(
        audio_view,
        generator_params,
        std::chrono::milliseconds(timeout_ms));
    if (!text) {
        return text.status();
    }
    *out_text = copy_string_to_c(text.release());
    return (nullptr == *out_text) ? HAILO_OUT_OF_HOST_MEMORY : HAILO_SUCCESS;
}

int yu_hailort_s2t_generate_segments(
    YuHailortSpeech2Text *ctx,
    const float *audio,
    size_t audio_count,
    int task,
    const char *language,
    float repetition_penalty,
    uint32_t timeout_ms,
    char **out_json)
{
    if ((nullptr == ctx) || (nullptr == audio) || (nullptr == out_json)) {
        return HAILO_INVALID_ARGUMENT;
    }
    *out_json = nullptr;

    auto params = ctx->speech2text->create_generator_params();
    if (!params) {
        return params.status();
    }
    auto generator_params = params.release();
    auto status = generator_params.set_task(static_cast<hailort::genai::Speech2TextTask>(task));
    if (HAILO_SUCCESS != status) {
        return status;
    }
    if ((nullptr != language) && ('\0' != language[0])) {
        status = generator_params.set_language(std::string_view(language));
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }
    status = generator_params.set_repetition_penalty(repetition_penalty);
    if (HAILO_SUCCESS != status) {
        return status;
    }

    auto audio_view = hailort::MemoryView(
        const_cast<float *>(audio),
        audio_count * sizeof(float));
    auto segments = ctx->speech2text->generate_all_segments(
        audio_view,
        generator_params,
        std::chrono::milliseconds(timeout_ms));
    if (!segments) {
        return segments.status();
    }

    std::string json = "[";
    bool first = true;
    for (const auto &segment : segments.value()) {
        if (!first) {
            json += ",";
        }
        first = false;
        char bounds[64];
        std::snprintf(bounds, sizeof(bounds), "{\"start_sec\":%.6f,\"end_sec\":%.6f,",
            static_cast<double>(segment.start_sec), static_cast<double>(segment.end_sec));
        json += bounds;
        json += "\"text\":\"" + escape_json_string(segment.text) + "\"}";
    }
    json += "]";

    *out_json = copy_string_to_c(json);
    return (nullptr == *out_json) ? HAILO_OUT_OF_HOST_MEMORY : HAILO_SUCCESS;
}

int yu_hailort_s2t_tokenize(
    YuHailortSpeech2Text *ctx,
    const char *text,
    int *tokens,
    size_t *tokens_count)
{
    if ((nullptr == ctx) || (nullptr == text) || (nullptr == tokens_count)) {
        return HAILO_INVALID_ARGUMENT;
    }
    auto expected = ctx->speech2text->tokenize(std::string(text));
    if (!expected) {
        return expected.status();
    }
    auto values = expected.release();
    if ((nullptr == tokens) || (*tokens_count < values.size())) {
        *tokens_count = values.size();
        return HAILO_INSUFFICIENT_BUFFER;
    }
    std::memcpy(tokens, values.data(), values.size() * sizeof(int));
    *tokens_count = values.size();
    return HAILO_SUCCESS;
}

int yu_hailort_llm_create(
    const char *model_path,
    const char *lora_name,
    bool optimize_memory_on_device,
    YuHailortLlm **out)
{
    if ((nullptr == model_path) || (nullptr == out)) {
        return HAILO_INVALID_ARGUMENT;
    }
    *out = nullptr;

    hailo_status vdevice_status;
    auto vdevice = shared_vdevice(vdevice_status);
    if (!vdevice) {
        return vdevice_status;
    }
    auto params = hailort::genai::LLMParams(
        std::string(model_path),
        (nullptr == lora_name) ? std::string() : std::string(lora_name),
        optimize_memory_on_device);
    auto llm = hailort::genai::LLM::create(vdevice, params);
    if (!llm) {
        return llm.status();
    }

    auto ctx = std::make_unique<YuHailortLlm>();
    ctx->vdevice = vdevice;
    ctx->llm = std::make_unique<hailort::genai::LLM>(llm.release());
    *out = ctx.release();
    return HAILO_SUCCESS;
}

void yu_hailort_llm_release(YuHailortLlm *ctx)
{
    delete ctx;
}

int yu_hailort_llm_generate_text(
    YuHailortLlm *ctx,
    const char *prompt,
    uint32_t timeout_ms,
    char **out_text)
{
    if ((nullptr == ctx) || (nullptr == prompt) || (nullptr == out_text)) {
        return HAILO_INVALID_ARGUMENT;
    }
    *out_text = nullptr;

    auto params = ctx->llm->create_generator_params();
    if (!params) {
        return params.status();
    }
    auto generator = ctx->llm->create_generator(params.value());
    if (!generator) {
        return generator.status();
    }
    auto status = generator->write(std::string(prompt));
    if (HAILO_SUCCESS != status) {
        return status;
    }
    auto completion = generator->generate();
    if (!completion) {
        return completion.status();
    }
    auto text = completion->read_all(std::chrono::milliseconds(timeout_ms));
    if (!text) {
        return text.status();
    }
    *out_text = copy_string_to_c(text.release());
    return (nullptr == *out_text) ? HAILO_OUT_OF_HOST_MEMORY : HAILO_SUCCESS;
}

int yu_hailort_llm_tokenize(
    YuHailortLlm *ctx,
    const char *text,
    int *tokens,
    size_t *tokens_count)
{
    if ((nullptr == ctx) || (nullptr == text) || (nullptr == tokens_count)) {
        return HAILO_INVALID_ARGUMENT;
    }
    auto expected = ctx->llm->tokenize(std::string(text));
    if (!expected) {
        return expected.status();
    }
    auto values = expected.release();
    if ((nullptr == tokens) || (*tokens_count < values.size())) {
        *tokens_count = values.size();
        return HAILO_INSUFFICIENT_BUFFER;
    }
    std::memcpy(tokens, values.data(), values.size() * sizeof(int));
    *tokens_count = values.size();
    return HAILO_SUCCESS;
}

int yu_hailort_llm_context_usage(YuHailortLlm *ctx, size_t *out)
{
    if ((nullptr == ctx) || (nullptr == out)) {
        return HAILO_INVALID_ARGUMENT;
    }
    auto expected = ctx->llm->get_context_usage_size();
    if (!expected) {
        return expected.status();
    }
    *out = expected.release();
    return HAILO_SUCCESS;
}

int yu_hailort_llm_max_context_capacity(YuHailortLlm *ctx, size_t *out)
{
    if ((nullptr == ctx) || (nullptr == out)) {
        return HAILO_INVALID_ARGUMENT;
    }
    auto expected = ctx->llm->max_context_capacity();
    if (!expected) {
        return expected.status();
    }
    *out = expected.release();
    return HAILO_SUCCESS;
}

int yu_hailort_llm_clear_context(YuHailortLlm *ctx)
{
    if (nullptr == ctx) {
        return HAILO_INVALID_ARGUMENT;
    }
    return ctx->llm->clear_context();
}

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
    YuHailortLlmStream **out)
{
    if ((nullptr == ctx) || (nullptr == out) || (0 == messages_count)
        || (nullptr == messages_json) || ((0 != tools_count) && (nullptr == tools_json))) {
        return HAILO_INVALID_ARGUMENT;
    }
    *out = nullptr;

    // Each entry is a caller-built (and caller-escaped, via Rust's
    // serde_json) {"role":...,"content":...} JSON string. write() applies
    // the model's chat template across the whole ordered exchange, so a
    // full multi-turn conversation can be expressed here, not just one turn.
    std::vector<std::string> messages;
    messages.reserve(messages_count);
    for (size_t i = 0; i < messages_count; ++i) {
        if (nullptr == messages_json[i]) {
            return HAILO_INVALID_ARGUMENT;
        }
        messages.emplace_back(messages_json[i]);
    }

    // Each entry is a caller-built {"name":...,"description":...,"parameters":...}
    // JSON string, per the SDK's write(prompt_json_strings, tools_json_strings)
    // contract. The SDK docs state tools may only be provided on a fresh
    // context — the caller (this shim's callers all clear_context() before
    // every generation) guarantees that, so no fresh-context check is done here.
    std::vector<std::string> tools;
    tools.reserve(tools_count);
    for (size_t i = 0; i < tools_count; ++i) {
        if (nullptr == tools_json[i]) {
            return HAILO_INVALID_ARGUMENT;
        }
        tools.emplace_back(tools_json[i]);
    }

    auto params = ctx->llm->create_generator_params();
    if (!params) {
        return params.status();
    }
    auto generator_params = params.release();
    if (nullptr != temperature) {
        auto status = generator_params.set_temperature(*temperature);
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }
    if (nullptr != top_p) {
        auto status = generator_params.set_top_p(*top_p);
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }
    if (nullptr != top_k) {
        auto status = generator_params.set_top_k(*top_k);
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }
    // Same HailoRT quirk as the VLM path: an explicit 0.0 frequency_penalty
    // is rejected with HAILO_INVALID_ARGUMENT, so skip the setter in that
    // case (0.0 is equivalent to the model's own "no penalty" default).
    if ((nullptr != frequency_penalty) && (0.0f != *frequency_penalty)) {
        auto status = generator_params.set_frequency_penalty(*frequency_penalty);
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }
    if (nullptr != max_generated_tokens) {
        auto status = generator_params.set_max_generated_tokens(*max_generated_tokens);
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }
    if (nullptr != do_sample) {
        auto status = generator_params.set_do_sample(*do_sample);
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }
    if (nullptr != seed) {
        auto status = generator_params.set_seed(*seed);
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }

    auto generator = ctx->llm->create_generator(generator_params);
    if (!generator) {
        return generator.status();
    }
    auto generator_ptr = std::make_unique<hailort::genai::LLMGenerator>(generator.release());
    auto write_status = generator_ptr->write(messages, tools);
    if (HAILO_SUCCESS != write_status) {
        return write_status;
    }
    auto completion = generator_ptr->generate();
    if (!completion) {
        return completion.status();
    }

    auto stream = std::make_unique<YuHailortLlmStream>();
    stream->generator = std::move(generator_ptr);
    stream->completion =
        std::make_unique<hailort::genai::LLMGeneratorCompletion>(completion.release());
    *out = stream.release();
    return HAILO_SUCCESS;
}

int yu_hailort_llm_stream_read(
    YuHailortLlmStream *stream,
    uint32_t timeout_ms,
    char **out_token,
    int *out_status)
{
    if ((nullptr == stream) || (nullptr == out_token) || (nullptr == out_status)) {
        return HAILO_INVALID_ARGUMENT;
    }
    *out_token = nullptr;

    auto token = stream->completion->read(std::chrono::milliseconds(timeout_ms));
    *out_status = static_cast<int>(stream->completion->generation_status());
    if (!token) {
        return token.status();
    }
    *out_token = copy_string_to_c(token.release());
    return (nullptr == *out_token) ? HAILO_OUT_OF_HOST_MEMORY : HAILO_SUCCESS;
}

void yu_hailort_llm_stream_release(YuHailortLlmStream *stream)
{
    delete stream;
}

int yu_hailort_vlm_create(
    const char *model_path,
    bool optimize_memory_on_device,
    YuHailortVlm **out)
{
    if ((nullptr == model_path) || (nullptr == out)) {
        return HAILO_INVALID_ARGUMENT;
    }
    *out = nullptr;

    hailo_status vdevice_status;
    auto vdevice = shared_vdevice(vdevice_status);
    if (!vdevice) {
        return vdevice_status;
    }
    auto params = hailort::genai::VLMParams(std::string(model_path), optimize_memory_on_device);
    auto vlm = hailort::genai::VLM::create(vdevice, params);
    if (!vlm) {
        return vlm.status();
    }

    auto ctx = std::make_unique<YuHailortVlm>();
    ctx->vdevice = vdevice;
    ctx->vlm = std::make_unique<hailort::genai::VLM>(vlm.release());
    *out = ctx.release();
    return HAILO_SUCCESS;
}

void yu_hailort_vlm_release(YuHailortVlm *ctx)
{
    delete ctx;
}

int yu_hailort_vlm_generate_text(
    YuHailortVlm *ctx,
    const char *prompt,
    const uint8_t *const *frames,
    const size_t *frame_sizes,
    size_t frame_count,
    uint32_t timeout_ms,
    char **out_text)
{
    if ((nullptr == ctx) || (nullptr == prompt) || (nullptr == out_text)) {
        return HAILO_INVALID_ARGUMENT;
    }
    if ((frame_count > 0) && ((nullptr == frames) || (nullptr == frame_sizes))) {
        return HAILO_INVALID_ARGUMENT;
    }
    *out_text = nullptr;

    std::vector<hailort::MemoryView> input_frames;
    input_frames.reserve(frame_count);
    for (size_t i = 0; i < frame_count; ++i) {
        if (nullptr == frames[i]) {
            return HAILO_INVALID_ARGUMENT;
        }
        input_frames.emplace_back(const_cast<uint8_t *>(frames[i]), frame_sizes[i]);
    }

    // The plain-string generate() overload does not apply the model's chat
    // template, so the image placeholder token is never inserted and
    // generation fails with HAILO_INVALID_OPERATION. Build a structured
    // message (one "image" content entry per frame, matching the Python
    // extension's _build_vlm_prompt()) and use the messages_json overload.
    std::string message = "{\"role\":\"user\",\"content\":[";
    for (size_t i = 0; i < frame_count; ++i) {
        message += "{\"type\":\"image\"},";
    }
    message += "{\"type\":\"text\",\"text\":\"" + escape_json_string(std::string(prompt)) + "\"}]}";
    std::vector<std::string> messages { message };

    auto params = ctx->vlm->create_generator_params();
    if (!params) {
        return params.status();
    }
    auto generator = ctx->vlm->create_generator(params.value());
    if (!generator) {
        return generator.status();
    }
    auto completion = generator->generate(messages, input_frames);
    if (!completion) {
        return completion.status();
    }
    auto text = completion->read_all(std::chrono::milliseconds(timeout_ms));
    if (!text) {
        return text.status();
    }
    *out_text = copy_string_to_c(text.release());
    return (nullptr == *out_text) ? HAILO_OUT_OF_HOST_MEMORY : HAILO_SUCCESS;
}

int yu_hailort_vlm_tokenize(
    YuHailortVlm *ctx,
    const char *text,
    int *tokens,
    size_t *tokens_count)
{
    if ((nullptr == ctx) || (nullptr == text) || (nullptr == tokens_count)) {
        return HAILO_INVALID_ARGUMENT;
    }
    auto expected = ctx->vlm->tokenize(std::string(text));
    if (!expected) {
        return expected.status();
    }
    auto values = expected.release();
    if ((nullptr == tokens) || (*tokens_count < values.size())) {
        *tokens_count = values.size();
        return HAILO_INSUFFICIENT_BUFFER;
    }
    std::memcpy(tokens, values.data(), values.size() * sizeof(int));
    *tokens_count = values.size();
    return HAILO_SUCCESS;
}

int yu_hailort_vlm_context_usage(YuHailortVlm *ctx, size_t *out)
{
    if ((nullptr == ctx) || (nullptr == out)) {
        return HAILO_INVALID_ARGUMENT;
    }
    auto expected = ctx->vlm->get_context_usage_size();
    if (!expected) {
        return expected.status();
    }
    *out = expected.release();
    return HAILO_SUCCESS;
}

int yu_hailort_vlm_max_context_capacity(YuHailortVlm *ctx, size_t *out)
{
    if ((nullptr == ctx) || (nullptr == out)) {
        return HAILO_INVALID_ARGUMENT;
    }
    auto expected = ctx->vlm->max_context_capacity();
    if (!expected) {
        return expected.status();
    }
    *out = expected.release();
    return HAILO_SUCCESS;
}

int yu_hailort_vlm_clear_context(YuHailortVlm *ctx)
{
    if (nullptr == ctx) {
        return HAILO_INVALID_ARGUMENT;
    }
    return ctx->vlm->clear_context();
}

int yu_hailort_vlm_input_frame_info(
    YuHailortVlm *ctx,
    uint32_t *frame_size,
    uint32_t *height,
    uint32_t *width,
    uint32_t *features,
    uint32_t *format_type,
    uint32_t *format_order)
{
    if ((nullptr == ctx) || (nullptr == frame_size) || (nullptr == height) || (nullptr == width)
        || (nullptr == features) || (nullptr == format_type) || (nullptr == format_order)) {
        return HAILO_INVALID_ARGUMENT;
    }
    *frame_size = ctx->vlm->input_frame_size();
    auto shape = ctx->vlm->input_frame_shape();
    *height = shape.height;
    *width = shape.width;
    *features = shape.features;
    *format_type = static_cast<uint32_t>(ctx->vlm->input_frame_format_type());
    *format_order = static_cast<uint32_t>(ctx->vlm->input_frame_format_order());
    return HAILO_SUCCESS;
}

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
    YuHailortVlmStream **out)
{
    if ((nullptr == ctx) || (nullptr == prompt) || (nullptr == out)) {
        return HAILO_INVALID_ARGUMENT;
    }
    if ((frame_count > 0) && ((nullptr == frames) || (nullptr == frame_sizes))) {
        return HAILO_INVALID_ARGUMENT;
    }
    *out = nullptr;

    std::vector<hailort::MemoryView> input_frames;
    input_frames.reserve(frame_count);
    for (size_t i = 0; i < frame_count; ++i) {
        if (nullptr == frames[i]) {
            return HAILO_INVALID_ARGUMENT;
        }
        input_frames.emplace_back(const_cast<uint8_t *>(frames[i]), frame_sizes[i]);
    }

    // Same structured-message requirement as yu_hailort_vlm_generate_text():
    // the plain-string overload skips the chat template and drops the image
    // placeholder token, so generation fails with HAILO_INVALID_OPERATION.
    std::vector<std::string> messages;
    if (nullptr != system_prompt) {
        messages.push_back(
            "{\"role\":\"system\",\"content\":[{\"type\":\"text\",\"text\":\""
            + escape_json_string(std::string(system_prompt)) + "\"}]}");
    }
    std::string message = "{\"role\":\"user\",\"content\":[";
    for (size_t i = 0; i < frame_count; ++i) {
        message += "{\"type\":\"image\"},";
    }
    message += "{\"type\":\"text\",\"text\":\"" + escape_json_string(std::string(prompt)) + "\"}]}";
    messages.push_back(message);

    auto params = ctx->vlm->create_generator_params();
    if (!params) {
        return params.status();
    }
    auto generator_params = params.release();
    if (nullptr != temperature) {
        auto status = generator_params.set_temperature(*temperature);
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }
    if (nullptr != top_p) {
        auto status = generator_params.set_top_p(*top_p);
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }
    if (nullptr != top_k) {
        auto status = generator_params.set_top_k(*top_k);
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }
    // HailoRT rejects an explicit 0.0 frequency_penalty with
    // HAILO_INVALID_ARGUMENT (confirmed on real hardware); 0.0 means "no
    // penalty", which is already the model's default, so skipping the
    // setter call in that case is behavior-preserving.
    if ((nullptr != frequency_penalty) && (0.0f != *frequency_penalty)) {
        auto status = generator_params.set_frequency_penalty(*frequency_penalty);
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }
    if (nullptr != max_generated_tokens) {
        auto status = generator_params.set_max_generated_tokens(*max_generated_tokens);
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }
    if (nullptr != do_sample) {
        auto status = generator_params.set_do_sample(*do_sample);
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }
    if (nullptr != seed) {
        auto status = generator_params.set_seed(*seed);
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }

    auto generator = ctx->vlm->create_generator(generator_params);
    if (!generator) {
        return generator.status();
    }
    auto generator_ptr = std::make_unique<hailort::genai::VLMGenerator>(generator.release());
    auto completion = generator_ptr->generate(messages, input_frames);
    if (!completion) {
        return completion.status();
    }

    auto stream = std::make_unique<YuHailortVlmStream>();
    stream->generator = std::move(generator_ptr);
    stream->completion =
        std::make_unique<hailort::genai::LLMGeneratorCompletion>(completion.release());
    *out = stream.release();
    return HAILO_SUCCESS;
}

int yu_hailort_vlm_stream_read(
    YuHailortVlmStream *stream,
    uint32_t timeout_ms,
    char **out_token,
    int *out_status)
{
    if ((nullptr == stream) || (nullptr == out_token) || (nullptr == out_status)) {
        return HAILO_INVALID_ARGUMENT;
    }
    *out_token = nullptr;

    auto token = stream->completion->read(std::chrono::milliseconds(timeout_ms));
    *out_status = static_cast<int>(stream->completion->generation_status());
    if (!token) {
        return token.status();
    }
    *out_token = copy_string_to_c(token.release());
    return (nullptr == *out_token) ? HAILO_OUT_OF_HOST_MEMORY : HAILO_SUCCESS;
}

void yu_hailort_vlm_stream_release(YuHailortVlmStream *stream)
{
    delete stream;
}
