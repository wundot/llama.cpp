
#include "bridge.h"

#include <mutex>
#include <string>

#include "chat.h"
#include "llama.h"

static llama_model *   global_model = nullptr;
static llama_context * global_ctx   = nullptr;
static std::mutex      model_mutex;

extern "C" void * load_model_wrapper(const char * model_path, int n_predict) {
    std::lock_guard<std::mutex> lock(model_mutex);

    if (!global_model) {
        common_params params;
        params.model.path = model_path;
        params.n_predict  = n_predict;

        global_model = llama_load_model_from_file(params.model.path.c_str());
        global_ctx   = llama_new_context_with_model(global_model);
    }

    return global_model;
}

extern "C" const char * run_inferance_wrapper(const char * prompt) {
    static thread_local std::ostringstream output_buffer;
    output_buffer.str("");
    output_buffer.clear();

    if (!global_model) {
        return "ERROR_MODEL_NOT_LOADED";
    }

    llama_context_params lparams = llama_context_default_params();
    lparams.n_ctx                = 2048;
    llama_context * ctx          = llama_new_context_with_model(global_model, lparams);
    if (!ctx) {
        return "ERROR_CONTEXT_CREATION_FAILED";
    }

    common_sampler * sampler = create_default_sampler(global_model);

    std::vector<llama_token> input_tokens = common_tokenize(ctx, prompt, true, true);
    if (input_tokens.empty()) {
        llama_free(ctx);
        common_sampler_free(sampler);
        return "ERROR_TOKENIZATION_FAILED";
    }

    llama_decode(ctx, llama_batch_get_one(input_tokens.data(), input_tokens.size()));
    int n_past = input_tokens.size();

    for (int i = 0; i < global_n_predict; ++i) {
        llama_token id = common_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(llama_model_get_vocab(global_model), id)) {
            break;
        }
        common_sampler_accept(sampler, id, true);
        output_buffer << common_token_to_piece(ctx, id, false);
        llama_decode(ctx, llama_batch_get_one(&id, 1));
        ++n_past;
    }

    static thread_local std::string output_string;
    output_string = output_buffer.str();

    common_sampler_free(sampler);
    llama_free(ctx);

    return output_string.c_str();
}

extern "C" const char * run_infer_with_sampling(const char * prompt, SamplingParams params) {
    std::lock_guard<std::mutex> lock(model_mutex);

    if (!global_ctx) {
        return "Error: Model not loaded.";
    }

    gpt_params gparams             = gpt_params_default();
    gparams.prompt                 = std::string(prompt);
    gparams.n_predict              = params.n_predict;
    gparams.sparams.temp           = params.temperature;
    gparams.sparams.top_k          = params.top_k;
    gparams.sparams.top_p          = params.top_p;
    gparams.sparams.repeat_penalty = params.repeat_penalty;

    static std::string output;
    output = run_chat(global_ctx, gparams);  // You must implement run_chat using llama.cpp API

    return output.c_str();
}

extern "C" void run_cleanup_wrapper() {
    std::lock_guard<std::mutex> lock(model_mutex);
    if (global_ctx) {
        llama_free(global_ctx);
    }
    if (global_model) {
        llama_free_model(global_model);
    }
    global_ctx   = nullptr;
    global_model = nullptr;
}
