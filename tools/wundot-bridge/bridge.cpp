#include "bridge.h"

#include <mutex>
#include <sstream>
#include <string>

#include "chat.h"
#include "common.h"
#include "console.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

static llama_context *    ctx     = nullptr;
static llama_model *      model   = nullptr;
static common_sampler *   sampler = nullptr;
static std::ostringstream output_buffer;
static std::mutex         infer_mutex;

extern "C" void * load_model_wrapper(const char * model_path) {
    std::lock_guard<std::mutex> lock(infer_mutex);

    common_params params;
    params.model     = std::string(model_path);
    params.n_ctx     = 2048;
    params.seed      = time(NULL);
    params.n_threads = std::thread::hardware_concurrency();
    params.n_predict = 128;

    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result res = common_init_from_params(params);
    model                  = res.model.release();
    ctx                    = res.context.release();

    if (!model || !ctx) {
        return nullptr;
    }

    common_sampler * smpl = common_sampler_init(model, params.sampling);
    if (!smpl) {
        return nullptr;
    }
    sampler = smpl;

    return static_cast<void *>(ctx);
}

extern "C" const char * run_inferance_wrapper(const char * prompt) {
    std::lock_guard<std::mutex> lock(infer_mutex);

    if (!ctx || !model || !sampler) {
        return "Model not initialized.";
    }

    output_buffer.str("");
    output_buffer.clear();

    std::vector<llama_token> input_tokens = common_tokenize(ctx, prompt, true, true);
    if (input_tokens.empty()) {
        return "Failed to tokenize input.";
    }

    const int n_ctx  = llama_n_ctx(ctx);
    int       n_past = 0;

    // Evaluate input tokens
    if (llama_decode(ctx, llama_batch_get_one(input_tokens.data(), input_tokens.size())) != 0) {
        return "Failed to evaluate input tokens.";
    }
    n_past += input_tokens.size();

    // Generate tokens
    for (int i = 0; i < 128; ++i) {
        llama_token id = common_sampler_sample(sampler, ctx, -1);
        common_sampler_accept(sampler, id, true);

        if (llama_vocab_is_eog(llama_model_get_vocab(model), id)) {
            break;
        }

        const std::string piece = common_token_to_piece(ctx, id, false);
        output_buffer << piece;

        if (llama_decode(ctx, llama_batch_get_one(&id, 1)) != 0) {
            break;
        }
        ++n_past;
    }

    return output_buffer.str().c_str();  // lifetime is static via ostringstream
}

extern "C" void run_cleanup_wrapper() {
    std::lock_guard<std::mutex> lock(infer_mutex);

    if (sampler) {
        common_sampler_free(sampler);
        sampler = nullptr;
    }

    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }

    if (model) {
        llama_free_model(model);
        model = nullptr;
    }

    llama_backend_free();
}
