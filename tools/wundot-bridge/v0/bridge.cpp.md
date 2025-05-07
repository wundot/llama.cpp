#include "bridge.h"

#include <ctime>  // Needed for time(NULL)
#include <mutex>
#include <sstream>
#include <string>
#include <thread>  // Needed for std::thread::hardware_concurrency

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
static int                g_n_predict = 128;

extern "C" void * load_model_wrapper(const char * model_path, int n_predict) {
    std::lock_guard<std::mutex> lock(infer_mutex);
    g_n_predict = n_predict;

    common_params params;
    // Set model path correctly
    params.model.path          = model_path;
    // Set context and prediction size
    params.n_ctx               = 2048;
    params.n_predict           = n_predict;
    // Set thread count inside cpuparams
    params.cpuparams.n_threads = std::thread::hardware_concurrency();
    // Set seed inside sampling struct
    params.sampling.seed       = static_cast<uint32_t>(time(NULL));

    // NUMA setup
    llama_backend_init();
    llama_numa_init(params.numa);

    // Initialize model and context
    common_init_result res = common_init_from_params(params);
    model                  = res.model.release();
    ctx                    = res.context.release();

    if (!model || !ctx) {
        return nullptr;
    }

    // Initialize sampler
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
    for (int i = 0; i < g_n_predict; ++i) {
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

    //  persists after function returns
    static std::string output_string;
    //  safe to return pointer
    output_string = output_buffer.str();
    // Go can call C.GoString() on it
    return output_string.c_str();
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
