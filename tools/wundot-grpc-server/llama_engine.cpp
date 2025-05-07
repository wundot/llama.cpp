#include "llama_engine.h"

#include <mutex>
#include <sstream>
#include <vector>

#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

static llama_model *    model   = nullptr;
static llama_context *  ctx     = nullptr;
static common_sampler * sampler = nullptr;
static std::mutex       llama_mutex;

namespace llama_engine {

void init(const std::string & model_path) {
    static bool initialized = false;
    if (initialized) {
        return;
    }

    common_params params;
    params.model     = model_path;
    params.n_ctx     = 2048;
    params.seed      = time(NULL);
    params.n_threads = std::thread::hardware_concurrency();

    llama_backend_init();
    common_init_result res = common_init_from_params(params);

    model   = res.model.release();
    ctx     = res.context.release();
    sampler = common_sampler_init(model, params.sampling);

    initialized = true;
    LOG("Model loaded from %s\n", model_path.c_str());
}

std::string generate_response(const std::string & prompt, int max_tokens) {
    std::lock_guard<std::mutex> lock(llama_mutex);

    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true, true);

    int                n_past   = 0;
    int                n_remain = max_tokens;
    std::ostringstream oss;

    while (n_remain > 0) {
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
            throw std::runtime_error("llama_decode() failed");
        }

        llama_token id = common_sampler_sample(sampler, ctx, -1);
        common_sampler_accept(sampler, id, true);

        std::string piece = common_token_to_piece(ctx, id);
        oss << piece;
        tokens = { id };

        --n_remain;
        ++n_past;

        if (llama_vocab_is_eog(llama_model_get_vocab(model), id)) {
            break;
        }
    }

    return oss.str();
}

}  // namespace llama_engine
