// llama_wrapper.cpp

#include "llama_wrapper.h"

#include <mutex>
#include <sstream>
#include <string>

#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

static std::mutex g_mutex;

struct LlamaSession {
    llama_model *    model;
    llama_context *  ctx;
    common_sampler * sampler;
    common_params    params;
};

extern "C" {

// Initialize the model and return a session pointer
void * llama_init_model(const char * model_path) {
    std::lock_guard<std::mutex> lock(g_mutex);

    auto * session = new LlamaSession();

    // Set basic parameters
    session->params.model          = model_path;
    session->params.seed           = time(NULL);
    session->params.n_threads      = 4;  // adjust as needed
    session->params.n_ctx          = 512;
    session->params.n_predict      = 128;
    session->params.sampling.top_k = 40;
    session->params.sampling.top_p = 0.9f;
    session->params.sampling.temp  = 0.8f;

    // Init backend and model
    llama_backend_init();
    llama_numa_init(false);

    common_init_result result = common_init_from_params(session->params);
    if (!result.model || !result.context) {
        delete session;
        return nullptr;
    }

    session->model = result.model.release();
    session->ctx   = result.context.release();

    session->sampler = common_sampler_init(session->model, session->params.sampling);
    if (!session->sampler) {
        llama_free(session->ctx);
        llama_free_model(session->model);
        delete session;
        return nullptr;
    }

    return session;
}

// Generate output from a prompt
const char * llama_generate(void * ctx_ptr, const char * prompt_cstr) {
    std::lock_guard<std::mutex> lock(g_mutex);

    static thread_local std::ostringstream output;
    output.str("");  // Clear previous content

    auto * session = static_cast<LlamaSession *>(ctx_ptr);
    if (!session) {
        return nullptr;
    }

    std::string              prompt(prompt_cstr);
    std::vector<llama_token> tokens = common_tokenize(session->ctx, prompt, true, true);

    if (tokens.empty()) {
        return nullptr;
    }

    int n_past   = 0;
    int n_remain = session->params.n_predict;

    std::vector<llama_token> embd = tokens;

    while (n_remain > 0) {
        if (llama_decode(session->ctx, llama_batch_get_one(embd.data(), embd.size()))) {
            return nullptr;
        }

        embd.clear();

        llama_token id = common_sampler_sample(session->sampler, session->ctx, -1);
        common_sampler_accept(session->sampler, id, true);

        embd.push_back(id);
        output << common_token_to_piece(session->ctx, id);

        --n_remain;

        if (llama_vocab_is_eog(llama_model_get_vocab(session->model), id)) {
            break;
        }
    }

    return output.str().c_str();
}

// Free resources
void llama_free_model(void * ctx_ptr) {
    std::lock_guard<std::mutex> lock(g_mutex);

    auto * session = static_cast<LlamaSession *>(ctx_ptr);
    if (!session) {
        return;
    }

    if (session->sampler) {
        common_sampler_free(session->sampler);
    }
    if (session->ctx) {
        llama_free(session->ctx);
    }
    if (session->model) {
        llama_free_model(session->model);
    }

    llama_backend_free();

    delete session;
}
}
