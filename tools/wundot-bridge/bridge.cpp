#include "bridge.h"

#include <ctime>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "chat.h"
#include "common.h"
#include "console.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

// Shared model instance
static llama_model * global_model = nullptr;
static std::mutex    global_model_mutex;
static int           global_n_predict = 110000;

struct InferenceSession {
    llama_context *          ctx;
    common_sampler *         sampler;
    std::vector<llama_token> embd_inp;
    int                      n_past;
    std::string              token_buffer;
};

extern "C" void * load_model_wrapper(const char * model_path, int n_predict) {
    std::lock_guard<std::mutex> lock(global_model_mutex);

    if (global_model != nullptr) {
        return global_model;
    }
    global_n_predict = n_predict;

    common_params params;
    params.model.path          = model_path;
    params.n_ctx               = 2048;
    params.n_predict           = n_predict;
    params.cpuparams.n_threads = std::thread::hardware_concurrency();
    params.sampling.seed       = static_cast<uint32_t>(time(NULL));

    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result res = common_init_from_params(params);
    global_model           = res.model.release();
    return global_model;
}

extern "C" void run_cleanup_wrapper() {
    std::lock_guard<std::mutex> lock(global_model_mutex);
    if (global_model) {
        llama_free_model(global_model);
        global_model = nullptr;
        llama_backend_free();
    }
}

extern "C" const char * run_inferance_wrapper(const char * prompt) {
    static thread_local std::ostringstream output_buffer;
    output_buffer.str("");
    output_buffer.clear();

    if (!global_model) {
        return "Model not loaded.";
    }

    llama_context_params lparams = llama_context_default_params();
    lparams.n_ctx                = 2048;
    llama_context * ctx          = llama_new_context_with_model(global_model, lparams);
    if (!ctx) {
        return "Failed to create context.";
    }

    common_sampler * sampler = common_sampler_init(global_model, common_sampling_params());

    std::vector<llama_token> input_tokens = common_tokenize(ctx, prompt, true, true);
    if (input_tokens.empty()) {
        return "Tokenization failed.";
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

extern "C" InferenceSession * session_create(const char * model_path, int n_predict) {
    load_model_wrapper(model_path, n_predict);
    InferenceSession *   session = new InferenceSession();
    llama_context_params lparams = llama_context_default_params();
    lparams.n_ctx                = 2048;
    session->ctx                 = llama_new_context_with_model(global_model, lparams);
    session->sampler             = common_sampler_init(global_model, common_sampling_params());
    session->n_past              = 0;
    return session;
}

extern "C" void session_free(InferenceSession * session) {
    if (!session) {
        return;
    }
    if (session->sampler) {
        common_sampler_free(session->sampler);
    }
    if (session->ctx) {
        llama_free(session->ctx);
    }
    delete session;
}

extern "C" void session_start_stream(InferenceSession * session, const char * prompt) {
    session->embd_inp = common_tokenize(session->ctx, prompt, true, true);
    llama_decode(session->ctx, llama_batch_get_one(session->embd_inp.data(), session->embd_inp.size()));
    session->n_past = session->embd_inp.size();
}

extern "C" const char * session_next_token(InferenceSession * session) {
    llama_token id = common_sampler_sample(session->sampler, session->ctx, -1);
    if (llama_vocab_is_eog(llama_model_get_vocab(global_model), id)) {
        return nullptr;
    }

    common_sampler_accept(session->sampler, id, true);
    llama_decode(session->ctx, llama_batch_get_one(&id, 1));
    session->n_past++;

    session->token_buffer = common_token_to_piece(session->ctx, id);
    return session->token_buffer.c_str();
}
