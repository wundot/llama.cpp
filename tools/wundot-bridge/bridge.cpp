#include "bridge.h"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

static constexpr int DEFAULT_POOL_SIZE = 8;

static int                    g_pool_size = DEFAULT_POOL_SIZE;
static llama_model *          g_model     = nullptr;
static common_params_sampling g_sampling_params;
static common_params          g_common_params;

struct InferenceSession {
    llama_context *  ctx;
    common_sampler * sampler;
};

static std::queue<InferenceSession> g_context_pool;
static std::mutex                   g_pool_mutex;
static std::condition_variable      g_pool_cv;

bool Load_Model(const char * model_path, int n_predict, int context_pool_size) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);

    if (g_model) {
        std::cout << "[LOG] Model already loaded. Skipping reinitialization.\n";
        return true;
    }

    g_pool_size = (context_pool_size > 0 && context_pool_size <= 128) ? context_pool_size : DEFAULT_POOL_SIZE;

    std::cout << "[LOG] Loading model from: " << model_path << "\n";
    std::cout << "[LOG] Allocating " << g_pool_size << " context sessions\n";

    g_common_params            = common_params();
    g_common_params.model.path = model_path;
    g_common_params.n_predict  = n_predict;

    llama_backend_init();
    llama_numa_init(g_common_params.numa);

    auto init = common_init_from_params(g_common_params);
    g_model   = init.model.release();
    if (!g_model) {
        std::cerr << "[ERROR] Failed to load model.\n";
        return false;
    }

    g_sampling_params = g_common_params.sampling;

    for (int i = 0; i < g_pool_size; ++i) {
        llama_context_params ctx_params = llama_context_default_params();
        llama_context *      ctx        = llama_init_from_model(g_model, ctx_params);
        if (!ctx) {
            return false;
        }

        llama_attach_threadpool(ctx, nullptr, nullptr);

        common_sampler * sampler = common_sampler_init(g_model, g_sampling_params);
        if (!sampler) {
            return false;
        }

        g_context_pool.push({ ctx, sampler });
    }

    std::cout << "[LOG] Model loaded and context pool initialized.\n";
    return true;
}

const char * Run_Inference(const char * system_prompt, const char * user_history, const char * current_prompt) {
    return Run_Inference_With_Params(system_prompt, user_history, current_prompt, &g_sampling_params, 128);
}

const char * Run_Inference_With_Params(const char * system_prompt, const char * user_history,
                                       const char * current_prompt, const common_params_sampling * params,
                                       int n_predict) {
    std::ostringstream           output;
    std::vector<common_chat_msg> chat_msgs;

    if (system_prompt && *system_prompt) {
        chat_msgs.push_back({ "system", system_prompt });
    }
    if (user_history && *user_history) {
        chat_msgs.push_back({ "user", user_history });
    }
    if (current_prompt && *current_prompt) {
        chat_msgs.push_back({ "user", current_prompt });
    }

    auto   chat_templates_ptr = common_chat_templates_init(g_model, "");
    auto * chat_templates     = chat_templates_ptr.get();

    common_chat_templates_inputs inputs;
    inputs.messages              = chat_msgs;
    inputs.add_generation_prompt = true;

    std::string formatted_prompt = common_chat_templates_apply(chat_templates, inputs).prompt;

    const llama_vocab * vocab   = llama_model_get_vocab(g_model);
    const bool          add_bos = llama_vocab_get_add_bos(vocab) && !g_common_params.use_jinja;

    std::vector<llama_token> tokens = common_tokenize(session.ctx, formatted_prompt, true, true);
    if (add_bos && !tokens.empty()) {
        tokens.insert(tokens.begin(), llama_vocab_bos(vocab));
    }

    for (llama_token t : tokens) {
        llama_decode(session.ctx, llama_batch_get_one(&t, 1));
    }

    for (int i = 0; i < n_predict; ++i) {
        llama_token id = common_sampler_sample(session.sampler, session.ctx, -1);
        common_sampler_accept(session.sampler, id, true);
        output << common_token_to_piece(session.ctx, id);
        if (llama_vocab_is_eog(llama_model_get_vocab(g_model), id)) {
            break;
        }
        llama_decode(session.ctx, llama_batch_get_one(&id, 1));
    }

    {
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        g_context_pool.push(session);
        g_pool_cv.notify_one();
    }

    static thread_local std::string thread_output;
    thread_output = output.str();

    auto end         = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[PROFILE] Inference took " << duration_ms << " ms\n";

    return thread_output.c_str();
}

void Run_Cleanup() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);

    while (!g_context_pool.empty()) {
        auto session = g_context_pool.front();
        g_context_pool.pop();

        if (session.sampler) {
            common_sampler_free(session.sampler);
        }
        if (session.ctx) {
            llama_free(session.ctx);
        }
    }

    if (g_model) {
        llama_free_model(g_model);
        g_model = nullptr;
    }

    llama_backend_free();
    std::cout << "[LOG] Cleanup complete.\n";
}
