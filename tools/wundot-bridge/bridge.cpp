#include "bridge.h"

#include <condition_variable>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "xsampling.h"
// ======== Forward Declarations for Local Profile Functions ========
sampling_params CreativeProfile();
sampling_params BalancedProfile();
sampling_params ConservativeProfile();
sampling_params FraudDetectionProfile();

constexpr int MAX_CONTEXT_POOL_SIZE = 8;

static llama_model *   g_model = nullptr;
static sampling_params g_sampling_params;

struct InferenceSession {
    llama_context *  ctx;
    common_sampler * sampler;
};

static std::queue<InferenceSession> g_context_pool;
static std::mutex                   g_pool_mutex;
static std::condition_variable      g_pool_cv;

// ======== Profile Definitions ========

sampling_params CreativeProfile() {
    sampling_params p = g_sampling_params;
    p.temp            = 0.9f;
    p.top_p           = 0.95f;
    p.top_k           = 40;
    p.repeat_penalty  = 1.0f;
    return p;
}

sampling_params BalancedProfile() {
    sampling_params p = g_sampling_params;
    p.temp            = 0.7f;
    p.top_p           = 0.9f;
    p.top_k           = 50;
    p.repeat_penalty  = 1.1f;
    return p;
}

sampling_params ConservativeProfile() {
    sampling_params p = g_sampling_params;
    p.temp            = 0.5f;
    p.top_p           = 0.85f;
    p.top_k           = 20;
    p.repeat_penalty  = 1.2f;
    return p;
}

sampling_params FraudDetectionProfile() {
    sampling_params p   = g_sampling_params;
    p.temp              = 0.35f;
    p.top_p             = 0.8f;
    p.top_k             = 25;
    p.repeat_penalty    = 1.35f;
    p.presence_penalty  = 0.25f;
    p.frequency_penalty = 0.15f;
    p.mirostat          = 0;
    return p;
}

extern "C" sampling_params Get_FraudDetection_Params(const char * profile_name) {
    std::string mode = profile_name ? profile_name : "balanced";

    if (mode == "strict") {
        sampling_params p = FraudDetectionProfile();
        p.temp            = 0.3f;
        p.top_p           = 0.75f;
        p.top_k           = 20;
        p.repeat_penalty  = 1.4f;
        return p;
    } else if (mode == "sensitive") {
        sampling_params p   = FraudDetectionProfile();
        p.presence_penalty  = 0.4f;
        p.frequency_penalty = 0.3f;
        return p;
    } else {
        return FraudDetectionProfile();
    }
}

// ======== Model Lifecycle ========

bool Load_Model(const char * model_path, int n_predict) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    if (g_model) {
        return true;
    }

    common_params params;
    params.model.path = model_path;
    params.n_predict  = n_predict;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    g_model   = init.model.release();
    if (!g_model) {
        return false;
    }

    // ✅ Field-by-field copy from common_params::sampling to sampling_params
    g_sampling_params.temp              = params.sampling.temp;
    g_sampling_params.top_p             = params.sampling.top_p;
    g_sampling_params.top_k             = params.sampling.top_k;
    g_sampling_params.repeat_penalty    = params.sampling.repeat_penalty;
    g_sampling_params.presence_penalty  = params.sampling.presence_penalty;
    g_sampling_params.frequency_penalty = params.sampling.frequency_penalty;
    g_sampling_params.mirostat          = params.sampling.mirostat;
    g_sampling_params.n_predict         = params.n_predict;

    for (int i = 0; i < MAX_CONTEXT_POOL_SIZE; ++i) {
        llama_context_params ctx_params = llama_context_default_params();
        llama_context *      ctx        = llama_init_from_model(g_model, ctx_params);
        if (!ctx) {
            return false;
        }

        common_sampler * sampler = common_sampler_init(g_model, g_sampling_params);
        if (!sampler) {
            return false;
        }

        g_context_pool.push({ ctx, sampler });
    }

    return true;
}

void Set_Sampling_Params(const sampling_params * custom_params) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_sampling_params = *custom_params;
}

const char * Run_Inference(const char * system_prompt, const char * user_history, const char * current_prompt) {
    if (!g_model) {
        return "ERROR_MODEL_NOT_LOADED";
    }

    InferenceSession session;

    {
        std::unique_lock<std::mutex> lock(g_pool_mutex);
        g_pool_cv.wait(lock, [] { return !g_context_pool.empty(); });

        session = g_context_pool.front();
        g_context_pool.pop();
    }

    if (session.sampler) {
        common_sampler_free(session.sampler);
    }
    session.sampler = common_sampler_init(g_model, g_sampling_params);

    std::ostringstream           output;
    std::vector<common_chat_msg> chat_msgs;

    if (system_prompt && strlen(system_prompt) > 0) {
        chat_msgs.push_back({ "system", system_prompt });
    }
    if (user_history && strlen(user_history) > 0) {
        chat_msgs.push_back({ "user", user_history });
    }
    if (current_prompt && strlen(current_prompt) > 0) {
        chat_msgs.push_back({ "user", current_prompt });
    }

    auto   chat_templates_ptr = common_chat_templates_init(g_model, "");
    auto * chat_templates     = chat_templates_ptr.get();

    common_chat_templates_inputs inputs;
    inputs.messages              = chat_msgs;
    inputs.add_generation_prompt = true;

    std::string              formatted_prompt = common_chat_templates_apply(chat_templates, inputs).prompt;
    std::vector<llama_token> tokens           = common_tokenize(session.ctx, formatted_prompt, true, true);

    for (llama_token t : tokens) {
        llama_decode(session.ctx, llama_batch_get_one(&t, 1));
    }

    for (int i = 0; i < g_sampling_params.n_predict; ++i) {
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
    }
    g_pool_cv.notify_one();

    static thread_local std::string thread_output;
    thread_output = output.str();
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
}

bool Load_Anchor_Persona(const char * system_prompt, const char * user_prompt) {
    return system_prompt && user_prompt;
}
