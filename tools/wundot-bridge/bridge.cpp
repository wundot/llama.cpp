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

// Runtime-configurable number of inference sessions (contexts + samplers)
static int g_pool_size = 8;

// Global model handle (loaded once per lifecycle)
static llama_model * g_model = nullptr;

// Sampling parameters used for generation (set externally)
static SamplingParams g_sampling_params;

// Each session holds a decoder context and a sampler instance
struct InferenceSession {
    llama_context *  ctx;
    common_sampler * sampler;
};

// Context pool with thread-safety mechanisms
static std::queue<InferenceSession> g_context_pool;
static std::mutex                   g_pool_mutex;
static std::condition_variable      g_pool_cv;

//
// Load_Model
// --------------------------------------------
// Loads the LLaMA model from the given path, initializes sampling parameters,
// and creates a pool of inference-ready contexts and samplers.
// This function must be called before Run_Inference.
//
bool Load_Model(const char * model_path, int n_predict, int context_pool_size) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);

    if (g_model) {
        std::cout << "[LOG] Model already loaded. Skipping reinitialization.\n";
        return true;
    }

    // Sanitize and set pool size
    if (context_pool_size > 0 && context_pool_size <= 128) {
        g_pool_size = context_pool_size;
    } else {
        g_pool_size = 8;
    }

    std::cout << "[LOG] Loading model from: " << model_path << "\n";
    std::cout << "[LOG] Allocating " << g_pool_size << " context sessions\n";

    // Prepare model loading parameters
    common_params params;
    params.model.path = model_path;
    params.n_predict  = n_predict;

    // Initialize underlying LLaMA runtime
    llama_backend_init();
    llama_numa_init(params.numa);

    // Load model from disk into memory
    auto init = common_init_from_params(params);
    g_model   = init.model.release();
    if (!g_model) {
        std::cerr << "[ERROR] Failed to load model.\n";
        return false;
    }

    // Copy initial sampling parameters into global struct
    g_sampling_params.temp              = params.sampling.temperature;
    g_sampling_params.top_p             = params.sampling.top_p;
    g_sampling_params.top_k             = params.sampling.top_k;
    g_sampling_params.repeat_penalty    = params.sampling.repeat_penalty;
    g_sampling_params.presence_penalty  = params.sampling.presence_penalty;
    g_sampling_params.frequency_penalty = params.sampling.frequency_penalty;
    g_sampling_params.mirostat          = params.sampling.mirostat;
    g_sampling_params.n_predict         = params.n_predict;

    // Preallocate N reusable contexts and samplers into a pool
    for (int i = 0; i < g_pool_size; ++i) {
        llama_context_params ctx_params = llama_context_default_params();
        llama_context *      ctx        = llama_init_from_model(g_model, ctx_params);
        if (!ctx) {
            return false;
        }

        // Convert SamplingParams -> common_params_sampling
        common_params_sampling sampling_config = {};
        sampling_config.temp                   = g_sampling_params.temperature;
        sampling_config.top_p                  = g_sampling_params.top_p;
        sampling_config.top_k                  = g_sampling_params.top_k;
        sampling_config.repeat_penalty         = g_sampling_params.repeat_penalty;
        sampling_config.presence_penalty       = g_sampling_params.presence_penalty;
        sampling_config.frequency_penalty      = g_sampling_params.frequency_penalty;
        sampling_config.mirostat               = g_sampling_params.mirostat;

        common_sampler * sampler = common_sampler_init(g_model, sampling_config);
        if (!sampler) {
            return false;
        }

        g_context_pool.push({ ctx, sampler });
    }

    std::cout << "[LOG] Model loaded and context pool initialized.\n";
    return true;
}

//
// Set_Sampling_Params
// --------------------------------------------
// Replaces the global sampling configuration with a custom struct
// sent from an external caller (e.g., Go via CGO).
//
void Set_Sampling_Params(const SamplingParams * custom_params) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);

    // Convert SamplingParams -> common_params_sampling
    common_params_sampling sampling_config = {};
    sampling_config.temp                   = custom_params->temperature;
    sampling_config.top_p                  = custom_params->top_p;
    sampling_config.top_k                  = custom_params->top_k;
    sampling_config.repeat_penalty         = custom_params->repeat_penalty;
    sampling_config.presence_penalty       = custom_params->presence_penalty;
    sampling_config.frequency_penalty      = custom_params->frequency_penalty;
    sampling_config.mirostat               = custom_params->mirostat;

    // Update global SamplingParams
    g_sampling_params = *custom_params;

    // Optionally reinitialize samplers in the pool
    std::queue<InferenceSession> new_pool;
    while (!g_context_pool.empty()) {
        auto session = g_context_pool.front();
        g_context_pool.pop();

        if (session.sampler) {
            common_sampler_free(session.sampler);
        }
        session.sampler = common_sampler_init(g_model, sampling_config);
        new_pool.push(session);
    }
    g_context_pool = new_pool;

    std::cout << "[LOG] Sampling parameters updated at runtime.\n";
}

//
// Run_Inference
// --------------------------------------------
// Performs text generation based on a system prompt, optional prior messages,
// and a current user input prompt.
// Returns the generated output as a C-compatible string (valid until next thread call).
//
const char * Run_Inference(const char * system_prompt, const char * user_history, const char * current_prompt) {
    if (!g_model) {
        return "ERROR_MODEL_NOT_LOADED";
    }

    // Start profiling timer
    auto start = std::chrono::high_resolution_clock::now();

    InferenceSession session;

    {
        // Wait for and acquire a reusable context from the pool
        std::unique_lock<std::mutex> lock(g_pool_mutex);
        g_pool_cv.wait(lock, [] { return !g_context_pool.empty(); });

        session = g_context_pool.front();
        g_context_pool.pop();
        std::cout << "[LOG] Context acquired. Remaining in pool: " << g_context_pool.size() << "/" << g_pool_size
                  << "\n";
    }

    // Convert SamplingParams -> common_params_sampling
    common_params_sampling sampling_config = {};
    sampling_config.temp                   = g_sampling_params.temperature;
    sampling_config.top_p                  = g_sampling_params.top_p;
    sampling_config.top_k                  = g_sampling_params.top_k;
    sampling_config.repeat_penalty         = g_sampling_params.repeat_penalty;
    sampling_config.presence_penalty       = g_sampling_params.presence_penalty;
    sampling_config.frequency_penalty      = g_sampling_params.frequency_penalty;
    sampling_config.mirostat               = g_sampling_params.mirostat;

    // Reset the sampler to use the latest sampling parameters
    if (session.sampler) {
        common_sampler_free(session.sampler);
    }
    session.sampler = common_sampler_init(g_model, sampling_config);

    // Assemble chat message sequence
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

    // Apply chat formatting (OpenAI-style)
    auto   chat_templates_ptr = common_chat_templates_init(g_model, "");
    auto * chat_templates     = chat_templates_ptr.get();

    common_chat_templates_inputs inputs;
    inputs.messages              = chat_msgs;
    inputs.add_generation_prompt = true;

    // Format into a full prompt string
    std::string formatted_prompt = common_chat_templates_apply(chat_templates, inputs).prompt;

    // Tokenize the prompt and feed it to the model context
    std::vector<llama_token> tokens = common_tokenize(session.ctx, formatted_prompt, true, true);
    for (llama_token t : tokens) {
        llama_decode(session.ctx, llama_batch_get_one(&t, 1));
    }

    // Perform token-by-token generation
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
        // Return the session to the pool
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        g_context_pool.push(session);
        g_pool_cv.notify_one();
        std::cout << "[LOG] Context released back to pool. Pool size: " << g_context_pool.size() << "/" << g_pool_size
                  << "\n";
    }

    // Output per-thread string safely
    static thread_local std::string thread_output;
    thread_output = output.str();

    // End profiling
    auto end         = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[PROFILE] Inference took " << duration_ms << " ms\n";

    return thread_output.c_str();
}

//
// Run_Cleanup
// --------------------------------------------
// Frees all context memory, samplers, and shuts down LLaMA backend.
//
void Run_Cleanup() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    std::cout << "[LOG] Cleaning up context pool and model resources...\n";

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

//
// Load_Anchor_Persona
// --------------------------------------------
// Legacy compatibility stub — currently unused.
// Always returns true if both prompts are non-null.
//
bool Load_Anchor_Persona(const char * system_prompt, const char * user_prompt) {
    return system_prompt && user_prompt;
}
