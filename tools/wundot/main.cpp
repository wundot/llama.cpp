#include <sstream>
#include <string>
#include <vector>

#include "app_context.h"
#include "arg.h"
#include "chat.h"
#include "chat_state.h"
#include "common.h"
#include "console.h"
#include "generation_loop.h"
#include "llama.h"
#include "llama_runtime.h"
#include "log.h"
#include "sampling.h"
#include "session_io.h"
#include "signal_handler.h"

static void print_usage(int argc, char ** argv) {
    (void) argc;
    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128 -no-cnv\n",
        argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -sys \"You are a helpful assistant\"\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    // Set up global params
    common_params params;
    app::g_params = &params;
    app::initialize_globals();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    common_init();
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.logits_all) {
        LOG_ERR("%s: please use the 'perplexity' tool for logits_all\n", __func__);
        return 0;
    }

    if (params.embedding) {
        LOG_ERR("%s: please use the 'embedding' tool for embeddings\n", __func__);
        return 0;
    }

    // Warn and fix invalid context values
    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: minimum context size is 8, using minimum.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: RoPE base set to %g\n", __func__, params.rope_freq_base);
    }
    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: RoPE scale set to %g\n", __func__, params.rope_freq_scale);
    }

    LOG_INF("%s: llama backend init\n", __func__);

    // Load model and attach threadpool
    llama_model *            model            = nullptr;
    llama_context *          ctx              = nullptr;
    struct ggml_threadpool * threadpool       = nullptr;
    struct ggml_threadpool * threadpool_batch = nullptr;

    if (!llama_runtime::initialize_backend(params, model, ctx, threadpool, threadpool_batch)) {
        return 1;
    }

    app::g_model = &model;
    app::g_ctx   = &ctx;

    // Create sampler
    common_sampler * smpl = common_sampler_init(model, params.sampling);
    if (!smpl) {
        LOG_ERR("Failed to initialize sampler\n");
        return 1;
    }
    app::g_smpl = &smpl;

    // Load chat template
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto                chat_templates =
        std::shared_ptr<common_chat_templates>(common_chat_templates_init(model, params.chat_template).release());
    std::vector<common_chat_msg> chat_msgs;

    // Format initial prompt using template
    std::string prompt;
    if (params.conversation_mode && params.enable_chat_template) {
        if (!params.system_prompt.empty()) {
            chat_state::format_system_prompt(params.system_prompt, chat_msgs, chat_templates, params.use_jinja);
        }
        if (!params.prompt.empty()) {
            chat_state::format_user_prompt(params.prompt, chat_msgs, chat_templates, params.use_jinja);
        }
        if (!params.system_prompt.empty() || !params.prompt.empty()) {
            prompt = chat_state::apply_chat_template(chat_msgs, chat_templates, !params.prompt.empty());
        }
    } else {
        prompt = params.prompt;
    }

    // Tokenize or load session
    std::vector<llama_token> session_tokens;
    if (!params.path_prompt_cache.empty()) {
        if (session_io::file_exists(params.path_prompt_cache) && !session_io::file_is_empty(params.path_prompt_cache)) {
            LOG_INF("Loading session from %s\n", params.path_prompt_cache.c_str());
            if (!session_io::load_session(ctx, params.path_prompt_cache, session_tokens)) {
                LOG_ERR("Failed to load session\n");
                return 1;
            }
        } else {
            LOG_INF("Session file not found or empty, starting new session\n");
        }
    }

    // Setup Ctrl+C signal handler
    platform::setup_sigint_handler();

    LOG_INF("Starting generation loop...\n");

    // Run modular generation loop
    generation::run_generation_loop(ctx, model, smpl, *app::g_input_tokens, session_tokens, *app::g_output_ss, prompt,
                                    &chat_msgs, chat_templates);

    // Save session if needed
    if (!params.path_prompt_cache.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        session_io::save_session(ctx, params.path_prompt_cache, session_tokens);
    }

    common_perf_print(ctx, smpl);
    common_sampler_free(smpl);
    llama_backend_free();
    llama_runtime::free_threadpools(threadpool, threadpool_batch);

    return 0;
}
