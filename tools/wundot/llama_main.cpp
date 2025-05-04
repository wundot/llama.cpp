#include "llama_main.h"

#include <sstream>
#include <string>
#include <vector>

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "console.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

namespace llama {

LlamaApp::LlamaApp(int argc, char ** argv) : argc_(argc), argv_(argv) {}

int LlamaApp::run() {
    if (!parse_args()) {
        return 1;
    }
    init_console();
    validate_params();
    init_backend();
    if (!load_model()) {
        return 1;
    }
    init_threadpool();
    attach_threadpool();
    if (!prepare_prompt()) {
        return 1;
    }
    init_sampler();
    run_loop();
    save_session();
    shutdown();
    return 0;
}

bool LlamaApp::parse_args() {
    return common_params_parse(argc_, argv_, params_, LLAMA_EXAMPLE_MAIN, print_usage);
}

void LlamaApp::init_console() {
    console::init(params_.simple_io, params_.use_color);
    atexit([] { console::cleanup(); });
}

void LlamaApp::validate_params() {
    if (params_.logits_all) {
        LOG_ERR("logits_all not supported in this mode\n");
        exit(0);
    }
    if (params_.embedding) {
        LOG_ERR("embedding not supported in this mode\n");
        exit(0);
    }
    if (params_.n_ctx != 0 && params_.n_ctx < 8) {
        LOG_WRN("Minimum context size is 8. Using 8.\n");
        params_.n_ctx = 8;
    }
}

void LlamaApp::init_backend() {
    llama_backend_init();
    llama_numa_init(params_.numa);
}

bool LlamaApp::load_model() {
    auto res = common_init_from_params(params_);
    model_   = res.model.release();
    ctx_     = res.context.release();
    return model_ != nullptr;
}

void LlamaApp::init_threadpool() {
    ggml_threadpool_params tpp = ggml_threadpool_params_from_cpu_params(params_.cpuparams);
    threadpool_                = ggml_threadpool_new(&tpp);
    if (!threadpool_) {
        LOG_ERR("Failed to initialize threadpool\n");
        exit(1);
    }
}

void LlamaApp::attach_threadpool() {
    llama_attach_threadpool(ctx_, threadpool_, nullptr);
}

bool LlamaApp::prepare_prompt() {
    if (params_.prompt.empty()) {
        LOG_ERR("Empty prompt\n");
        return false;
    }
    input_tokens_ = common_tokenize(ctx_, params_.prompt, true, true);
    return !input_tokens_.empty();
}

void LlamaApp::init_sampler() {
    sampler_ = common_sampler_init(model_, params_.sampling);
    if (!sampler_) {
        LOG_ERR("Failed to initialize sampler\n");
        exit(1);
    }
}

void LlamaApp::run_loop() {
    LOG("Starting generation...\n");
    for (llama_token tok : input_tokens_) {
        LOG("Token: %d -> %s\n", tok, common_token_to_piece(ctx_, tok).c_str());
    }
}

void LlamaApp::save_session() {
    if (!params_.path_prompt_cache.empty()) {
        llama_state_save_file(ctx_, params_.path_prompt_cache.c_str(), input_tokens_.data(), input_tokens_.size());
    }
}

void LlamaApp::shutdown() {
    if (ctx_) {
        llama_free(ctx_);
    }
    if (model_) {
        llama_free_model(model_);
    }
    if (sampler_) {
        common_sampler_free(sampler_);
    }
    if (threadpool_) {
        ggml_threadpool_free(threadpool_);
    }
    llama_backend_free();
}

void LlamaApp::print_usage(int, char **) {
    LOG("Usage: llama_main [options]\n");
}

}  // namespace llama
