#include "llama_runtime.h"

#include <csignal>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "arg.h"
#include "console.h"
#include "log.h"

namespace llama {

Runtime::Runtime() :
    model_(nullptr, llama_free),
    context_(nullptr, llama_free),
    sampler_(nullptr, common_sampler_free),
    chat_templates_(nullptr, common_chat_templates_free) {}

Runtime::~Runtime() {
    shutdown();
}

bool Runtime::parse_args(int argc, char ** argv) {
    return common_params_parse(argc, argv, params_, LLAMA_EXAMPLE_MAIN, nullptr);
}

void Runtime::initialize() {
    llama_backend_init();
    llama_numa_init(params_.numa);
    console::init(params_.simple_io, params_.use_color);
    atexit([]() { console::cleanup(); });
    handle_signals();
}

void Runtime::load_model() {
    common_init_result result = common_init_from_params(params_);
    if (!result.model || !result.context) {
        throw std::runtime_error("Failed to load model or context.");
    }
    model_.reset(result.model.release());
    context_.reset(result.context.release());
    sampler_.reset(common_sampler_init(model_.get(), params_.sampling));
    if (!sampler_) {
        throw std::runtime_error("Failed to initialize sampler.");
    }
}

void Runtime::setup_chat_context() {
    chat_templates_.reset(common_chat_templates_init(model_.get(), params_.chat_template).release());
}

void Runtime::prepare_prompt() {
    if (params_.prompt.empty()) {
        return;
    }
    input_tokens_ = common_tokenize(context_.get(), params_.prompt, true, true);
    if (input_tokens_.empty()) {
        throw std::runtime_error("Prompt tokenization failed.");
    }
}

void Runtime::run_loop() {
    int                      n_past   = 0;
    int                      n_remain = params_.n_predict;
    std::vector<llama_token> embd;

    while (n_remain > 0) {
        if (!embd.empty()) {
            if (llama_decode(context_.get(), llama_batch_get_one(embd.data(), embd.size()))) {
                throw std::runtime_error("llama_decode failed");
            }
            n_past += embd.size();
            embd.clear();
        }

        llama_token next = common_sampler_sample(sampler_.get(), context_.get(), -1);
        common_sampler_accept(sampler_.get(), next, true);
        output_tokens_.push_back(next);
        output_ss_ << common_token_to_piece(context_.get(), next);
        std::cout << common_token_to_piece(context_.get(), next);

        embd.push_back(next);
        --n_remain;
    }
    std::cout << std::endl;
}

void Runtime::shutdown() {
    llama_backend_free();
    console::cleanup();
}

void Runtime::handle_signals() {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sa;
    sa.sa_handler = [](int signo) {
        if (signo == SIGINT) {
            std::exit(130);
        }
    };
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
#elif defined(_WIN32)
    SetConsoleCtrlHandler(
        [](DWORD type) -> BOOL {
            if (type == CTRL_C_EVENT) {
                std::exit(130);
            }
            return FALSE;
        },
        TRUE);
#endif
}

}