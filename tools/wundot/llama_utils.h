// llama_utils.cpp
#include <fstream>
#include <stdexcept>

#include "llama_utils.h"
#include "log.h"

namespace llama {

// --- TokenSampler ---

TokenSampler::TokenSampler(llama_model * model, const sampling_params & params) {
    sampler_ = common_sampler_init(model, params);
    if (!sampler_) {
        throw std::runtime_error("Failed to initialize token sampler");
    }
}

TokenSampler::~TokenSampler() {
    common_sampler_free(sampler_);
}

llama_token TokenSampler::sample(llama_context * ctx) {
    return common_sampler_sample(sampler_, ctx, -1);
}

void TokenSampler::accept(llama_token token, bool use_grammar) {
    common_sampler_accept(sampler_, token, use_grammar);
}

// --- SessionCache ---

bool SessionCache::load(const std::string & path, llama_context * ctx, std::vector<llama_token> & tokens) {
    if (path.empty()) {
        return false;
    }
    std::ifstream f(path.c_str(), std::ios::binary | std::ios::ate);
    if (!f.good() || f.tellg() == 0) {
        return false;
    }

    tokens.resize(llama_n_ctx(ctx));
    size_t out_count = 0;
    bool   ok        = llama_state_load_file(ctx, path.c_str(), tokens.data(), tokens.size(), &out_count);
    tokens.resize(out_count);
    return ok;
}

bool SessionCache::save(const std::string & path, llama_context * ctx, const std::vector<llama_token> & tokens) {
    if (path.empty()) {
        return false;
    }
    return llama_state_save_file(ctx, path.c_str(), tokens.data(), tokens.size());
}

// --- ChatFormatter ---

ChatFormatter::ChatFormatter(llama_model * model, const std::string & template_path) :
    templates_(common_chat_templates_init(model, template_path).release(), common_chat_templates_free) {}

std::string ChatFormatter::format_user(const std::string & prompt) {
    common_chat_msg msg{ "user", prompt };
    messages_.push_back(msg);
    return common_chat_format_single(templates_.get(), messages_, msg, true, false);
}

std::string ChatFormatter::format_system(const std::string & prompt) {
    common_chat_msg msg{ "system", prompt };
    messages_.push_back(msg);
    return common_chat_format_single(templates_.get(), messages_, msg, false, false);
}

}  // namespace llama
