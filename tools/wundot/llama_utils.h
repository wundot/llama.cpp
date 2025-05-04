#pragma once

#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;
struct common_sampler;
struct sampling_params;
struct common_chat_templates;
struct common_chat_msg;

namespace llama {

class TokenSampler {
  public:
    TokenSampler(llama_model * model, const sampling_params & params);
    ~TokenSampler();

    int  sample(llama_context * ctx);
    void accept(int token, bool use_grammar);

  private:
    common_sampler * sampler_;
};

class SessionCache {
  public:
    bool load(const std::string & path, llama_context * ctx, std::vector<int> & tokens);
    bool save(const std::string & path, llama_context * ctx, const std::vector<int> & tokens);
};

class ChatFormatter {
  public:
    ChatFormatter(llama_model * model, const std::string & template_path);
    std::string format_user(const std::string & prompt);
    std::string format_system(const std::string & prompt);

  private:
    std::unique_ptr<common_chat_templates, void (*)(common_chat_templates *)> templates_;
    std::vector<common_chat_msg>                                              messages_;
};

}  // namespace llama
