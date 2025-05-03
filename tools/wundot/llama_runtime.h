#pragma once

#include <memory>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

namespace llama {

class Runtime {
  public:
    Runtime();
    ~Runtime();

    bool parse_args(int argc, char ** argv);
    void initialize();
    void load_model();
    void setup_chat_context();
    void prepare_prompt();
    void run_loop();
    void shutdown();

  private:
    std::unique_ptr<llama_model, decltype(&llama_free)>             model_;
    std::unique_ptr<llama_context, decltype(&llama_free)>           context_;
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler_;

    common_params            params_;
    std::vector<llama_token> input_tokens_;
    std::vector<llama_token> output_tokens_;
    std::ostringstream       output_ss_;

    std::unique_ptr<common_chat_templates, decltype(&common_chat_templates_free)> chat_templates_;
    std::vector<common_chat_msg>                                                  chat_msgs_;

    bool is_interacting_          = false;
    bool need_insert_eot_         = false;
    bool waiting_for_first_input_ = false;

    void handle_signals();
};

}  // namespace llama
