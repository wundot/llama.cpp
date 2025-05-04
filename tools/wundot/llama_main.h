#pragma once

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

class LlamaApp {
  public:
    LlamaApp(int argc, char ** argv);
    ~LlamaApp();

    int run();

  private:
    bool        parse_args();
    static void print_usage(int, char **);
    void        init_console();
    void        validate_params();
    void        init_backend();
    bool        load_model();
    void        init_threadpool();
    void        attach_threadpool();
    bool        prepare_prompt();
    void        init_sampler();
    void        run_loop();
    void        save_session();
    void        shutdown();

  private:
    int     argc_;
    char ** argv_;

    common_params    params_;
    llama_model *    model_   = nullptr;
    llama_context *  ctx_     = nullptr;
    common_sampler * sampler_ = nullptr;

    ggml_threadpool * threadpool_       = nullptr;
    ggml_threadpool * threadpool_batch_ = nullptr;

    std::vector<llama_token>     embd_inp_;
    std::vector<llama_token>     session_tokens_;
    std::vector<common_chat_msg> chat_msgs_;

    std::vector<int>   input_tokens_;
    std::vector<int>   output_tokens_;
    std::ostringstream output_ss_;
    std::ostringstream assistant_ss_;

    bool is_interacting_          = false;
    bool need_insert_eot_         = false;
    bool waiting_for_first_input_ = false;

    std::unique_ptr<common_chat_templates> chat_templates_;
    const llama_vocab *                    vocab_ = nullptr;
};

}  // namespace llama
