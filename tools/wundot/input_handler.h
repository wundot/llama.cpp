#ifndef INPUT_HANDLER_H
#define INPUT_HANDLER_H

#include <string>
#include <vector>

#include "common.h"
#include "llama.h"

class InputHandler {
  public:
    static std::vector<llama_token> tokenize_input(llama_context * ctx, const std::string & input, bool add_bos);
    static void                     handle_interactive_mode(llama_context * ctx, std::vector<llama_token> & embd_inp,
                                                            bool & is_interacting);
    static std::string              format_chat_message(const std::string & role, const std::string & content);
};

#endif  // INPUT_HANDLER_H
