#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "llama.h"
#include "sampling.h"

namespace generation {

void run_generation_loop(llama_context * ctx, llama_model * model, common_sampler * smpl,
                         std::vector<llama_token> & input_tokens, std::vector<llama_token> & session_tokens,
                         std::ostringstream & output_stream, std::string initial_prompt = "",
                         std::vector<common_chat_msg> *         chat_history   = nullptr,
                         std::shared_ptr<common_chat_templates> chat_templates = nullptr);

}
