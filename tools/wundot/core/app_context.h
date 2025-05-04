#pragma once

#include <sstream>
#include <vector>

#include "common.h"
#include "llama.h"
#include "sampling.h"

namespace app {

extern llama_context **           g_ctx;
extern llama_model **             g_model;
extern common_sampler **          g_smpl;
extern common_params *            g_params;
extern std::vector<llama_token> * g_input_tokens;
extern std::vector<llama_token> * g_output_tokens;
extern std::ostringstream *       g_output_ss;
extern bool                       is_interacting;
extern bool                       need_insert_eot;

void initialize_globals();

}  // namespace app
