#include "core/app_context.h"

namespace app {

llama_context **           g_ctx           = nullptr;
llama_model **             g_model         = nullptr;
common_sampler **          g_smpl          = nullptr;
common_params *            g_params        = nullptr;
std::vector<llama_token> * g_input_tokens  = nullptr;
std::vector<llama_token> * g_output_tokens = nullptr;
std::ostringstream *       g_output_ss     = nullptr;
bool                       is_interacting  = false;
bool                       need_insert_eot = false;

void initialize_globals() {
    static llama_context *          ctx_instance   = nullptr;
    static llama_model *            model_instance = nullptr;
    static common_sampler *         smpl_instance  = nullptr;
    static std::vector<llama_token> input_tokens;
    static std::vector<llama_token> output_tokens;
    static std::ostringstream       output_stream;

    g_ctx           = &ctx_instance;
    g_model         = &model_instance;
    g_smpl          = &smpl_instance;
    g_input_tokens  = &input_tokens;
    g_output_tokens = &output_tokens;
    g_output_ss     = &output_stream;
}

}  // namespace app
