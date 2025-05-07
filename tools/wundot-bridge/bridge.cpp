#include <cstdio>
#include <string>

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"

// Expose the model loading function
extern "C" void * load_model(const char * model_path, int n_ctx, int n_threads) {
    common_params params;
    params.model               = model_path;
    params.n_ctx               = n_ctx;
    params.cpuparams.n_threads = n_threads;

    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result llama_init = common_init_from_params(params);
    if (!llama_init.model) {
        fprintf(stderr, "Error: Unable to load model\n");
        return nullptr;
    }

    return llama_init.context.get();
}

// Expose the prompt processing function
extern "C" const char * process_prompt(void * ctx, const char * prompt) {
    static std::string output;
    output.clear();

    llama_context * context = static_cast<llama_context *>(ctx);
    if (!context) {
        fprintf(stderr, "Error: Invalid context\n");
        return nullptr;
    }

    // Tokenize the prompt
    std::vector<llama_token> tokens = common_tokenize(context, prompt, true, true);

    // Generate response
    std::ostringstream response_stream;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string token_str = common_token_to_piece(context, tokens[i], false);
        response_stream << token_str;
    }

    output = response_stream.str();
    return output.c_str();
}
