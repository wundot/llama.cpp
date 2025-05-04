#include <iostream>
#include <vector>

#include "common.h"
#include "core_processor.h"
#include "init.h"
#include "input_handler.h"
#include "llama.h"
#include "log.h"
#include "output_generator.h"
#include "signal_handler.h"
#include <arg.h>

int main(int argc, char ** argv) {
    // Parse command-line arguments
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    // Initialize the application
    Initializer::initialize(params);
    SignalHandler::setup_signal_handlers();

    // Load model and context
    llama_model *    model = nullptr;
    llama_context *  ctx   = nullptr;
    common_sampler * smpl  = nullptr;

    Initializer::load_model_and_context(&model, &ctx, params);
    smpl = common_sampler_init(model, params.sampling);

    // Tokenize the prompt
    std::vector<llama_token> embd_inp = InputHandler::tokenize_input(ctx, params.prompt, true);

    // Handle interactive mode
    bool is_interacting = false;
    if (params.interactive) {
        is_interacting = true;
        InputHandler::handle_interactive_mode(ctx, embd_inp, is_interacting);
    }

    // Core processing loop
    int n_past = 0, n_remain = params.n_predict;
    while (n_remain > 0) {
        CoreProcessor::process_tokens(ctx, smpl, embd_inp, n_past, n_remain);

        // Display output
        const std::string token_str = common_token_to_piece(ctx, embd_inp.back(), params.special);
        OutputGenerator::display_output(token_str, false);

        // Check for user interaction in interactive mode
        if (is_interacting && params.interactive) {
            InputHandler::handle_interactive_mode(ctx, embd_inp, is_interacting);
        }
    }

    // Save session if required
    if (!params.path_prompt_cache.empty()) {
        OutputGenerator::save_session(params.path_prompt_cache, embd_inp);
    }

    // Cleanup resources
    SignalHandler::cleanup_resources();
    Initializer::cleanup();

    return 0;
}
