#include "llama_runtime.h"
#include "llama_utils.h"

int main(int argc, char ** argv) {
    try {
        llama::Runtime runtime;
        if (!runtime.parse_args(argc, argv)) {
            return 1;
        }

        runtime.initialize();
        runtime.load_model();

        llama::ChatFormatter formatter(runtime.get_model(), runtime.get_params().chat_template);
        runtime.setup_chat_context(formatter);

        llama::SessionCache session_cache;
        session_cache.load(runtime.get_params().path_prompt_cache, runtime.get_context(), runtime.get_input_tokens());

        runtime.prepare_prompt(formatter);

        llama::TokenSampler sampler(runtime.get_model(), runtime.get_params().sampling);
        runtime.run_loop(sampler);

        session_cache.save(runtime.get_params().path_prompt_cache, runtime.get_context(), runtime.get_input_tokens());

        runtime.shutdown();
    } catch (const std::exception & ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
