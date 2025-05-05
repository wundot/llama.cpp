#include "generation_loop.h"

#include "app_context.h"
#include "chat.h"
#include "common.h"
#include "console.h"
#include "log.h"

namespace generation {

void run_generation_loop(llama_context * ctx, llama_model * model, common_sampler * smpl,
                         std::vector<llama_token> & input_tokens, std::vector<llama_token> & session_tokens,
                         std::ostringstream & output_stream, std::string initial_prompt,
                         std::vector<common_chat_msg> *         chat_history,
                         std::shared_ptr<common_chat_templates> chat_templates) {
    const common_params & params = *app::g_params;

    // Silence unused parameter warnings (can be removed when used)
    (void) input_tokens;
    (void) chat_history;
    (void) chat_templates;

    const llama_vocab *      vocab = llama_model_get_vocab(model);
    std::vector<llama_token> embd_inp;

    // Setup prompt
    if (!initial_prompt.empty()) {
        embd_inp = common_tokenize(ctx, initial_prompt, true, true);
    } else if (!session_tokens.empty()) {
        embd_inp = session_tokens;
    }

    const bool add_bos = llama_vocab_get_add_bos(vocab) && !params.use_jinja;
    if (embd_inp.empty() && add_bos) {
        embd_inp.push_back(llama_vocab_bos(vocab));
    }

    // Setup runtime state
    int n_past     = 0;
    int n_remain   = params.n_predict;
    int n_consumed = 0;

    std::ostringstream       assistant_ss;
    std::vector<llama_token> embd;

    // Generation loop
    while (n_remain != 0 || params.interactive) {
        if (!embd.empty()) {
            if (n_past + (int) embd.size() >= llama_n_ctx(ctx)) {
                LOG_WRN("Context full. Stopping generation.\n");
                break;
            }

            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = std::min(params.n_batch, (int) embd.size() - i);
                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
                    LOG_ERR("llama_decode failed\n");
                    return;
                }
                n_past += n_eval;
            }

            if (!params.path_prompt_cache.empty() && session_tokens.size() < llama_n_ctx(ctx)) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
            }
        }

        embd.clear();

        // If we've consumed the prompt, start sampling
        if ((int) embd_inp.size() <= n_consumed) {
            llama_token id = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, id, true);
            embd.push_back(id);
            --n_remain;
        } else {
            // Continue consuming prompt
            while ((int) embd_inp.size() > n_consumed && (int) embd.size() < params.n_batch) {
                embd.push_back(embd_inp[n_consumed]);
                common_sampler_accept(smpl, embd_inp[n_consumed], false);
                ++n_consumed;
            }
        }

        // Output generated tokens
        for (auto id : embd) {
            std::string token_str = common_token_to_piece(ctx, id);
            output_stream << token_str;
            if (app::g_output_tokens) {
                app::g_output_tokens->push_back(id);
            }
            LOG("%s", token_str.c_str());
        }

        // If interactive and special token encountered
        if (!embd.empty() && llama_vocab_is_eog(vocab, embd.back()) && !params.interactive) {
            LOG(" [end of text]\n");
            break;
        }
    }
}

}  // namespace generation
