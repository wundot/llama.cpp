// core_processor.cpp
#include "core_processor.h"

#include "log.h"

void CoreProcessor::process_tokens(llama_context * ctx, common_sampler * smpl, std::vector<llama_token> & embd,
                                   int & n_past, int & n_remain) {
    LOG_DBG("Processing tokens...\n");

    // Example implementation:
    for (int i = 0; i < (int) embd.size(); i += 16) {  // Process in batches of 16
        int n_eval = std::min((int) embd.size() - i, 16);
        if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
            LOG_ERR("%s: failed to decode tokens\n", __func__);
            return;
        }
        n_past += n_eval;
        --n_remain;
    }

    LOG_DBG("Tokens processed successfully.\n");
}

void CoreProcessor::shift_context_window(llama_context * ctx, int & n_past, int n_keep) {
    LOG_DBG("Shifting context window...\n");

    // Example implementation:
    const int n_left    = n_past - n_keep;
    const int n_discard = n_left / 2;
    llama_kv_self_seq_rm(ctx, 0, n_keep, n_keep + n_discard);
    llama_kv_self_seq_add(ctx, 0, n_keep + n_discard, n_past, -n_discard);
    n_past -= n_discard;

    LOG_DBG("Context window shifted successfully.\n");
}

void CoreProcessor::extend_context_window(llama_context * ctx, int ga_n, int ga_w, int & n_past) {
    LOG_DBG("Extending context window...\n");

    // Example implementation:
    while (n_past >= ga_n + ga_w) {
        const int ib = (ga_n * ga_n) / ga_w;
        const int bd = (ga_w / ga_n) * (ga_n - 1);
        const int dd = (ga_w / ga_n) - ib * bd - ga_w;
        llama_kv_self_seq_add(ctx, 0, ga_n, n_past, ib * bd);
        llama_kv_self_seq_div(ctx, 0, ga_n + ib * bd, ga_n + ib * bd + ga_w, ga_n);
        llama_kv_self_seq_add(ctx, 0, ga_n + ib * bd + ga_w, n_past + ib * bd, dd);
        n_past -= bd;
        ga_n += ga_w / ga_n;
    }

    LOG_DBG("Context window extended successfully.\n");
}
