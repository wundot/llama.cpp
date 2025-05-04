// core_processor.h
#ifndef CORE_PROCESSOR_H
#define CORE_PROCESSOR_H

#include "common.h"
#include "llama.h"
#include "sampling.h"

class CoreProcessor {
  public:
    static void process_tokens(llama_context * ctx, common_sampler * smpl, std::vector<llama_token> & embd,
                               int & n_past, int & n_remain);
    static void shift_context_window(llama_context * ctx, int & n_past, int n_keep);
    static void extend_context_window(llama_context * ctx, int ga_n, int ga_w, int & n_past);
  private:
    static void setup_threadpool(const cpu_params & cpuparams);
};

#endif  // CORE_PROCESSOR_H
