#pragma once

#include "common.h"
#include "llama.h"

namespace llama_runtime {

bool initialize_backend(const common_params & params, llama_model *& model, llama_context *& ctx,
                        struct ggml_threadpool *& threadpool, struct ggml_threadpool *& threadpool_batch);

}
