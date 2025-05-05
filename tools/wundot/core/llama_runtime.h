#pragma once

#include "common.h"
#include "ggml/ggml.h"
#include "llama.h"

namespace llama_runtime {
bool initialize_backend(const common_params & params, llama_model *& model, llama_context *& ctx,
                        ggml_threadpool *& threadpool, ggml_threadpool *& threadpool_batch);

void free_threadpools(ggml_threadpool * threadpool, ggml_threadpool * threadpool_batch);  // <-- ensure this is declared
}  // namespace llama_runtime
