#pragma once

#include "common.h"
#include "llama.h"

struct ggml_threadpool;  // Forward declaration

namespace llama_runtime {
bool initialize_backend(const common_params & params, llama_model *& model, llama_context *& ctx,
                        ggml_threadpool *& threadpool, ggml_threadpool *& threadpool_batch);

void free_threadpools(ggml_threadpool * threadpool, ggml_threadpool * threadpool_batch);
}  // namespace llama_runtime
