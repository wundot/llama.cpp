// binder.cpp
#include "bridge.h"

#include <mutex>
#include <string>

#include "chat.h"
#include "common.h"
#include "llama.h"

static std::string result;
static std::mutex  result_mutex;

extern "C" {

void * init_model(const char * model_path) {
    llama_context * ctx = llama_init_from_file(model_path);  // your model loader
    return static_cast<void *>(ctx);
}

const char * run_inference(void * ctx_ptr, const char * prompt) {
    llama_context * ctx = static_cast<llama_context *>(ctx_ptr);

    std::lock_guard<std::mutex> lock(result_mutex);
    // Replace this with your actual inference logic
    result = llama_infer(ctx, prompt);
    return result.c_str();
}

void free_model(void * ctx_ptr) {
    llama_context * ctx = static_cast<llama_context *>(ctx_ptr);
    llama_free(ctx);
}
}
