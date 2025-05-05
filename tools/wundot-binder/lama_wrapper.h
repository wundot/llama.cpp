// llama_wrapper.h

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void *       llama_init_model(const char * model_path);
const char * llama_generate(void * ctx, const char * prompt);
void         llama_free_model(void * ctx);

#ifdef __cplusplus
}
#endif
