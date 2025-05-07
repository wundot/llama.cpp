#ifndef BRIDGE_H
#define BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Context type definition (opaque pointer)
typedef void llama_context;

// Function to load the model
void * load_model(const char * model_path, int n_ctx, int n_threads);

// Function to process a prompt and generate a response
const char * process_prompt(void * ctx, const char * prompt);

#ifdef __cplusplus
}
#endif

#endif  // BRIDGE_H
