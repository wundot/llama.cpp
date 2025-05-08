#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// InferenceSession handle for stateful inference across multiple calls
typedef struct InferenceSession InferenceSession;

// Load and initialize the model
// model_path: file path to .gguf model
// n_predict: max tokens to generate during inference
void * load_model_wrapper(const char * model_path, int n_predict);

// Free model and backend
void run_cleanup_wrapper();

// Run a stateless inference on the provided prompt
// Returns a pointer to the output string
const char * run_inferance_wrapper(const char * prompt);

// Stateful inference session
InferenceSession * session_create(const char * model_path, int n_predict);
void               session_free(InferenceSession * session);

// Begin a new prompt stream
void session_start_stream(InferenceSession * session, const char * prompt);

// Fetch the next token from the session (streaming generation)
// Returns nullptr if end-of-generation is reached
const char * session_next_token(InferenceSession * session);

#ifdef __cplusplus
}
#endif
