#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Structure for sampling parameters
typedef struct {
    float temperature;
    int   top_k;
    float top_p;
    float repeat_penalty;
    int   n_predict;
} SamplingParams;

// InferenceSession handle for stateful inference across multiple calls
typedef struct InferenceSession InferenceSession;

// Load and initialize the model
void * load_model_wrapper(const char * model_path, int n_predict);

// Free model and backend
void run_cleanup_wrapper();

// Run a stateless inference on the provided prompt
const char * run_inferance_wrapper(const char * prompt);

// Run inference with custom sampling parameters
const char * run_infer_with_sampling(const char * prompt, SamplingParams params);

// Stateful inference session
InferenceSession * session_create(const char * model_path, int n_predict);
void               session_free(InferenceSession * session);
void               session_start_stream(InferenceSession * session, const char * prompt);
const char *       session_next_token(InferenceSession * session);

#ifdef __cplusplus
}
#endif
