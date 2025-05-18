#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_STOP_SEQUENCES 4
#define MAX_STOP_LENGTH    64

// Extended structure for full sampling control
typedef struct {
    float temperature;
    int   top_k;
    float top_p;
    float repeat_penalty;
    int   n_predict;
    float frequency_penalty;
    float presence_penalty;
    int   num_stop_sequences;
    char  stop_sequences[MAX_STOP_SEQUENCES][MAX_STOP_LENGTH];
} SamplingParams;

// Forward declaration of stateful inference session
typedef struct InferenceSession InferenceSession;

// Load model once globally
void * load_model_wrapper(const char * model_path, int n_predict);

// Cleanup model
void run_cleanup_wrapper();

// Stateless inference
const char * run_inferance_wrapper(const char * prompt);

// Stateless inference with full sampling control
const char * run_infer_with_sampling(const char * prompt, SamplingParams params);

// Stateful inference (streaming support)
InferenceSession * session_create(const char * model_path, int n_predict);
void               session_free(InferenceSession * session);
void               session_start_stream(InferenceSession * session, const char * prompt);
const char *       session_next_token(InferenceSession * session);

#ifdef __cplusplus
}
#endif
