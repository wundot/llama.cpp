#ifndef WUNDOT_BRIDGE_H
#define WUNDOT_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Global model loading (one-time)
void * load_model_wrapper(const char * model_path, int n_predict);
void   run_cleanup_wrapper();

// Stateless full inference (for simple use)
const char * run_inferance_wrapper(const char * prompt);

// Inference session struct forward-declared
typedef struct InferenceSession InferenceSession;

// Session-based streaming interface
InferenceSession * session_create(const char * model_path, int n_predict);
void               session_free(InferenceSession * session);

void         session_start_stream(InferenceSession * session, const char * prompt);
const char * session_next_token(InferenceSession * session);

#ifdef __cplusplus
}
#endif

#endif  // WUNDOT_BRIDGE_H
