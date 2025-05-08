#ifndef WUNDOT_BRIDGE_H
#define WUNDOT_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Load the model from the specified path with prediction size
void * load_model_wrapper(const char * model_path, int n_predict);

// Run single inference with full response
const char * run_inferance_wrapper(const char * prompt);

// Start streaming inference with a prompt
void start_stream_wrapper(const char * prompt);

// Get next token from stream; returns NULL when done
const char * next_token_wrapper();

// End streaming inference session
void end_stream_wrapper();

// Free model and context memory
void run_cleanup_wrapper();

#ifdef __cplusplus
}
#endif

#endif  // WUNDOT_BRIDGE_H
