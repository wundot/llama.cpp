// tools/wundot-bridge/bridge.h

#ifndef WUNDOT_BRIDGE_H
#define WUNDOT_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Load the model from the specified path
void * load_model_wrapper(const char * model_path, int n_predict);

// Run inference with the given prompt
const char * run_inferance_wrapper(const char * prompt);

// Free model and context memory
void run_cleanup_wrapper();

#ifdef __cplusplus
}
#endif

#endif  // WUNDOT_BRIDGE_H
