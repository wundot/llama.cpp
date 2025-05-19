#ifndef BRIDGE_H
#define BRIDGE_H

#include "common.h"

// Loads the model once and initializes context pool.
// Must be called before any inference.
extern "C" bool Load_Model(const char * model_path, int n_predict, int context_pool_size);

// Runs inference using the default sampling parameters.
extern "C" const char * Run_Inference(const char * system_prompt, const char * user_history,
                                      const char * current_prompt);

// Runs inference using custom sampling parameters for this specific request.
extern "C" const char * Run_Inference_With_Params(const char * system_prompt, const char * user_history,
                                                  const char * current_prompt, const common_params_sampling * params);

// Frees all resources and shuts down LLaMA backend.
extern "C" void Run_Cleanup();

#endif  // BRIDGE_H
