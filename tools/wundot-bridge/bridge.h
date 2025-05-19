#ifndef BRIDGE_H
#define BRIDGE_H

#include "common.h"

// Initialize and load the model into memory. Creates a context pool.
extern "C" bool Load_Model(const char * model_path, int n_predict, int context_pool_size);

// Run inference with default global sampling parameters.
extern "C" const char * Run_Inference(const char * system_prompt, const char * user_history,
                                      const char * current_prompt);

// Run inference using provided sampling parameters and prediction limit.
extern "C" const char * Run_Inference_With_Params(const char * system_prompt, const char * user_history,
                                                  const char * current_prompt, const common_params_sampling * params,
                                                  int n_predict);

// Cleanup all contexts, samplers, and unload the model.
extern "C" void Run_Cleanup();

#endif  // BRIDGE_H
