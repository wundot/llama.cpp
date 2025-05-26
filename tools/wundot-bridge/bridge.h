#ifndef WUNDOT_BRIDGE_H
#define WUNDOT_BRIDGE_H

#include <stdbool.h>  // Required for 'bool' type when used with CGO/C

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#ifdef __cplusplus
extern "C" {
#endif

// Check whether the model has been successfully loaded
bool Is_Model_Loaded(void);

// Initialize and load model into memory with N inference contexts
void Load_Model(const char * model_path, int n_predict, int context_pool_size);

// Run inference with default global sampling parameters
const char * Run_Inference(const char * system_prompt, const char * user_history, const char * current_prompt);

// Run inference with explicitly provided sampling parameters and predict length
const char * Run_Inference_With_Params(const char * system_prompt, const char * user_history,
                                       const char * current_prompt, const common_params_sampling * params,
                                       int n_predict);

// Clean up memory, models, and inference sessions
void Run_Cleanup(void);

#ifdef __cplusplus
}
#endif

#endif  // WUNDOT_BRIDGE_H
