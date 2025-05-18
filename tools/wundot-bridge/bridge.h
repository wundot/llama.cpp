#ifndef BRIDGE_H
#define BRIDGE_H

#include "xsampling.h"

#ifdef __cplusplus
extern "C" {
#endif

// Load the model and initialize the context pool.
// - model_path: path to the model file (.gguf, etc).
// - n_predict: max tokens to generate per call.
// - context_pool_size: number of reusable contexts (suggest 1–128).
bool Load_Model(const char * model_path, int n_predict, int context_pool_size);

// Override global sampling configuration for generation.
void Set_Sampling_Params(const sampling_params * custom_params);

// Run inference and return the generated output.
// Thread-safe. Result is valid until the next call on the same thread.
const char * Run_Inference(const char * system_prompt, const char * user_history, const char * current_prompt);

// Free all contexts, samplers, and model memory.
void Run_Cleanup();

// Placeholder for legacy compatibility. Always returns true if inputs are not null.
bool Load_Anchor_Persona(const char * system_prompt, const char * user_prompt);

#ifdef __cplusplus
}
#endif

#endif  // BRIDGE_H
