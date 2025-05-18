#ifndef BRIDGE_H
#define BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "sampling.h"

// Load the model and pre-initialize a pool of contexts/samplers.
// model_path: path to the model file (e.g., .gguf)
// n_predict: default max token count per inference
bool Load_Model(const char * model_path, int n_predict);

// Override the global sampling parameters dynamically at runtime.
// custom_params: pointer to a sampling_params struct with desired settings
void Set_Sampling_Params(const sampling_params * custom_params);

// Perform inference using:
// - system_prompt: defines AI behavior or role (e.g., "You are a legal assistant")
// - user_history: optional prior user message (may be NULL or empty)
// - current_prompt: the actual prompt triggering inference
// Returns: generated response (valid until next call on the same thread)
const char * Run_Inference(const char * system_prompt, const char * user_history, const char * current_prompt);

// Frees all resources including model, samplers, and contexts
void Run_Cleanup();

// (Legacy placeholder) For API compatibility — actual persona injection is per-request
bool Load_Anchor_Persona(const char * system_prompt, const char * user_prompt);

#ifdef __cplusplus
}
#endif

#endif  // BRIDGE_H
