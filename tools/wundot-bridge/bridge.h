#ifndef BRIDGE_H
#define BRIDGE_H

#include "xsampling.h"  // ✅ C-compatible struct definition for sampling_params

#ifdef __cplusplus
extern "C" {
#endif

// Initialize model and context pool
bool Load_Model(const char * model_path, int n_predict);

// Set global sampling parameters
void Set_Sampling_Params(const sampling_params * custom_params);

// Run inference using system prompt, optional user history, and the current prompt
const char * Run_Inference(const char * system_prompt, const char * user_history, const char * current_prompt);

// Cleanup all resources: model, samplers, and contexts
void Run_Cleanup();

// Placeholder function for legacy compatibility; always returns true if inputs are valid
bool Load_Anchor_Persona(const char * system_prompt, const char * user_prompt);

// Get a predefined sampling profile based on fraud detection use cases
sampling_params Get_FraudDetection_Params(const char * profile_name);

#ifdef __cplusplus
}
#endif

#endif  // BRIDGE_H
