#ifndef BRIDGE_H
#define BRIDGE_H

#include "xsampling.h"

#ifdef __cplusplus
extern "C" {
#endif

// C-compatible function declarations only
bool Load_Model(const char * model_path, int n_predict);

void Set_Sampling_Params(const sampling_params * custom_params);

const char * Run_Inference(const char * system_prompt, const char * user_history, const char * current_prompt);

void Run_Cleanup();

bool Load_Anchor_Persona(const char * system_prompt, const char * user_prompt);

// Add this to match bridge.cpp declaration
sampling_params Get_FraudDetection_Params(const char * profile_name);

#ifdef __cplusplus
}
#endif

#endif  // BRIDGE_H
