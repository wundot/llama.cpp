#ifndef BRIDGE_H
#define BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Load model and initialize context pool
bool Load_Model(const char * model_path, int n_predict, int context_pool_size);

// Run inference using the model with specified prompt components
const char * Run_Inference_With_Params(const char * system_prompt, const char * user_history,
                                       const char * current_prompt, const sampling_params * params);
const char * Run_Inference(const char * system_prompt, const char * user_history, const char * current_prompt);

// Cleanup all model and context resources
void Run_Cleanup();

#ifdef __cplusplus
}
#endif

#endif  // BRIDGE_H
