#ifndef BRIDGE_H
#define BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Load the model and initialize a context pool for concurrent use
// model_path: path to the .gguf or compatible model file
// n_predict: maximum number of tokens to generate per inference
bool Load_Model(const char * model_path, int n_predict);

// Anchors a persona by accepting system and user prompts (stub for API consistency)
// For real usage, provide persona info per Run_Inference call
bool Load_Anchor_Persona(const char * system_prompt, const char * user_prompt);

// Replaces the simpler version of Run_Inference
// Accepts:
//   - system_prompt: defines AI persona (e.g., "You are a fraud detection assistant")
//   - user_history: optional prior user statement (can be empty)
//   - current_prompt: the actual current user input to infer from
// Returns:
//   - thread-local char* with generated response
const char * Run_Inference(const char * system_prompt, const char * user_history, const char * current_prompt);

// Frees all memory: model, contexts, samplers, and shuts down the backend
void Run_Cleanup();

#ifdef __cplusplus
}
#endif

#endif  // BRIDGE_H
