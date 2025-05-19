#ifndef WUNDOT_BRIDGE_H
#define WUNDOT_BRIDGE_H

#include "common.h"
#include "llama.h"
#include "sampling.h"

// Local declarations
static void            ApplyFraudDetectionProfile(common_params_sampling & s);
static common_chat_msg MakeChatMsg(const std::string & role, const std::string & content);

// Initialize and load model into memory with N inference contexts
bool Load_Model(const char * model_path, int n_predict, int context_pool_size = 8);

// Run inference with default global sampling parameters
const char * Run_Inference(const char * system_prompt, const char * user_history, const char * current_prompt);

// Run inference with explicitly provided sampling parameters and predict length
const char * Run_Inference_With_Params(const char * system_prompt, const char * user_history,
                                       const char * current_prompt, const common_params_sampling * params,
                                       int n_predict = 128);

// Clean up memory, models, and inference sessions
void Run_Cleanup();

#endif  // WUNDOT_BRIDGE_H
