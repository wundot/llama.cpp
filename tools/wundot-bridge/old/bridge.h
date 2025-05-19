#ifndef WUNDOT_BRIDGE_H
#define WUNDOT_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Struct must match layout with Go-side CGO struct
typedef struct {
    float temp;
    float top_p;
    int   top_k;
    float penalty_repeat;
    float penalty_present;
    float penalty_freq;
    int   mirostat;
} common_params_sampling;

/**
 * Initializes the LLaMA model from file, sets up thread-safe context pool.
 *
 * @param model_path path to the model file
 * @param n_predict number of tokens to predict per inference
 * @param context_pool_size number of reusable contexts to preload
 * @return 1 on success, 0 on failure
 */
int Load_Model(const char * model_path, int n_predict, int context_pool_size);

/**
 * Runs inference with default (predefined) fraud detection sampling parameters.
 *
 * @param system_prompt the initial system instruction
 * @param user_history optional previous user inputs
 * @param current_prompt current user input
 * @return generated output text (valid until next thread call)
 */
const char * Run_Inference(const char * system_prompt, const char * user_history, const char * current_prompt);

/**
 * Runs inference using a custom sampling parameter configuration.
 *
 * @param system_prompt the initial system instruction
 * @param user_history optional previous user inputs
 * @param current_prompt current user input
 * @param params pointer to sampling parameters
 * @param n_predict number of tokens to generate
 * @return generated output text (valid until next thread call)
 */
const char * Run_Inference_With_Params(const char * system_prompt, const char * user_history,
                                       const char * current_prompt, const common_params_sampling * params,
                                       int n_predict);

/**
 * Applies the built-in optimized profile for fraud/scam detection.
 * This mutates the `common_params_sampling` passed in.
 */
void ApplyFraudDetectionProfile(common_params_sampling * params);

/**
 * Frees all loaded model resources and context sessions.
 */
void Run_Cleanup(void);

#ifdef __cplusplus
}
#endif

#endif  // WUNDOT_BRIDGE_H
