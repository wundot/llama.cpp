// binder.h
#ifndef WUNDOT_BRIDGE_H
#define WUNDOT_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize model context
void * init_model(const char * model_path);

// Run inference and get result string
const char * run_inference(void * ctx, const char * prompt);

// Free the model context
void free_model(void * ctx);

#ifdef __cplusplus
}
#endif

#endif  // WUNDOT_BRIDGE_H
