#ifndef INIT_H
#define INIT_H

#include "common.h"
#include "llama.h"

class Initializer {
  public:
    static void initialize(const common_params & params);
    static void cleanup();

  private:
    static void setup_threadpool(const cpu_params & cpuparams);
    static void load_model_and_context(llama_model ** model, llama_context ** ctx, const common_params & params);
};

#endif  // INIT_H
