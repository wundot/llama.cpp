#include "initializer.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "llama.h"
#include "log.h"

// Static variables for global access
static llama_model **   g_model;
static llama_context ** g_ctx;

void Initializer::initialize(const common_params & params) {
    // Initialize logging
    LOG_INF("%s: Initializing application...\n", __func__);

    // Log system information
    LOG_INF("%s: %s\n", __func__, common_params_get_system_info(params).c_str());

    // Load the model and context
    load_model_and_context(g_model, g_ctx, params);

    // Set up the thread pool
    setup_threadpool(params.cpuparams);

    LOG_INF("%s: Initialization complete.\n", __func__);
}

void Initializer::cleanup() {
    // Cleanup resources
    if (*g_ctx) {
        llama_free(*g_ctx);
        *g_ctx = nullptr;
    }
    if (*g_model) {
        llama_model_free(*g_model);
        *g_model = nullptr;
    }

    LOG_INF("%s: Cleanup complete.\n", __func__);
}

void Initializer::setup_threadpool(const cpu_params & cpuparams) {
    LOG_INF("%s: Setting up threadpool with n_threads = %d\n", __func__, (int) cpuparams.n_threads);

    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    auto * ggml_threadpool_new_fn =
        (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto * ggml_threadpool_free_fn =
        (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    struct ggml_threadpool_params tpp = ggml_threadpool_params_from_cpu_params(cpuparams);

    struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        LOG_ERR("%s: Failed to create threadpool with n_threads = %d\n", __func__, tpp.n_threads);
        exit(1);
    }

    // Attach threadpool to the context
    llama_attach_threadpool(*g_ctx, threadpool, nullptr);

    LOG_INF("%s: Threadpool initialized successfully.\n", __func__);
}

void Initializer::load_model_and_context(llama_model ** model, llama_context ** ctx, const common_params & params) {
    LOG_INF("%s: Loading model and applying LoRA adapter, if any...\n", __func__);

    common_init_result llama_init = common_init_from_params(params);
    *model                        = llama_init.model.get();
    *ctx                          = llama_init.context.get();

    if (*model == nullptr) {
        LOG_ERR("%s: Error: Unable to load model\n", __func__);
        exit(1);
    }

    const int n_ctx_train = llama_model_n_ctx_train(*model);
    const int n_ctx       = llama_n_ctx(*ctx);

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: Model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }

    LOG_INF("%s: Model and context loaded successfully.\n", __func__);
}
