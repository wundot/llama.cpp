#include "core/llama_runtime.h"

#include "log.h"

bool llama_runtime::initialize_backend(const common_params & params, llama_model *& model, llama_context *& ctx,
                                       struct ggml_threadpool *& threadpool,
                                       struct ggml_threadpool *& threadpool_batch) {
    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result llama_init = common_init_from_params(params);
    model                         = llama_init.model.get();
    ctx                           = llama_init.context.get();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: error initializing model/context\n", __func__);
        return false;
    }

    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    auto * ggml_threadpool_new_fn =
        (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto * ggml_threadpool_free_fn =
        (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    struct ggml_threadpool_params tpp_batch = ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    struct ggml_threadpool_params tpp       = ggml_threadpool_params_from_cpu_params(params.cpuparams);

    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            LOG_ERR("%s: failed to initialize batch threadpool\n", __func__);
            return false;
        }
        tpp.paused = true;
    }

    threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        LOG_ERR("%s: failed to initialize threadpool\n", __func__);
        return false;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);
    return true;
}
