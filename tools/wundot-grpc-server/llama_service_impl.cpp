#include "llama_service_impl.h"

#include "llama_engine.h"

grpc::Status LlamaServiceImpl::GenerateText(grpc::ServerContext *, const llama::TextRequest * request,
                                            llama::TextResponse * response) {
    try {
        std::string result = llama_engine::generate_response(request->prompt(), request->max_tokens());
        response->set_generated_text(result);
        return grpc::Status::OK;
    } catch (const std::exception & ex) {
        return grpc::Status(grpc::StatusCode::INTERNAL, ex.what());
    }
}
