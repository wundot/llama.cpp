#pragma once
#include "llama_service.grpc.pb.h"

class LlamaServiceImpl final : public llama::LlamaService::Service {
  public:
    grpc::Status GenerateText(grpc::ServerContext * context, const llama::TextRequest * request,
                              llama::TextResponse * response) override;
};
