#include <grpcpp/grpcpp.h>

#include "llama_engine.h"
#include "llama_service_impl.h"

int main(int argc, char ** argv) {
    std::string model_path     = "./models/llama.gguf";
    std::string server_address = "0.0.0.0:50051";

    llama_engine::init(model_path);

    LlamaServiceImpl    service;
    grpc::ServerBuilder builder;

    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "LLaMA gRPC Server listening on " << server_address << std::endl;
    server->Wait();

    return 0;
}
