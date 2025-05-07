

protoc \
  --proto_path=proto \
  --cpp_out=generated \
  --grpc_out=generated \
  --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) \
  proto/llama_service.proto
