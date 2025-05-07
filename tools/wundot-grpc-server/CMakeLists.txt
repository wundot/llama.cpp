cmake_minimum_required(VERSION 3.14)
project(llama_grpc_server C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable gRPC / Protobuf
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Protobuf_INCLUDE_DIRS}
    ${gRPC_INCLUDE_DIRS}
    ./third_party/llama.cpp  # Adjust path to llama.cpp
)

# Source files
set(SOURCES
    server.cpp
    llama_service_impl.cpp
    llama_engine.cpp
    llama_service.pb.cc
    llama_service.grpc.pb.cc
)

# Define executable
add_executable(llama_grpc_server ${SOURCES})

# Link libraries
target_link_libraries(llama_grpc_server
    PRIVATE
        grpc++
        protobuf::libprotobuf
        Threads::Threads
        common       # Assuming you have `common` as a library
        llama        # Assuming you have `llama` as a library
)

# Install binary
install(TARGETS llama_grpc_server RUNTIME DESTINATION bin)
