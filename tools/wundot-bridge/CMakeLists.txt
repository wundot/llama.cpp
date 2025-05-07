cmake_minimum_required(VERSION 3.14)

# Project metadata
project(wundot-bridge LANGUAGES CXX)

# Set the target name
set(TARGET wundot-bridge)

# Build the shared library
add_library(${TARGET} SHARED bridge.cpp)

# Include necessary directories
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include         # For llama-cpp.h, llama.h
    ${CMAKE_SOURCE_DIR}/common          # For arg.h, chat.h, etc.
    ${CMAKE_CURRENT_SOURCE_DIR}         # For bridge.h
)

# Link required libraries
target_link_libraries(${TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})

# Require C++17
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Install the shared library
install(TARGETS ${TARGET} LIBRARY)
