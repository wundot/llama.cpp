# Minimum required version of CMake
cmake_minimum_required(VERSION 3.14)

# Project name and language
project(wundot_bridge LANGUAGES CXX)

# Set the output shared library name
set(TARGET wundot-bridge)

# Add the source file for the shared library
add_library(${TARGET} SHARED bridge.cpp)

# Include directories for headers
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include           # For llama-cpp.h, llama.h
    ${CMAKE_SOURCE_DIR}/common            # For common utilities like arg.h, chat.h, etc.
    ${CMAKE_CURRENT_SOURCE_DIR}           # For bridge.h in the current directory
)

# Link the required libraries
target_link_libraries(${TARGET} PRIVATE
    llama                                 # The main llama library
    ggml                                  # The GGML library
    common                                # Common utilities library
    Threads::Threads                      # Thread support
)

# Specify the required C++ standard
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Compiler flags for shared library compatibility
set_target_properties(${TARGET} PROPERTIES
    POSITION_INDEPENDENT_CODE ON          # Ensure Position Independent Code (PIC) is enabled
    PREFIX ""                             # Remove "lib" prefix on Unix systems if desired
    SUFFIX ".so"                          # Ensure the output is named as a shared library
)

# Install the shared library
install(TARGETS ${TARGET}
    LIBRARY DESTINATION ${LLAMA_LIB_INSTALL_DIR}  # Install to the library directory
    PUBLIC_HEADER DESTINATION ${LLAMA_INCLUDE_INSTALL_DIR}  # Install headers
)

# Optional: Export the shared library for use by other projects
export(TARGETS ${TARGET} FILE ${CMAKE_BINARY_DIR}/${TARGET}-targets.cmake)