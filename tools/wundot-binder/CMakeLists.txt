cmake_minimum_required(VERSION 3.10)

# Project metadata
project(llama_wrapper LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Output target name
set(TARGET llama_wrapper)

# Create shared library (not executable)
add_library(${TARGET} SHARED
    llama_wrapper.cpp
    # Add more source files from your monolithic codebase as needed
)

# Include directories
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(${TARGET} PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# Optional: set output directory
set_target_properties(${TARGET} PROPERTIES
    OUTPUT_NAME "llama_wrapper"                    # Produces libllama.so
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Optional install target
# install(TARGETS ${TARGET} LIBRARY DESTINATION lib)
