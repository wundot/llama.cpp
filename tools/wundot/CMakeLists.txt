cmake_minimum_required(VERSION 3.16)
project(wundot-cli LANGUAGES C CXX)

set(TARGET_NAME wundot-cli)
set(CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core)
set(PLATFORM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/platform)
set(IO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/io)

set(SRC_MAIN main.cpp)
set(SRC_CORE
    ${CORE_DIR}/app_context.cpp
    ${CORE_DIR}/chat_state.cpp
    ${CORE_DIR}/llama_runtime.cpp
    ${CORE_DIR}/generation_loop.cpp
)
set(SRC_PLATFORM
    ${PLATFORM_DIR}/signal_handler.cpp
)
set(SRC_IO
    ${IO_DIR}/session_io.cpp
)

set(INCLUDE_DIRS
    ${CORE_DIR}
    ${PLATFORM_DIR}
    ${IO_DIR}
)

add_executable(${TARGET_NAME}
    ${SRC_MAIN}
    ${SRC_CORE}
    ${SRC_PLATFORM}
    ${SRC_IO}
)

target_link_libraries(${TARGET_NAME} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET_NAME} PRIVATE ${INCLUDE_DIRS})
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)
install(TARGETS ${TARGET_NAME} RUNTIME)
