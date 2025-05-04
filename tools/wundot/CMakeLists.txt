set(TARGET_NAME wundot-cli)
set(CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core)
set(PLATFORM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/platform)
set(IO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/io)

set(SRC_MAIN main.cpp)
set(SRC_CORE
    core/app_context.cpp
    core/chat_state.cpp
    core/llama_runtime.cpp
)
set(SRC_PLATFORM
    ${PLATFORM_DIR}/signal_handler.cpp
)
set(SRC_IO
    io/session_io.cpp
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
target_include_directories(${TARGET_NAME}
    PRIVATE
        ${INCLUDE_DIRS}
)
install(TARGETS ${TARGET_NAME} RUNTIME)

target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)
