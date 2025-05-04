set(TARGET wundot-cli)
# Add the modularized source files
set(SOURCES
    main.cpp
    init.cpp
    signal_handler.cpp
    input_handler.cpp
    core_processor.cpp
    output_generator.cpp
    utils.cpp
)

add_executable(${TARGET} ${SOURCES})
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
