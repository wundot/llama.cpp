set(TARGET wundot-cli)
add_executable(${TARGET} 
main.cpp
llama_main.cpp  # ← Make sure this is included
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
