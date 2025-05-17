rm -rf build && mkdir build && cd build
<!-- buld with CPU only -->
cmake .. && make -j$(nproc)
<!-- build with CPU & GPU  -->
 cmake -DGGML_CUDA=ON .. && mkdir build && cd build
