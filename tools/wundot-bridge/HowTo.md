rm -rf build && mkdir build && cd build
<!-- buld with CPU only -->
cmake .. && make -j$(nproc)
<!-- build with CPU & GPU  -->
 cmake -DGGML_CUDA=ON .. && make -j$(nproc)


git config --global push.autoSetupRemote true


git pull origin safe-wundot-llama
