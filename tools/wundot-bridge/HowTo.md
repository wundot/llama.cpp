rm -rf build && mkdir build && cd build
<!-- buld with CPU only -->
cmake .. && make -j$(nproc)
<!-- build with CPU & GPU  -->
 cmake -DGGML_CUDA=ON .. && make -j$(nproc)


git config --global push.autoSetupRemote true


git pull origin safe-wundot-llama

git pull origin llama.cpp

git branch --set-upstream-to=origin/llama.cpp main



./wundot-llama/build/bin/llama-cli  -m models/meta-llama-unsloth-community/maverick/17B/Q4_K_M/Llama-4-Maverick-17B-128E-Instruct-Q4_K_M-00001-of-00005.gguf -no-cnv
  --prompt "Analyze the following message for any signs of fraud or scam tactics: 'Dear customer, your account has been suspended due to suspicious activity. Please click this link to verify your identity immediately: <http://bit.ly/fake-login>'"  --n-predict 80 \
  --repeat-penalty 1.2 \
  --temperature 0.2 \
  --stop "Answer:" \
  --log-disable

./wundot-llama/build/bin/llama-cli  -m models/meta-llama-unsloth-community/maverick/17B/Q4_K_M/Llama-4-Maverick-17B-128E-Instruct-Q4_K_M-00001-of-00005.gguf -no-cnv \
  --temperature 0.2 \
  --stop "Answer:" \
  --log-disable

