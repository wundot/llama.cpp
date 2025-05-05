cmake ..
make -j$(nproc)


Llama-4-Maverick-17B-128E-Instruct-Q4_K_M-00001-of-00005.gguf

./wundot-llm/build/bin/llama-cli  -m models/meta-llama-unsloth-community/maverick/17B/Q4_K_M/Llama-4-Maverick-17B-128E-Instruct-Q4_K_M-00001-of-00005.gguf -no-cnv --prompt "Who is john travolta ?"


./wundot-llm/build/bin/llama-cli  -m models/meta-llama-unsloth-community/maverick/17B/Q4_K_M/Llama-4-Maverick-17B-128E-Instruct-Q4_K_M-00001-of-00005.gguf  --prompt "Who is William Echenim "


./build/bin/wundot-cli -m models/ggml-vocab-aquila.gguf   -p "Hello, how are you?" -n 64
