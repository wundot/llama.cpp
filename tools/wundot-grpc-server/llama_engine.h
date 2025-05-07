#pragma once
#include <string>

namespace llama_engine {
void        init(const std::string & model_path);
std::string generate_response(const std::string & prompt, int max_tokens);
}  // namespace llama_engine
