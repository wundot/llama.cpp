#pragma once

#include <string>
#include <vector>

#include "llama.h"

namespace session_io {

bool file_exists(const std::string & path);
bool file_is_empty(const std::string & path);

bool load_session(llama_context * ctx, const std::string & path, std::vector<llama_token> & out_tokens);

bool save_session(llama_context * ctx, const std::string & path, const std::vector<llama_token> & tokens);

}  // namespace session_io
