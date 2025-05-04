#include "session_io.h"

#include <fstream>

bool session_io::file_exists(const std::string & path) {
    std::ifstream f(path.c_str());
    return f.good();
}

bool session_io::file_is_empty(const std::string & path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

bool session_io::load_session(llama_context * ctx, const std::string & path, std::vector<llama_token> & out_tokens) {
    out_tokens.resize(llama_n_ctx(ctx));
    size_t n_token_count_out = 0;
    bool   success =
        llama_state_load_file(ctx, path.c_str(), out_tokens.data(), out_tokens.capacity(), &n_token_count_out);
    out_tokens.resize(n_token_count_out);
    return success;
}

bool session_io::save_session(llama_context * ctx, const std::string & path, const std::vector<llama_token> & tokens) {
    return llama_state_save_file(ctx, path.c_str(), tokens.data(), tokens.size());
}
