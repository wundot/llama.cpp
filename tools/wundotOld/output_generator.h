#ifndef OUTPUT_GENERATOR_H
#define OUTPUT_GENERATOR_H

#include <string>
#include <vector>

#include "common.h"
#include "llama.h"

class OutputGenerator {
  public:
    static void display_output(const std::string & token_str, bool is_user_input);
    static void save_session(const std::string & path, const std::vector<llama_token> & session_tokens);
};

#endif  // OUTPUT_GENERATOR_H
