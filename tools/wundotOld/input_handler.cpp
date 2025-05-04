#include "input_handler.h"

#include <iostream>
#include <sstream>
#include <string>

#include "common.h"
#include "llama.h"
#include "log.h"
#include <console.h>

std::vector<llama_token> InputHandler::tokenize_input(llama_context * ctx, const std::string & input, bool add_bos) {
    LOG_DBG("Tokenizing input: '%s'\n", input.c_str());
    return common_tokenize(ctx, input, add_bos, true);
}

void InputHandler::handle_interactive_mode(llama_context * ctx, std::vector<llama_token> & embd_inp,
                                           bool & is_interacting) {
    if (!is_interacting) {
        return;
    }

    LOG_INF("Waiting for user input...\n");
    std::string buffer;
    console::set_display(console::user_input);

    // Read user input line by line
    std::string line;
    bool        another_line = true;
    do {
        another_line = console::readline(line, /* multiline_input= */ false);
        buffer += line;
    } while (another_line);

    console::set_display(console::reset);

    if (buffer.empty()) {
        LOG_INF("EOF by user\n");
        is_interacting = false;
        return;
    }

    // Remove trailing newline if present
    if (!buffer.empty() && buffer.back() == '\n') {
        buffer.pop_back();
    }

    LOG_DBG("User input: '%s'\n", buffer.c_str());

    // Tokenize the input and append to the embedding input
    auto tokens = tokenize_input(ctx, buffer, false);
    embd_inp.insert(embd_inp.end(), tokens.begin(), tokens.end());

    is_interacting = false;
}

std::string InputHandler::format_chat_message(const std::string & role, const std::string & content) {
    LOG_DBG("Formatting chat message for role: '%s', content: '%s'\n", role.c_str(), content.c_str());
    std::ostringstream formatted_message;
    formatted_message << "[" << role << "]: " << content;
    return formatted_message.str();
}
