#include "output_generator.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "common.h"
#include "llama.h"
#include "log.h"
#include <console.h>

void OutputGenerator::display_output(const std::string & token_str, bool is_user_input) {
    // Set the display mode based on whether the input is from the user or the model
    if (is_user_input) {
        console::set_display(console::user_input);
    } else {
        console::set_display(console::model_output);
    }

    // Display the token string to the console
    LOG("%s", token_str.c_str());

    // Reset the display mode after output
    console::set_display(console::reset);
}

void OutputGenerator::save_session(const std::string & path, const std::vector<llama_token> & session_tokens) {
    if (path.empty()) {
        LOG_WRN("%s: Session path is empty, skipping save.\n", __func__);
        return;
    }

    LOG_INF("%s: Saving session to '%s'\n", __func__, path.c_str());

    // Open the file in binary mode for writing
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("%s: Failed to open session file for writing: '%s'\n", __func__, path.c_str());
        return;
    }

    // Write the session tokens to the file
    for (const auto & token : session_tokens) {
        file.write(reinterpret_cast<const char *>(&token), sizeof(llama_token));
    }

    if (!file.good()) {
        LOG_ERR("%s: Error occurred while writing session file: '%s'\n", __func__, path.c_str());
    } else {
        LOG_INF("%s: Session saved successfully to '%s'\n", __func__, path.c_str());
    }
}
