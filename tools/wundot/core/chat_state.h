#pragma once

#include <memory>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"

namespace chat_state {

void format_system_prompt(const std::string & system_prompt, std::vector<common_chat_msg> & chat_msgs,
                          const std::shared_ptr<common_chat_templates> & chat_templates, bool use_jinja);

std::string format_user_prompt(const std::string & user_prompt, std::vector<common_chat_msg> & chat_msgs,
                               const std::shared_ptr<common_chat_templates> & chat_templates, bool use_jinja);

std::string apply_chat_template(const std::vector<common_chat_msg> &           chat_msgs,
                                const std::shared_ptr<common_chat_templates> & chat_templates,
                                bool                                           add_generation_prompt);

}  // namespace chat_state
