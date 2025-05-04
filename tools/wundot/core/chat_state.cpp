#include "chat_state.h"

#include "log.h"

namespace chat_state {

void format_system_prompt(const std::string & system_prompt, std::vector<common_chat_msg> & chat_msgs,
                          const std::shared_ptr<common_chat_templates> & chat_templates, bool use_jinja) {
    if (!system_prompt.empty()) {
        common_chat_msg msg = { "system", system_prompt };
        chat_msgs.push_back(msg);
        std::string formatted = common_chat_format_single(chat_templates.get(), chat_msgs, msg, false, use_jinja);
        LOG_DBG("formatted system prompt: '%s'\n", formatted.c_str());
    }
}

std::string format_user_prompt(const std::string & user_prompt, std::vector<common_chat_msg> & chat_msgs,
                               const std::shared_ptr<common_chat_templates> & chat_templates, bool use_jinja) {
    common_chat_msg msg       = { "user", user_prompt };
    auto            formatted = common_chat_format_single(chat_templates.get(), chat_msgs, msg, true, use_jinja);
    chat_msgs.push_back(msg);
    LOG_DBG("formatted: '%s'\n", formatted.c_str());
    return formatted;
}

std::string apply_chat_template(const std::vector<common_chat_msg> &           chat_msgs,
                                const std::shared_ptr<common_chat_templates> & chat_templates,
                                bool                                           add_generation_prompt) {
    common_chat_templates_inputs inputs;
    inputs.messages              = chat_msgs;
    inputs.add_generation_prompt = add_generation_prompt;

    return common_chat_templates_apply(chat_templates.get(), inputs).prompt;
}

}  // namespace chat_state
