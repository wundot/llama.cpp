// common/chat_internal.h

#pragma once

#include <memory>

struct common_chat_template;

struct common_chat_templates {
    bool                                  has_explicit_template;
    std::unique_ptr<common_chat_template> template_default;
    std::unique_ptr<common_chat_template> template_tool_use;
};
