#include "chat_utils.h"

#include <cstdlib>  // for malloc
#include <cstring>  // for strncpy

// C-compatible implementation
common_chat_msg MakeChatMsg(const char * role, const char * content) {
    common_chat_msg msg;

    // Copy role and content into new heap-allocated strings (C-compatible)
    msg.role              = strdup(role);
    msg.content           = strdup(content);
    msg.content_parts     = nullptr;
    msg.num_parts         = 0;
    msg.tool_calls        = nullptr;
    msg.num_tool_calls    = 0;
    msg.reasoning_content = nullptr;
    msg.tool_name         = nullptr;
    msg.tool_call_id      = nullptr;

    return msg;
}
