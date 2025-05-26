#include "chat_utils.h"

common_chat_msg MakeChatMsg(const std::string & role, const std::string & content) {
    common_chat_msg msg;
    msg.role              = role;
    msg.content           = content;
    msg.content_parts     = {};  // empty vector
    msg.tool_calls        = {};  // empty vector
    msg.reasoning_content = "";
    msg.tool_name         = "";
    msg.tool_call_id      = "";
    return msg;
}
