#include "chat_utils.h"

common_chat_msg MakeChatMsg(const std::string & role, const std::string & content) {
    common_chat_msg msg;
    msg.role              = role;
    msg.content           = content;
    msg.content_parts     = {};
    msg.tool_calls        = {};
    msg.reasoning_content = "";
    msg.tool_name         = "";
    msg.tool_call_id      = "";
    return msg;
}
