#pragma once

#include "chat.h"

#ifdef __cplusplus
extern "C" {
#endif

// C-compatible wrapper for MakeChatMsg
// Accepts raw C strings and returns a chat message struct
common_chat_msg MakeChatMsg(const char * role, const char * content);

#ifdef __cplusplus
}
#endif
