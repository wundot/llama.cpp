// chat_utils.h
#pragma once

#include <string>

#include "chat.h"

// Utility to create a basic chat message with default fields
common_chat_msg MakeChatMsg(const std::string & role, const std::string & content);
