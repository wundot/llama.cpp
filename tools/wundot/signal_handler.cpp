#include "signal_handler.h"

#include <csignal>
#include <cstdlib>
#include <iostream>

#include "log.h"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#    include <unistd.h>
#elif defined(_WIN32)
#    include <windows.h>
#endif

static bool is_interacting = false;

void SignalHandler::setup_signal_handlers() {
    LOG_INF("%s: Setting up signal handlers...\n", __func__);

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = SignalHandler::sigint_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    if (sigaction(SIGINT, &sigint_action, nullptr) != 0) {
        LOG_ERR("%s: Failed to set up SIGINT handler.\n", __func__);
        exit(1);
    }
#elif defined(_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (SignalHandler::sigint_handler(SIGINT), TRUE) : FALSE;
    };
    if (!SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), TRUE)) {
        LOG_ERR("%s: Failed to set up console control handler.\n", __func__);
        exit(1);
    }
#endif

    LOG_INF("%s: Signal handlers set up successfully.\n", __func__);
}

void SignalHandler::cleanup_resources() {
    LOG_INF("%s: Cleaning up resources...\n", __func__);

    // Perform any necessary cleanup here
    // For example, freeing memory, closing files, etc.

    LOG_INF("%s: Resources cleaned up successfully.\n", __func__);
}

void SignalHandler::sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting) {
            LOG_INF("\n%s: Interrupted by user. Exiting gracefully...\n", __func__);
            SignalHandler::cleanup_resources();
            _exit(130);  // Exit with code 130 (standard for SIGINT)
        } else {
            LOG_INF("\n%s: Interrupted during interaction. Pausing model generation...\n", __func__);
            is_interacting = true;  // Pause generation and wait for user input
        }
    }
}
