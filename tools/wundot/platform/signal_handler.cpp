#include "platform/signal_handler.h"

#include "common.h"
#include "console.h"
#include "core/app_context.h"
#include "log.h"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#    include <signal.h>
#    include <unistd.h>
#elif defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

namespace platform {

static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!app::is_interacting && app::g_params->interactive) {
            app::is_interacting  = true;
            app::need_insert_eot = true;
        } else {
            console::cleanup();
            LOG("\n");
            common_perf_print(*app::g_ctx, *app::g_smpl);

            LOG("Interrupted by user\n");
            common_log_pause(common_log_main());

            _exit(130);
        }
    }
}

void setup_sigint_handler() {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = sigint_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
#elif defined(_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
}

}  // namespace platform
