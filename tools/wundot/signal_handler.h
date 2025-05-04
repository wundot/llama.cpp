#ifndef SIGNAL_HANDLER_H
#define SIGNAL_HANDLER_H

#include "log.h"

class SignalHandler {
  public:
    static void setup_signal_handlers();
    static void cleanup_resources();

  private:
    static void sigint_handler(int signo);
};

#endif  // SIGNAL_HANDLER_H
