#include "llama_main.h"

int main(int argc, char ** argv) {
    llama::LlamaApp app(argc, argv);
    return app.run();
}
