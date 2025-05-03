#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "llama_utils.h"

#include "doctest.h"
#include "llama.h"
#include "sampling.h"

TEST_CASE("TokenSampler basic functionality") {
    llama_model * model = llama_load_model("models/test-model.gguf");
    REQUIRE(model != nullptr);

    sampling_params sparams{};
    sparams.seed = 42;

    llama::TokenSampler sampler(model, sparams);
    llama_context *     ctx = llama_new_context(model);
    REQUIRE(ctx != nullptr);

    llama_token tok = sampler.sample(ctx);
    sampler.accept(tok, true);

    llama_free(ctx);
    llama_free(model);
}

TEST_CASE("SessionCache save and load") {
    llama_model *       model = llama_load_model("models/test-model.gguf");
    llama_context *     ctx   = llama_new_context(model);
    llama::SessionCache cache;

    std::vector<llama_token> tokens = { 1, 2, 3, 4 };
    CHECK(cache.save("test_session.bin", ctx, tokens));

    std::vector<llama_token> loaded;
    CHECK(cache.load("test_session.bin", ctx, loaded));
    CHECK_EQ(tokens, loaded);

    llama_free(ctx);
    llama_free(model);
}

TEST_CASE("ChatFormatter formats messages") {
    llama_model *        model = llama_load_model("models/test-model.gguf");
    llama::ChatFormatter formatter(model, "chat_templates/default.tpl");

    std::string result1 = formatter.format_system("You are a helper.");
    std::string result2 = formatter.format_user("What is AI?");

    CHECK(!result1.empty());
    CHECK(!result2.empty());

    llama_free(model);
}
