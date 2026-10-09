// Correctness test for the HomeBred expert cache patch: logits with the cache
// active (any residency) must match logits without it.
// Usage: hbec_test <moe.gguf>
#include "llama.h"
#include "llama-hbec.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> run(const char * path, int slots, const std::vector<std::vector<int>> * residency, int * cached_layers) {
    hbec_set_load_config(slots, HBEC_DEVICE_CPU);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); exit(2); }
    *cached_layers = hbec_model_layers(model);
    if (residency) {
        for (int il = 0; il < (int) residency->size(); ++il) {
            if (!hbec_layer_enabled(model, il)) continue;
            const auto & r = (*residency)[il];
            int n = hbec_set_residency(model, il, r.data(), (int) r.size());
            if (n < 0) { fprintf(stderr, "set_residency failed %d\n", n); exit(3); }
        }
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 256; cp.n_ubatch = 32; cp.n_threads = 2; cp.n_threads_batch = 2;
    llama_context * ctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * text = "hello world the model is an expert of the world and the model";
    std::vector<llama_token> toks(128);
    int n = llama_tokenize(vocab, text, (int) strlen(text), toks.data(), (int) toks.size(), true, false);
    toks.resize(n);
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        batch.token[i] = toks[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = 1;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); exit(4); }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<float> out((size_t) n * n_vocab);
    for (int i = 0; i < n; ++i) memcpy(out.data() + (size_t) i * n_vocab, llama_get_logits_ith(ctx, i), n_vocab * sizeof(float));

    // re-target residency between decodes (exercises evictions and the diff logic)
    if (residency && getenv("HBEC_TEST_RETARGET")) {
        for (int il = 0; il < (int) residency->size(); ++il) {
            if (!hbec_layer_enabled(model, il)) continue;
            std::vector<int> r = { 6, 2, 0 };
            hbec_set_residency(model, il, r.data(), (int) r.size());
        }
    }
    // single-token decode step too (the generation path)
    llama_batch one = llama_batch_init(1, 0, 1);
    one.token[0] = toks[1]; one.pos[0] = n; one.n_seq_id[0] = 1; one.seq_id[0][0] = 0; one.logits[0] = 1; one.n_tokens = 1;
    if (llama_decode(ctx, one) != 0) { fprintf(stderr, "decode 1 failed\n"); exit(5); }
    const float * l1 = llama_get_logits_ith(ctx, 0);
    out.insert(out.end(), l1, l1 + n_vocab);

    int64_t pr, ev, by; hbec_drain_counters(model, &pr, &ev, &by);
    if (residency) printf("  counters: promotions=%lld evictions=%lld bytes=%lld\n", (long long) pr, (long long) ev, (long long) by);
    llama_batch_free(batch); llama_batch_free(one);
    llama_free(ctx);
    llama_model_free(model);
    return out;
}

static double maxdiff(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i])); return m;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [--negative-control]\n", argv[0]); return 1; }
    if (argc > 2 && strcmp(argv[2], "--negative-control") == 0) {
        // slots left zeroed: results MUST differ, otherwise the slot path isn't being used
        llama_backend_init();
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
        int layers = 0;
        auto base = run(argv[1], 0, nullptr, &layers);
        std::vector<std::vector<int>> all(4, std::vector<int>{0,1,2,3,4,5,6,7});
        auto got = run(argv[1], 8, &all, &layers);
        double d = maxdiff(base, got);
        printf("%s: negative control differs (max |dlogit|=%.3g)\n", d > 1e-3 ? "PASS" : "FAIL", d);
        return d > 1e-3 ? 0 : 1;
    }
    llama_backend_init();
    llama_log_set([](ggml_log_level, const char * t, void *) { if (strstr(t, "hbec") || strstr(t, "expert cache")) fputs(t, stdout); }, nullptr);
    printf("hbec_version=%d\n", hbec_version());
    int layers = 0, fails = 0;
    auto base = run(argv[1], 0, nullptr, &layers);
    printf("baseline: cached layers=%d\n", layers);
    if (layers != 0) { printf("FAIL: cache active without config\n"); fails++; }

    const int L = 4;
    std::vector<std::vector<int>> none(L), some(L), all(L), other(L);
    for (int l = 0; l < L; ++l) { some[l] = {0, 3, 5}; all[l] = {0,1,2,3,4,5,6,7}; other[l] = {7, 1}; }

    struct C { const char * name; int slots; std::vector<std::vector<int>> * r; } cases[] = {
        {"slots=4 none resident", 4, &none},
        {"slots=4 three resident", 4, &some},
        {"slots=8 all resident", 8, &all},
        {"slots=2 two resident", 2, &other},
    };
    for (auto & c : cases) {
        auto got = run(argv[1], c.slots, c.r, &layers);
        double d = maxdiff(base, got);
        bool ok = layers == L && d < 1e-4;
        printf("%s: %s (cached layers=%d, max |dlogit|=%.3g)\n", ok ? "PASS" : "FAIL", c.name, layers, d);
        if (!ok) fails++;
    }
    // re-target between decodes
    setenv("HBEC_TEST_RETARGET", "1", 1);
    {
        auto got = run(argv[1], 4, &some, &layers);
        double d = maxdiff(base, got);
        bool ok = d < 1e-4;
        printf("%s: re-target residency between decodes (max |dlogit|=%.3g)\n", ok ? "PASS" : "FAIL", d);
        if (!ok) fails++;
    }
    unsetenv("HBEC_TEST_RETARGET");
    printf(fails ? "%d FAILURE(S)\n" : "ALL PASSED\n", fails);
    return fails;
}
