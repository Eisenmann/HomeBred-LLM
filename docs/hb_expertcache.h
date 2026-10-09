/*
 * hb_expertcache — proposed C ABI for HomeBred-LLM's Phase 2 per-expert VRAM
 * cache (see docs/tiered-memory-architecture.md). NOT IMPLEMENTED YET.
 *
 * Build against the exact llama.cpp revision LLamaSharp ships. The library
 * replaces MUL_MAT_ID for routed-expert tensors with a slot-indirected kernel:
 * each layer owns `slots` GPU expert slots; a residency table maps
 * (layer, expert) -> slot or -1. Misses are computed on the CPU from the
 * memory-mapped (warm/cold) weights; hot misses are promoted asynchronously on
 * a dedicated stream. The managed side (NativeExpertCacheBackend) only needs
 * the functions below.
 */
#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hbec_cache hbec_cache;

typedef struct {
    int32_t  n_layers;
    int32_t  n_experts;
    int32_t  slots_per_layer;   /* chosen by the planner from the VRAM budget */
    int32_t  prefetch_lookahead; /* 0 = off, 1 = gate look-ahead for layer N+1 */
    int32_t  device;            /* backend device index (CUDA0 = 0) */
} hbec_params;

typedef struct {
    int64_t hits, misses, promotions, evictions, bytes_uploaded;
} hbec_counters;

int32_t      hbec_version(void);

/* Attach to a loaded llama_model / llama_context (opaque pointers from llama.h). */
hbec_cache * hbec_attach(void * llama_model, void * llama_context, const hbec_params * params);
void         hbec_detach(hbec_cache * cache);

/* Seed residency with the hottest experts (pairs of layer, expert). */
void         hbec_prefill(hbec_cache * cache, const int32_t * layer_expert_pairs, int32_t n_pairs);

/* Per-expert routing priority used by the eviction policy (e.g. decayed counts). */
void         hbec_set_priority(hbec_cache * cache, int32_t layer, const float * priority, int32_t n_experts);

/* Counters since the previous call. */
hbec_counters hbec_drain_counters(hbec_cache * cache);

#ifdef __cplusplus
}
#endif
