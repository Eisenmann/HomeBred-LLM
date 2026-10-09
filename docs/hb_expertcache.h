// Copy of the C API added by native/llama.cpp-hbec/hbec.patch (include/llama-hbec.h).
// HomeBred-LLM expert cache ("hbec") — per-expert VRAM slots for MoE models.
//
// Private patch on top of llama.cpp (not an upstream feature). For each MoE
// layer whose routed-expert tensors live in host memory, a small set of
// "slots" (copies of individual experts) is allocated on a GPU buffer. The
// MoE graph then runs two paths: resident experts are computed from the GPU
// slots, the rest from the host tensors on the CPU, and the results are
// combined with the routing weights so the output is identical to the
// unpatched computation. Which experts are resident is decided by the caller
// (HomeBred-LLM's rebalancer) via hbec_set_residency between decode calls.
#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HBEC_VERSION 1

#define HBEC_DEVICE_FIRST_GPU (-1) // first non-CPU device of the model
#define HBEC_DEVICE_CPU       (-2) // host buffer (testing: exercises the graph path without a GPU)

LLAMA_API int32_t hbec_version(void);

// Configures the expert cache for the NEXT llama_model_load_from_file* call in
// this process (consumed by it). slots_per_layer <= 0 disables the cache.
LLAMA_API void    hbec_set_load_config(int32_t slots_per_layer, int32_t device);

// Number of layers with an expert cache (0 = cache not active for this model).
LLAMA_API int32_t hbec_model_layers(const struct llama_model * model);
LLAMA_API int32_t hbec_model_slots (const struct llama_model * model);
LLAMA_API int64_t hbec_model_bytes (const struct llama_model * model);
LLAMA_API bool    hbec_layer_enabled(const struct llama_model * model, int32_t il);

// Makes `experts` (up to slots_per_layer ids) resident in layer il's slots.
// Experts already resident keep their slot; only missing ones are uploaded.
// Must not run concurrently with a decode on any context of this model.
// Returns the number of experts uploaded, or a negative error code.
LLAMA_API int32_t hbec_set_residency(struct llama_model * model, int32_t il, const int32_t * experts, int32_t n_experts);

// Writes the expert id held by each slot (-1 = empty); returns the slot count.
LLAMA_API int32_t hbec_get_residency(const struct llama_model * model, int32_t il, int32_t * out_experts, int32_t capacity);

// Counters accumulated since the previous call.
LLAMA_API void    hbec_drain_counters(struct llama_model * model, int64_t * promotions, int64_t * evictions, int64_t * bytes_uploaded);

#ifdef __cplusplus
}
#endif
