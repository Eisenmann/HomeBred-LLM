# llama.cpp-hbec — per-expert VRAM cache (Phase 2)

A small patch on top of llama.cpp that lets HomeBred-LLM keep the
**individual** hottest experts of every MoE layer in VRAM. Everything else
keeps running from RAM or disk, as in Phase 1. Stock llama.cpp can only place
a layer's experts all together, because each layer stores them in one fused
tensor.

> This is a private fork for this app. Per llama.cpp's contribution policy
> (`AGENTS.md`), don't submit it upstream.

## How it works

* **At load:** `hbec_set_load_config(slots, device)` arms the cache. For each
  MoE layer whose routed-expert tensors stay in host memory (the planner pins
  them there with tensor overrides), the patch allocates `slots` expert-sized
  slots on the GPU. It also allocates an `expert → slot` map and a residency
  mask.
* **In the graph (`build_moe_ffn`):** the routed experts run on two paths.
  * The **GPU path** runs `mul_mat_id` over the slots, using the slot ids of
    the selected experts.
  * The **CPU path** runs `mul_mat_id` over the host tensors.
  * Each routed pick gets its full router weight on exactly one path and
    exactly 0 on the other, so the result is **bit-identical** to unpatched
    llama.cpp. `test/hbec_test.cpp` checks this.
  * On the CPU path, picks that are already resident are redirected to the
    token's top-1 expert, so the CPU doesn't read extra experts for them.
* **Changing what's resident:** `hbec_set_residency(model, layer, experts[])`
  uploads only the experts that aren't already in a slot. HomeBred-LLM calls it
  under the model's chat lock, so it always runs between decodes. The
  rebalancer uses the learned routing profile to decide which experts to keep.
* **When it's skipped:** layers with expert biases or scales, LoRA adapters,
  and Llama 4 fall back to the normal path.

API: [`include/llama-hbec.h`](hbec.patch) (inside the patch).

## Build

The patch is pinned to llama.cpp `3f7c29d318e3` (b8816). That's the exact
revision LLamaSharp 0.27.0's natives are built from, so LLamaSharp's bindings
keep working.

Windows (CUDA), from a VS 2022 Developer PowerShell with CUDA 12 installed:

```powershell
cd native\llama.cpp-hbec
.\build.ps1 -Backend cuda
```

Linux:

```bash
cd native/llama.cpp-hbec
./build.sh cuda      # or: ./build.sh cpu
```

The script clones llama.cpp, applies `hbec.patch`, builds `llama` and
`mtmd`, and copies the libraries to `HomeBred-LLM/native/`. It then runs the
correctness test on `TestData/test_tiny_moe.gguf`. The app build copies
`native/` to its output folder. At startup, `NativeLibraries.ConfigureCustomLlama`
points LLamaSharp at those libraries (or at `$HOMEBRED_LLAMA_NATIVE_DIR`). The
app then reports "per-expert VRAM cache available", and the planner starts
using expert slots whenever its estimate says they're faster.

## Verified / not verified

* Verified (CPU build, Linux):
  * Logits are bit-identical with the cache on and off, for F32 and Q8_0
    experts, with 0, some, or all experts resident, and when residency changes
    between decodes.
  * A negative control (`HBEC_TEST_SKIP_UPLOAD=1`) changes the logits, which
    proves the slot path is actually used.
  * LLamaSharp 0.27 still loads and runs the patched library.
  * HomeBred's P/Invoke bindings work (`TieringVerificationTests` §13).
* **Not verified:** CUDA, because there was no GPU in the build environment.
  Every op the patch adds (`get_rows` on I32/F32, F32→I32 `cpy`, broadcast
  `sub`/`mul`, `mul_mat_id`) is implemented in ggml-cuda at this revision, but
  run `hbec_test` on your GPU build before relying on it. Turn the cache off
  per model under Config → Memory tiers → "Per-expert VRAM cache".
