# GPU acceleration (CUDA / Vulkan)

HomeBred-LLM ships with the **CPU** llama.cpp backend only, to keep the download small.
Until a GPU backend is installed, the app runs on CPU + RAM (+ disk), the VRAM tier of the
memory planner is inactive, and the Capacity Calculator shows
*"GPU detected, but not usable: this build's llama.cpp backend is CPU-only"*.

Installing a GPU backend is a copy-files step — no rebuild needed.

## Easiest way: the Compute switch

Open **Calculator → Compute** and choose **GPU**. The app checks your system and lists what is
missing, with exact versions and download links:

- GPU and NVIDIA driver version (CUDA 12 needs driver ≥ 527.41 on Windows / ≥ 525.60.13 on Linux)
- Vulkan loader (AMD / Intel)
- the llama.cpp `b8816` GPU build in the `native` folder (the exact path is shown)
- CUDA 12 runtime libraries (NVIDIA)
- whether the GPU backend is already loaded — if the files are in place but the app still runs on
  the CPU backend, it asks you to restart (llama.cpp is loaded once per process)

Press **⬇ Set up GPU backend automatically** and the app downloads the official llama.cpp `b8816`
build for your GPU from GitHub (NVIDIA: CUDA 12.4 build + CUDA runtime DLLs, ~hundreds of MB;
AMD/Intel: Vulkan build), verifies the checksum and unpacks only the libraries into
`%LOCALAPPDATA%\HomeBred-LLM\native` (Linux: `~/.local/share/HomeBred-LLM/native`). Then press
**Restart app**. Automatic setup covers Windows (CUDA, Vulkan) and Linux (Vulkan); llama.cpp
publishes no prebuilt Linux CUDA build, so there use Vulkan or build it
(`native/llama.cpp-hbec/build.sh cuda`).

Press **Re-check** after installing something. **CPU** forces CPU + RAM (+ disk) even when a GPU
backend is installed; the choice is remembered. The manual steps below are what the checker asks for.

## Requirements

| GPU | Backend | You need |
|-----|---------|----------|
| NVIDIA | CUDA 12 | Current NVIDIA driver (CUDA 12 capable) + the CUDA runtime DLLs from the `cudart` archive below |
| AMD / Intel / NVIDIA | Vulkan | Current GPU driver with Vulkan support |

The llama.cpp build **must match the version bundled with LLamaSharp 0.27.0: tag `b8816`**.
Other versions may fail to load or crash.

## Steps (Windows)

1. Open the llama.cpp releases page, tag **b8816**:
   <https://github.com/ggml-org/llama.cpp/releases/tag/b8816>
2. Download the archive for your GPU:
   - NVIDIA: `llama-b8816-bin-win-cuda-12.x-x64.zip` **and** the matching
     `cudart-llama-bin-win-cuda-12.x-x64.zip`
   - Vulkan: `llama-b8816-bin-win-vulkan-x64.zip`
3. Create a folder named `native` next to `HomeBred-LLM.exe`, or use `%LOCALAPPDATA%\HomeBred-LLM\native`
   (or any folder, and set the environment variable `HOMEBRED_LLAMA_NATIVE_DIR` to it).
4. Extract the **DLLs** into it: `llama.dll`, `ggml*.dll` (incl. `ggml-cuda.dll` or `ggml-vulkan.dll`),
   `mtmd.dll` if present, and — for CUDA — the `cudart`/`cublas` DLLs.
   Do not copy the `.exe` tools.
5. Start HomeBred-LLM, open **Calculator** and press the hardware benchmark button
   (the detected backend is cached in the hardware profile until you do).
6. Check that the hardware card shows your GPU and VRAM, and that the VRAM budget is enabled.

Linux: same idea with `libllama.so` and `libggml*.so` from the matching `ubuntu`/`vulkan` archive.

## Troubleshooting

- **Still "CPU-only"** — wrong llama.cpp version, missing CUDA runtime DLLs, or an old driver.
  Remove the `native` folder to return to the bundled CPU backend.
- **App fails to start after copying** — the DLLs do not match b8816; remove the folder.
- **Per-expert VRAM cache** — needs the patched build from `native/llama.cpp-hbec`, not the stock
  archives; the option stays unavailable otherwise.
