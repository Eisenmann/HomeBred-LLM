<div align="center">

# 🏠 HomeBred-LLM

**Cross-platform self-hosted LLM desktop app — no Ollama, no Docker, no Python.**

[![.NET 8](https://img.shields.io/badge/.NET-8.0-512BD4?logo=dotnet)](https://dotnet.microsoft.com/en-us/download/dotnet/8.0)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-0078D4)](https://github.com/your-org/homebred-llm/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![LLamaSharp](https://img.shields.io/badge/inference-LLamaSharp%200.18-blueviolet)](https://github.com/SciSharp/LLamaSharp)
[![CUDA](https://img.shields.io/badge/GPU-CUDA%2012-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Avalonia](https://img.shields.io/badge/UI-Avalonia%2011.2-8B44AC)](https://avaloniaui.net)

[**Download**](#installation) · [**Build from source**](#build-from-source) · [**Contributing**](#contributing)

</div>

---

## What is HomeBred-LLM?

HomeBred-LLM is a cross-platform desktop application that lets you download, run, and chat with large language models entirely on your own machine — on Windows, Linux, or macOS.

It embeds **llama.cpp** in-process via [LLamaSharp](https://github.com/SciSharp/LLamaSharp) — the same inference engine that powers Ollama — without requiring any external runtime, server process, or cloud dependency. Everything from model management to GPU monitoring to the chat UI lives inside one self-contained desktop app built with [Avalonia UI](https://avaloniaui.net).

```
You download a GGUF from HuggingFace → HomeBred-LLM loads it → you chat → GPU metrics are collected → done.
No background services. No terminal. No config files.
```

---

## Features

| | Feature | Detail |
|--|---------|--------|
| 📦 | **Model Library** | Search HuggingFace, browse GGUF files by quantization and size, download with a real-time progress bar |
| ⚙️ | **Model Configuration** | Per-model sliders for temperature, top-p, top-k, repeat penalty, context size, GPU layers, batch size, and system prompt |
| ▶️ | **Start / Stop / Remove** | Load a model into GPU VRAM, unload it without restarting the app, or fully delete the file and database record |
| 📊 | **Live Analytics** | GPU utilization, VRAM used/total, GPU temperature, CPU, RAM, tokens/sec, and time-to-first-token — sampled every 5 seconds |
| 🗑️ | **Analytics Cleanup** | Delete metrics for any model over any date range without touching the model itself |
| 💬 | **Streaming Chat** | Token-by-token streaming output, multiple named sessions, per-message TPS and latency stats |

---

## How it works (no external services)

| Concern | Technology |
|---------|-----------|
| LLM inference | [LLamaSharp](https://github.com/SciSharp/LLamaSharp) — P/Invoke bindings to llama.cpp, runs **in-process** |
| GPU kernels | `LLamaSharp.Backend.Cuda12` (CUDA 12) · `LLamaSharp.Backend.Cpu` (fallback) — bundled NuGet natives |
| Model downloads | `HttpClient` → `huggingface.co/api` — no Python or HF CLI needed |
| GPU monitoring | Dynamic `NativeLibrary` load of `nvml.dll` (Windows) / `libnvidia-ml.so` (Linux) |
| CPU monitoring | `PerformanceCounter` on Windows · `/proc/stat` delta on Linux · GC memory API everywhere |
| Local database | SQLite via EF Core — auto-created at first launch in the platform data directory |
| Background metrics | `System.Timers.Timer` ticking every 5 s |
| Charts | [LiveChartsCore](https://github.com/beto-rodriguez/LiveCharts2) (SkiaSharp, Avalonia backend) |
| UI framework | [Avalonia UI](https://avaloniaui.net) 11.2 + [CommunityToolkit.Mvvm](https://github.com/CommunityToolkit/dotnet) |

---

## Requirements

| | Windows | Linux | macOS |
|--|---------|-------|-------|
| .NET 8.0 | ✅ | ✅ | ✅ |
| NVIDIA GPU (CUDA 12) | optional | optional | — |
| CPU-only fallback | ✅ | ✅ | ✅ (Apple Silicon via Metal planned) |

- **.NET 8.0** — install the [runtime](https://dotnet.microsoft.com/en-us/download/dotnet/8.0) or publish self-contained
- **Disk space** — models range from ~2 GB (Q4 7B) to ~40 GB (Q4 70B)
- **VRAM** — 4 GB minimum for small models; 8–16 GB for 13B+ at full GPU offload

---

## Installation

### Pre-built release *(recommended)*

1. Go to [Releases](https://github.com/your-org/homebred-llm/releases/latest)
2. Download the archive for your platform (`win-x64`, `linux-x64`, or `osx-x64`)
3. Extract and run — no installer required

### Build from source

**Prerequisites:** [.NET 8 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/8.0)

```bash
git clone https://github.com/your-org/homebred-llm.git
cd homebred-llm

# Run in development mode
dotnet run --project HomeBred-LLM

# Publish self-contained for your platform
dotnet publish HomeBred-LLM -c Release -r win-x64  --self-contained -o ./publish/win
dotnet publish HomeBred-LLM -c Release -r linux-x64 --self-contained -o ./publish/linux
dotnet publish HomeBred-LLM -c Release -r osx-x64  --self-contained -o ./publish/mac
```

> **GPU vs CPU build:** Both backends are included by default. LLamaSharp selects CUDA automatically if drivers are present and falls back to CPU otherwise. To ship a CPU-only build, remove the `LLamaSharp.Backend.Cuda12` reference from the `.csproj`.

---

## Project structure

```
HomeBred-LLM/
├── HomeBred-LLM.sln
└── HomeBred-LLM/
    ├── Program.cs               # Avalonia entry point
    ├── App.axaml / App.axaml.cs # Application bootstrap + DI host
    ├── MainWindow.axaml         # Shell with DataTemplate-based navigation
    ├── Models/                  # EF Core entities
    │   ├── LocalModel.cs
    │   ├── ModelConfiguration.cs
    │   ├── AnalyticsMetric.cs
    │   ├── ChatSession.cs + ChatMessage.cs
    │   └── DownloadJob.cs
    ├── Data/
    │   └── AppDbContext.cs      # SQLite context, auto-created on startup
    ├── Services/
    │   ├── LlamaSharpService.cs      # In-process llama.cpp inference
    │   ├── HuggingFaceService.cs     # HF Hub REST client + streaming download
    │   ├── GpuMetricsService.cs      # Cross-platform NVML + CPU sampling
    │   ├── AnalyticsRepository.cs    # Time-series metrics CRUD
    │   └── MetricsCollectorService.cs # Background timer
    ├── ViewModels/              # CommunityToolkit.Mvvm source-generated VMs
    ├── Views/                   # Avalonia .axaml views
    ├── Converters/              # Avalonia value converters
    └── Themes/Dark.axaml        # Dark theme resource dictionary
```

---

## Supported models

Any model available as a GGUF file on HuggingFace works, including:

- **LLaMA 3 / 3.1 / 3.2 / 3.3** (Meta)
- **Mistral / Mixtral** (Mistral AI)
- **Qwen 2 / 2.5** (Alibaba)
- **Phi-3 / Phi-4** (Microsoft)
- **Gemma 2 / 3** (Google)
- **DeepSeek** (DeepSeek AI)
- Any other architecture supported by llama.cpp

Recommended quantizations for a balance of quality and speed: `Q4_K_M`, `Q5_K_M`, `Q6_K`.

---

## Analytics collected

| Metric | Source | Interval |
|--------|--------|----------|
| GPU utilization % | NVML `nvmlDeviceGetUtilizationRates` | 5 s |
| VRAM used / total | NVML `nvmlDeviceGetMemoryInfo` | 5 s |
| GPU temperature | NVML `nvmlDeviceGetTemperature` | 5 s |
| CPU utilization % | `PerformanceCounter` (Win) · `/proc/stat` delta (Linux) | 5 s |
| RAM used / total | GC memory API (cross-platform) | 5 s |
| Tokens per second | LLamaSharp eval timing | Per request |
| Time to first token | Wall-clock from send to first token | Per request |
| Total inference time | Wall-clock end-to-end | Per request |
| Prompt / output token counts | LLamaSharp token counter | Per request |

All metrics are stored in SQLite and queryable by model and date range. You can delete any selection from the Analytics screen.

---

## Data storage

| Platform | Path |
|----------|------|
| Windows | `%LOCALAPPDATA%\HomeBred-LLM\` |
| Linux | `~/.local/share/HomeBred-LLM/` |
| macOS | `~/Library/Application Support/HomeBred-LLM/` |

```
HomeBred-LLM/
├── homebred.db    # SQLite — models, configs, analytics, chat history
└── models/        # Downloaded GGUF files
```

Uninstalling is deleting the binary and this folder — no registry entries, no system-wide changes.

---

## Contributing

Contributions are welcome. Please open an issue before starting significant work so we can coordinate.

```bash
# Fork, then clone your fork
git clone https://github.com/your-username/homebred-llm.git
cd homebred-llm

# Create a feature branch
git checkout -b feature/my-improvement

# Build and run
dotnet build HomeBred-LLM
dotnet run --project HomeBred-LLM

# Push and open a PR
```

### Good first issues

- ROCm (AMD GPU) support via `LLamaSharp.Backend.OpenCL`
- Apple Silicon / Metal backend once LLamaSharp ships `LLamaSharp.Backend.Metal`
- Multi-GPU model splitting UI
- Export analytics to CSV / JSON
- Model import from a local GGUF path (without HuggingFace)
- Session search and message export

---

## Acknowledgements

HomeBred-LLM is built on the shoulders of several excellent open-source projects:

- [**llama.cpp**](https://github.com/ggml-org/llama.cpp) — the inference engine underneath everything
- [**LLamaSharp**](https://github.com/SciSharp/LLamaSharp) — C# bindings that make llama.cpp usable from .NET
- [**Avalonia UI**](https://github.com/AvaloniaUI/Avalonia) — cross-platform .NET UI framework
- [**LiveCharts2**](https://github.com/beto-rodriguez/LiveCharts2) — analytics charts
- [**CommunityToolkit.Mvvm**](https://github.com/CommunityToolkit/dotnet) — source-generated MVVM
- [**Entity Framework Core**](https://github.com/dotnet/efcore) — SQLite persistence

---

## License

[MIT](LICENSE) © 2025 HomeBred-LLM contributors
