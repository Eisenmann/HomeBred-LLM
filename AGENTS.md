# AGENTS.md — building, running and testing HomeBred-LLM

Guide for AI coding agents (and humans) working in this repository. Read it before changing code.
Everything here was checked against the repo; where something could not be verified, it says so.

## What this is

HomeBred-LLM is a desktop app (Avalonia 11, .NET 10, C#) that runs GGUF models in-process through
llama.cpp (LLamaSharp 0.27.0). Large models are split across VRAM, RAM and disk ("tiered memory").
Data: SQLite via EF Core 8.

| Path | Purpose |
|------|---------|
| `HomeBred-LLM/` | The app. `Services/Tiering/` = tier planner, capacity calculator, GPU checker/installer; `Services/Llama/` = inference; `ViewModels/`, `Views/` = UI |
| `TieringVerificationTests/` | Main test suite (console app, 100+ checks, incl. native llama.cpp tests) |
| `GgufVerificationTests/` | GGUF reader / DB migration checks |
| `UiSmokeTests/` | Headless UI run: boots the real app without a display and saves one PNG per screen |
| `native/llama.cpp-hbec/` | Patched llama.cpp (per-expert VRAM cache) — build scripts, patch, C++ test |
| `docs/` | `tiered-memory-architecture.md` (design), `gpu-setup.md` (GPU for users) |
| `.github/workflows/build.yml` | CI: self-contained publish for linux-x64, win-x64, osx-arm64, osx-x64 |

There is **no xUnit/NUnit**. Each test project is a console app: it prints `PASS:`/`FAIL:` lines and its
**exit code is the number of failures** (0 = good).

## Prerequisites

- .NET SDK **10.0** (`dotnet --version`). Nothing else is needed for build and the tests.
- No GPU needed. The app bundles only the **CPU** llama.cpp backend on purpose (small download).
  Do **not** add `LLamaSharp.Backend.Cuda12/Vulkan` package references: GPU support is installed by the
  user at runtime (Calculator → Compute → GPU, see `docs/gpu-setup.md`).
- Optional, only for the native expert-cache work: CMake, a C++ compiler, git (and CUDA/Vulkan SDK for GPU builds).

## Build

```bash
dotnet build HomeBred-LLM.sln          # everything
dotnet build HomeBred-LLM/HomeBred-LLM.csproj   # app only
```

**NuGet gotcha:** `HomeBred-LLM/nuget.config` lists a Windows-only local feed
(`C:\Users\user\source\repos\AIHelpers\local-nuget`). On any other machine restore fails with `NU1301`.
Override the config instead of editing the file:

```bash
dotnet restore HomeBred-LLM.sln --configfile .github/nuget.ci.config     # nuget.org only (CI config)
dotnet build   HomeBred-LLM.sln --no-restore
```

In a sandbox without internet to nuget.org, point `--configfile` at a config whose only source is a local
folder of `.nupkg` files, restore with it, then build with `--no-restore`. If `dotnet run`/`build` complains
about `api.nuget.org` after the project file changed, the assets file is stale: restore again with `--configfile`.

Publish like CI does (self-contained, one RID):

```bash
dotnet publish HomeBred-LLM/HomeBred-LLM.csproj -c Release -r win-x64 --self-contained -o publish/win-x64
```

The build copies `HomeBred-LLM/native/**` to the output **only if that folder exists** (locally built
llama.cpp). It is git-ignored, so CI never has it — keep it that way.

## Run the app

```bash
dotnet run --project HomeBred-LLM
```

Needs a desktop session. **Agents without a display: use `UiSmokeTests` instead** (below).

App data (database, downloaded models, in-app GPU backend) lives in `%LOCALAPPDATA%\HomeBred-LLM`
(Linux: `~/.local/share/HomeBred-LLM`). **Automated runs must not touch the real data:** set
`HOMEBRED_DATA_DIR` to a temp folder first (`UiSmokeTests` does this itself).

Useful environment variables:

| Variable | Effect |
|----------|--------|
| `HOMEBRED_DATA_DIR` | Relocates all app data |
| `HOMEBRED_LLAMA_NATIVE_DIR` | Load llama.cpp libraries from this folder instead of the bundled CPU backend |
| `HF_TOKEN` | Hugging Face token for gated/private repos |

## Test

Run from the repo root. All of these were run successfully in a Linux CPU-only sandbox unless noted.

### 1. TieringVerificationTests — main suite

```bash
dotnet run --project TieringVerificationTests
```

Expect `ALL PASSED` and exit code 0. Sections: size tables, catalog, estimators, planner, capacity calculator,
usage profile, warm tier, hardware probe, schema upgrade, **native** tiered load + routing profiler (11),
expert-slot planner (12), **native** expert cache (13), GPU installer/checker.

- Needs `HomeBred-LLM/TestData/*.gguf` (found relative to the build output).
- Section 11 prints `SKIP` when no llama.cpp native library can load; section 13 prints `SKIP` with stock
  llama.cpp ("no hbec_* exports"). A `SKIP` is not a failure — but say so in your report instead of claiming
  those sections passed. To run 13, build the patched library first (see Native below).
- Add a check for every behavior you change; use the existing `Check(cond, "message")` helper.

### 2. GgufVerificationTests

```bash
dotnet run --project GgufVerificationTests
```

GGUF reader and the DB migration check (exit code = number of failed checks; also read the `FAIL:` lines).
In the Linux sandbox it exits 1 because one check, `native log text surfaced in exception message`, fails because no llama.cpp
native library could be loaded (`TypeInitializationException` in `NativeApi`). That is an environment
limit, not a regression, but it was not verified on a machine with the native backend loadable. The "real
model file" step is skipped when the hard-coded Windows path does not exist.

### 3. UiSmokeTests — see the UI without a display

```bash
dotnet run --project UiSmokeTests                      # PNGs → artifacts/ui-shots/
dotnet run --project UiSmokeTests -- /some/out/dir     # custom folder
```

It uses a temporary data dir, seeds two test GGUF models, opens the Model Library, starts the tiny MoE model,
visits Calculator (dense, MoE, GPU check), Config and Analytics, and saves `1-models.png` … `6-gpu-check.png`.
**Open the PNGs** (image viewer / Read tool) after any UI change — compiling is not enough; XAML binding
mistakes only show at runtime. It prints Avalonia binding warnings at the end; the expected count is **0**, so any warning is yours.
Exit code 1 means an exception during the run; 2 means test data is missing.

On Linux the project pulls SkiaSharp/HarfBuzz native assets automatically (Windows/macOS need nothing).

### 4. Native expert cache (only when touching `native/llama.cpp-hbec/`)

```bash
cd native/llama.cpp-hbec
./build.sh cpu          # or: ./build.sh cuda | vulkan      (Windows: .\build.ps1 -Backend cuda)
```

Clones llama.cpp at the pinned commit (`3f7c29d3…`, b8816 — must match LLamaSharp 0.27.0), applies
`hbec.patch`, builds, copies libraries to `HomeBred-LLM/native/`, and runs `test/hbec_test.cpp`
(logits must be identical with the cache on/off; the negative control must differ). Then rerun
`TieringVerificationTests` — section 13 now runs. **Verified: CPU build on Linux. Not verified: CUDA, and
`build.ps1` has never been executed.** Do not claim GPU results you did not measure.

## Before you say "done"

1. `dotnet build HomeBred-LLM.sln` is clean (no new warnings from your files).
2. `dotnet run --project TieringVerificationTests` → `ALL PASSED`; list any `SKIP`s.
3. UI changed → run `UiSmokeTests` and **look at** the relevant PNG(s).
4. DB model changed → see "Conventions" (schema) and extend section 10 of the tiering tests.
5. State plainly what you could not test (GPU, Windows, real downloads).

## Conventions and traps

- **Schema changes:** the DB is created with `EnsureCreated`, and older databases are upgraded by
  `AppDbContextSchemaReconciler` (`Data/AppDbContext.cs`). A new column needs **both** the `CREATE TABLE` text
  and an `ALTER TABLE … ADD COLUMN … DEFAULT` for existing databases. No EF migrations.
- **LiveCharts:** axis arrays must never be empty (it throws) — keep `new Axis()` defaults. Dark theme needs
  explicit legend/label paints (`ViewModels/ChartTheme.cs`).
- **Two truths about the GPU:** NVML says what card is installed (`GpuMetricsService`); the llama.cpp backend
  says whether it can be used (`HardwareProbe.GetBackendDevices`). VRAM budgets must follow the backend
  (`HardwareSpec.HasGpuBackend`), and the user's CPU/GPU choice (`HardwareProfile.ComputeMode`) can force CPU.
- **llama.cpp loads once per process.** Changing native libraries needs an app restart. Load natives only
  after `NativeLibraries.ConfigureCustomLlama()` (called first in `Program.Main`); any code that touches
  `NativeApi` earlier will lock in the wrong backend.
- **Never rebuild a file from stale output.** Edit in place and keep the file's encoding (many files have a
  BOM and CRLF line endings).
- **Hugging Face tab is ONNX-only** (search filter `onnx`); downloads fetch a whole variant folder via
  `HuggingFaceService.DownloadDirectoryAsync`.
- `*.gguf` is git-ignored. Test files in `HomeBred-LLM/TestData/` must be added with `git add -f`.
- Estimates (speed, fit) are models, not measurements. Calibration is stored in `HardwareProfile`.
- Do not commit or push unless the user asks; the owner commits their own work.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `NU1301` local source `C:\Users\…\local-nuget` | Use `--configfile .github/nuget.ci.config` (see Build) |
| `TypeInitializationException … NativeApi` | No loadable llama.cpp native library; check `HOMEBRED_LLAMA_NATIVE_DIR`, or the runtime backend package restore |
| "No GPU device in the llama.cpp backend" while a GPU is present | Only the CPU backend is loaded — Calculator → Compute → GPU, or `docs/gpu-setup.md` |
| UI test shows a blank/odd screen | Check the printed Avalonia warnings; look for missing `DataContext` or a bad binding |
| `hbec_*` section skipped | Stock llama.cpp loaded; build `native/llama.cpp-hbec` |
| Download 404 on Hugging Face | Wrong repo/variant path, or gated repo without `HF_TOKEN` |
