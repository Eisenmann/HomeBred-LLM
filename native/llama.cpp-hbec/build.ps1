# Builds the HomeBred expert-cache patched llama.cpp on Windows and copies the
# DLLs into HomeBred-LLM\native\, which the app loads instead of the LLamaSharp
# backend package (see HomeBred-LLM\NativeLibraries.cs).
#
# Requirements: Visual Studio 2022 (C++ workload), CMake, Git, and for CUDA the
# NVIDIA CUDA Toolkit 12.x. Run from a "x64 Native Tools" / Developer PowerShell.
#
#   .\build.ps1 -Backend cuda
#   .\build.ps1 -Backend cpu
param(
    [ValidateSet("cuda", "vulkan", "cpu")] [string] $Backend = "cuda"
)
$ErrorActionPreference = "Stop"

# Exact llama.cpp revision LLamaSharp 0.27.0's natives are built from.
$LlamaCommit = "3f7c29d318e317b63f54c558bc69803963d7d88c"
$Here = $PSScriptRoot
$Work = Join-Path $Here ".work"
$Out  = Join-Path $Here "..\..\HomeBred-LLM\native"
$Src  = Join-Path $Work "llama.cpp"

New-Item -ItemType Directory -Force -Path $Work | Out-Null
if (-not (Test-Path (Join-Path $Src ".git"))) {
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp $Src
}
Push-Location $Src
try {
    git fetch --quiet origin $LlamaCommit 2>$null
    git checkout --quiet --force $LlamaCommit
    git clean -fdq src include
    git apply --whitespace=nowarn (Join-Path $Here "hbec.patch")

    $extra = @()
    if ($Backend -eq "cuda")   { $extra += "-DGGML_CUDA=ON" }
    if ($Backend -eq "vulkan") { $extra += "-DGGML_VULKAN=ON" }

    cmake -S . -B build -A x64 -DBUILD_SHARED_LIBS=ON `
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON `
        -DLLAMA_CURL=OFF -DGGML_NATIVE=ON @extra
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
    cmake --build build --config Release --target llama mtmd -j
    if ($LASTEXITCODE -ne 0) { throw "build failed" }

    New-Item -ItemType Directory -Force -Path $Out | Out-Null
    Copy-Item build\bin\Release\*.dll $Out -Force

    # Correctness check: logits must be identical with the expert cache on/off.
    $model = Join-Path $Here "..\..\HomeBred-LLM\TestData\test_tiny_moe.gguf"
    cl /nologo /O2 /std:c++17 /EHsc (Join-Path $Here "test\hbec_test.cpp") /Iinclude /Iggml\include `
        /Fe:build\hbec_test.exe /link /LIBPATH:build\src\Release /LIBPATH:build\ggml\src\Release llama.lib ggml.lib ggml-base.lib
    if ($LASTEXITCODE -eq 0 -and (Test-Path $model)) {
        $env:PATH = "$(Resolve-Path build\bin\Release);$env:PATH"
        .\build\hbec_test.exe $model
    }
    Write-Host "Installed patched llama.cpp ($Backend) into $Out"
}
finally { Pop-Location }
