#!/usr/bin/env bash
# Builds the HomeBred expert-cache patched llama.cpp and copies the shared
# libraries into HomeBred-LLM/native/, which the app loads instead of the
# LLamaSharp backend package (see HomeBred-LLM/NativeLibraries.cs).
#
#   ./build.sh            # CPU only
#   ./build.sh cuda       # NVIDIA (needs the CUDA toolkit)
#   ./build.sh vulkan     # Vulkan SDK
set -euo pipefail

# Exact llama.cpp revision LLamaSharp 0.27.0's natives are built from
# (LLamaSharpBinaries release 3f7c29d318e317b6). Must match: LLamaSharp binds
# to this ABI.
LLAMA_COMMIT=3f7c29d318e317b63f54c558bc69803963d7d88c
BACKEND="${1:-cpu}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$HERE/.work"
OUT="$HERE/../../HomeBred-LLM/native"

mkdir -p "$WORK"
if [ ! -d "$WORK/llama.cpp/.git" ]; then
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$WORK/llama.cpp"
fi
cd "$WORK/llama.cpp"
git fetch --quiet origin "$LLAMA_COMMIT" || true
git checkout --quiet --force "$LLAMA_COMMIT"
git clean -fdq src include
git apply --whitespace=nowarn "$HERE/hbec.patch"

EXTRA=()
case "$BACKEND" in
  cuda)   EXTRA+=(-DGGML_CUDA=ON) ;;
  vulkan) EXTRA+=(-DGGML_VULKAN=ON) ;;
  cpu)    ;;
  *) echo "unknown backend: $BACKEND" >&2; exit 1 ;;
esac

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_CURL=OFF -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH='$ORIGIN' "${EXTRA[@]}"
cmake --build build -j"$(nproc)" --target llama mtmd

mkdir -p "$OUT"
cp -a build/bin/lib*.so* "$OUT/"

# Correctness check: logits must be identical with the expert cache on/off.
g++ -O2 -std=c++17 "$HERE/test/hbec_test.cpp" -Iinclude -Iggml/include -Lbuild/bin \
  -lllama -lggml -lggml-base -Wl,-rpath,"$PWD/build/bin" -o build/hbec_test
MODEL="$HERE/../../HomeBred-LLM/TestData/test_tiny_moe.gguf"
if [ -f "$MODEL" ]; then build/hbec_test "$MODEL"; fi

echo "Installed patched llama.cpp ($BACKEND) into $OUT"
