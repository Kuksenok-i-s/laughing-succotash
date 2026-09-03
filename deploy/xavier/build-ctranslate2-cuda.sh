#!/usr/bin/env bash
# Build CTranslate2 4.8.1 with CUDA 11.4 / sm_72 for Jetson AGX Xavier (JetPack 5).
#
# PyPI aarch64 wheels are CPU-only. nvcc is not on the Xavier host; it lives in
# nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3. Flash-attention stays off (CUDA 11.4
# cannot compile it). mean_gpu.cu is patched: 11.4 has no bfloat16 /= float.
#
# Installs the C++ library to ~/.local/opt/ctranslate2 and the Python bindings
# into ~/.assistant/venv-whisper. gpu-transcriber-xavier.service puts that prefix
# on LD_LIBRARY_PATH.
set -euo pipefail

CTRANSLATE_VERSION=4.8.1
IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
SRC=${CTRANSLATE2_SRC:-$HOME/.local/src/CTranslate2}
PREFIX=${CTRANSLATE2_PREFIX:-$HOME/.local/opt/ctranslate2}
VENV=${WHISPER_VENV:-$HOME/.assistant/venv-whisper}
JOBS=${JOBS:-$(($(nproc) - 1))}
if (( JOBS < 1 )); then JOBS=1; fi

HERE=$(cd "$(dirname "$0")" && pwd)
PATCH=$HERE/ctranslate2-v4.8.1-mean_gpu-cuda114.patch

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "missing whisper venv at $VENV" >&2
  exit 1
fi
if [[ ! -f "$PATCH" ]]; then
  echo "missing patch $PATCH" >&2
  exit 1
fi

mkdir -p "$(dirname "$SRC")" "$PREFIX"

if [[ ! -f "$SRC/CMakeLists.txt" ]]; then
  echo "Cloning CTranslate2 v${CTRANSLATE_VERSION}..."
  if ! git clone --branch "v${CTRANSLATE_VERSION}" --recursive --depth 1 \
      https://github.com/OpenNMT/CTranslate2.git "$SRC"; then
    echo "GitHub clone failed. Rsync a v${CTRANSLATE_VERSION} tree (with submodules) to $SRC" >&2
    exit 1
  fi
fi

# Idempotent: already-patched trees skip cleanly.
if grep -q 'CUDA 11.4 has no bfloat16' "$SRC/src/ops/mean_gpu.cu"; then
  echo "mean_gpu.cu already patched"
else
  echo "Patching mean_gpu.cu for CUDA 11.4..."
  patch -p1 -d "$SRC" < "$PATCH"
fi

echo "Compiling with $IMAGE (-j$JOBS)..."
docker run --rm \
  -v "$SRC:/src" \
  -v "$PREFIX:/opt/ct2" \
  "$IMAGE" \
  bash -lc "
    set -eux
    cd /src
    rm -rf build
    mkdir build
    cd build
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_CUDA=ON \
      -DWITH_CUDNN=ON \
      -DWITH_MKL=OFF \
      -DWITH_FLASH_ATTN=OFF \
      -DOPENMP_RUNTIME=COMP \
      -DBUILD_CLI=OFF \
      -DCUDA_ARCH_LIST=7.2 \
      -DCMAKE_INSTALL_PREFIX=/opt/ct2
    make -j$JOBS
    make install
    ls -lh /opt/ct2/lib
  "

export CTRANSLATE2_ROOT=$PREFIX
export LD_LIBRARY_PATH=$PREFIX/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

echo "Building Python 3.13 bindings..."
"$VENV/bin/pip" install -r "$SRC/python/install_requirements.txt"
"$VENV/bin/pip" uninstall -y ctranslate2 || true
"$VENV/bin/pip" install --no-build-isolation "$SRC/python"

echo "Verifying CUDA backend..."
"$VENV/bin/python" - << 'PY'
import ctranslate2

print("version", ctranslate2.__version__)
types = ctranslate2.get_supported_compute_types("cuda")
print("cuda_types", types)
if "float16" not in types:
    raise SystemExit("CUDA backend has no float16")
PY
echo "OK: CTranslate2 ${CTRANSLATE_VERSION} CUDA is in $PREFIX and $VENV"
