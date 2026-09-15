#!/usr/bin/env bash
# Build the P2P all-reduce extension (bf16, NVLink peer-to-peer).
#
#   ./build.sh            # builds ./libp2p_ar.so next to this script
#
# Notes
#  * B300 is SM103 and DeepGEMM's sm_103 path needs the **long-form gencode**
#    (`-gencode arch=compute_103a,code=sm_103a`); the short `-arch=sm_103a`
#    loses the `a` suffix in fatbin mode on CUDA 13.2 (see AGENTS §8).
#  * No dependency on sgl-kernel: this is a standalone .so loaded with ctypes,
#    so enabling the feature does not require rebuilding sglang.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/libp2p_ar.so}"
NVCC="${NVCC:-nvcc}"

ARCH_FLAGS=()
# NOTE: probe the *configured* compiler, and never pipe its help straight into
# `grep -q`: grep exits on first match, nvcc then dies of SIGPIPE (141) and with
# `set -o pipefail` the whole pipeline looks like a failure -- which silently
# dropped the sm_103a gencode and produced a .so that cannot run on SM103 (the
# build "succeeded"; the failure only showed up at load time). Capture first.
NVCC_HELP="$("$NVCC" --help 2>&1 || true)"
if grep -q "compute_103a" <<<"$NVCC_HELP"; then
  ARCH_FLAGS+=("-gencode" "arch=compute_103a,code=sm_103a")
else
  echo "WARNING: $NVCC has no compute_103a support -> SM103 (B300) is NOT covered" >&2
fi
# common fallbacks so the .so also builds on other boxes
ARCH_FLAGS+=("-gencode" "arch=compute_90a,code=sm_90a")
ARCH_FLAGS+=("-gencode" "arch=compute_80,code=sm_80")

set -x
"$NVCC" -O3 -std=c++17 --use_fast_math -Xcompiler -fPIC -shared \
  "${ARCH_FLAGS[@]}" \
  -I"$HERE" \
  -o "$OUT" "$HERE/p2p_ar_bf16.cu"
set +x
echo "built: $OUT"
ls -la "$OUT"
