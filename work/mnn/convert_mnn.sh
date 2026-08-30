#!/bin/bash
# ONNX -> MNN, for the roadmap 6 backend bake-off against ncnn.
#
# Why MNN at all: ncnn's Vulkan lost to its own NEON on 2dfan4 and arcface, which says the
# GPU column has headroom rather than that the GPU is weak.  MNN reaches Adreno through
# OPENCL rather than Vulkan, and that is the cheapest way to find out whether the ceiling
# is the GPU or ncnn's use of it.  If OpenCL is materially faster the runtime choice moves,
# and the C++ seam should not be written until this is known.
#
# ⚠ Build MNNConvert with the HOST compiler and libMNN with the LINUX ndk.  A previous
# attempt configured build_android_64 with the WINDOWS NDK path from inside WSL and
# produced two build directories containing nothing but a CMakeCache.
#
#   ~/MNN/build_host   cmake -DMNN_BUILD_CONVERTER=ON  && ninja MNNConvert
#   ~/MNN/build_a64    cmake -DCMAKE_TOOLCHAIN_FILE=$HOME/ndk/.../android.toolchain.cmake \
#                            -DANDROID_ABI=arm64-v8a -DMNN_OPENCL=ON -DMNN_VULKAN=ON \
#                            -DMNN_ARM82=ON -DMNN_BUILD_BENCHMARK=ON -DMNN_SEP_BUILD=OFF
set -uo pipefail

CONV="$HOME/MNN/build_host/MNNConvert"
WIN="/mnt/c/Users/abrah/Desktop/CC/facefusion-mobile/work/onnx"
DST="$HOME/ffmnn"
mkdir -p "$DST"
cd "$DST"

[ -x "$CONV" ] || { echo "no MNNConvert at $CONV -- build it first"; exit 1; }

# fp16 storage, to match what the ncnn set was measured with. Comparing an fp16 ncnn model
# against an fp32 MNN one would measure the precision, not the runtime.
convert () {
    local name="$1"
    if [ -f "$name.mnn" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "== $name already converted (FORCE=1 to redo) =="; return 0
    fi
    if [ ! -f "$name.onnx" ] || [ "$WIN/$name.onnx" -nt "$name.onnx" ]; then
        cp "$WIN/$name.onnx" . || { echo "COPY FAILED $name"; return 1; }
    fi
    echo "== MNNConvert $name =="
    "$CONV" -f ONNX --modelFile "$name.onnx" --MNNModel "$name.mnn" \
            --bizCode ff --fp16 > "$name.mnnconv.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$name.mnn" ]; then
        echo "  FAILED (rc=$rc):"; tail -12 "$name.mnnconv.log" | sed 's/^/    /'; return 1
    fi
    ls -la "$name.mnn" | sed 's/^/  /'
    # MNNConvert reports ops it had to fall back on; catching that here is cheaper than
    # finding it as a mystery in the OpenCL timing.
    grep -iE "not support|unsupported|fallback" "$name.mnnconv.log" | head -5 | sed 's/^/  ⚠ /'
}

WANTED="$*"
want () { [ -z "$WANTED" ] && return 0; case " $WANTED " in *" $1 "*) return 0;; *) return 1;; esac; }

want yoloface_8n_b1_sim        && convert yoloface_8n_b1_sim
want 2dfan4_heatmaps_sim       && convert 2dfan4_heatmaps_sim
want arcface_w600k_r50_b1_sim  && convert arcface_w600k_r50_b1_sim
want hyperswap_1a_256_fp32     && convert hyperswap_1a_256_fp32
want nsfw_2_sim                && convert nsfw_2_sim
want gpen_bfr_256_sim          && convert gpen_bfr_256_sim

echo "=== done ==="
ls -la "$DST"/*.mnn 2>/dev/null
