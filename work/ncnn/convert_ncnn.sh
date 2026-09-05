#!/bin/bash
# ONNX -> ncnn, for roadmap 6: the non-Qualcomm backend.
#
# ⚠ This script lived only in WSL $HOME until 2026-08-30, which meant the recipe that
# produced every artefact in ~/ffncnn was not in the repo and not in any commit. The same
# gap hid `simplify_onnx.py` for months. Conversion recipes belong here.
#
# FORCE=1 reconverts even when the .ncnn.param already exists -- needed whenever the ONNX
# has changed underneath, which it has for gpen (the blur-block surgery).
#
# Shapes come from the ONNX graph inputs, in graph order -- hyperswap takes source[1,512]
# BEFORE target[1,3,256,256], and getting that backwards produces a model that converts
# cleanly and benchmarks nonsense.
#
# Copied to the WSL filesystem first: hyperswap_fp32 is 805 MB and the 9p mount makes that
# the slowest part of the whole job.
set -uo pipefail

PNNX="$HOME/pnnx/pnnx-20260704-linux/pnnx"
# The ONNX directory, derived from this script's own location rather than hardcoded:
# work/<backend>/convert_*.sh -> work/onnx. A literal /mnt/c/Users/<name>/... path both
# leaks a username into a public repo and only ever worked on one machine.
WIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../onnx" && pwd)"
DST="$HOME/ffncnn"
mkdir -p "$DST"
cd "$DST"

convert () {
    local name="$1"; shift
    local shapes="$1"; shift
    if [ -f "$name.ncnn.param" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "== $name already converted (FORCE=1 to redo) =="; return 0
    fi
    # A stale .onnx is worse than a missing one: it converts cleanly and benchmarks the
    # WRONG GRAPH. Re-copy whenever the source is newer than what is here.
    if [ -f "$name.onnx" ] && [ "$WIN/$name.onnx" -nt "$name.onnx" ]; then
        echo "== $name.onnx is stale, re-copying =="; rm -f "$name.onnx"
    fi
    if [ ! -f "$name.onnx" ]; then
        echo "== copying $name.onnx from the 9p mount =="
        cp "$WIN/$name.onnx" . || { echo "COPY FAILED $name"; return 1; }
    fi
    echo "== pnnx $name  inputshape=$shapes =="
    # fp16=1 is pnnx's default and is what we want stored: it is the GPU question.
    /usr/bin/time -f "  wall %E  maxrss %MkB" \
        "$PNNX" "$name.onnx" "inputshape=$shapes" > "$name.pnnx.log" 2>&1
    local rc=$?
    grep -E "^FLOPS|^memory OPS|^model inputshape" "$name.pnnx.log" | sed 's/^/  /'
    tail -2 "$name.pnnx.log" | grep -E "wall|maxrss" | sed 's/^/  /'
    if [ $rc -ne 0 ] || [ ! -f "$name.ncnn.param" ]; then
        echo "  FAILED (rc=$rc) -- last lines:"; tail -15 "$name.pnnx.log" | sed 's/^/    /'
        return 1
    fi
    ls -la "$name.ncnn.param" "$name.ncnn.bin" | sed 's/^/  /'
    # Any layer ncnn could not map shows up as a bare op name here; catching it now is
    # cheaper than discovering it as a silent CPU fallback in the Vulkan timing.
    echo "  layer types: $(awk 'NR>2{print $1}' "$name.ncnn.param" | sort -u | tr '\n' ' ')"
}

# Convert everything, or just the graphs named on the command line:
#   convert_ncnn.sh                     # all of them
#   FORCE=1 convert_ncnn.sh gpen_bfr_256_sim
# Reconverting hyperswap costs an 805 MB copy across the 9p mount, so a subset matters.
WANTED="$*"
want () { [ -z "$WANTED" ] && return 0; case " $WANTED " in *" $1 "*) return 0;; *) return 1;; esac; }

want 2dfan4_heatmaps       && convert 2dfan4_heatmaps        "[1,3,256,256]"
want arcface_w600k_r50_b1  && convert arcface_w600k_r50_b1   "[1,3,112,112]"
# The enhancer goes through pnnx_prep.py first -- pnnx will not fold a transB=0 Gemm
# into InnerProduct, and GPEN's style mapping network is nine of them:
#   py -3.10 work/ncnn/pnnx_prep.py work/onnx/gpen_bfr_256_sim.onnx work/onnx/gpen_ncnn.onnx
want gpen_ncnn             && convert gpen_ncnn              "[1,3,256,256]"
want hyperswap_1a_256_fp32 && convert hyperswap_1a_256_fp32  "[1,512],[1,3,256,256]"
want yoloface_8n_b1        && convert yoloface_8n_b1         "[1,3,640,640]"
# The content gate. It is MANDATORY on both branch lines -- ffpipe treats a missing gate as
# an init failure, not a fallback -- so no non-Qualcomm build can ship until this converts.
want nsfw_2_sim            && convert nsfw_2_sim             "[1,3,384,384]"

echo "=== ALL DONE ==="
ls -la "$DST"/*.ncnn.param
