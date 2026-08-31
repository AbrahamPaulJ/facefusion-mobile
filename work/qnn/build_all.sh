#!/usr/bin/env bash
# Build every MVP graph W8A16 with the corrected quantiser options (per-channel only --
# see the QOPT block in convert.sh for why per-row is disabled).
#
#   ./build_all.sh              # all four
#   ./build_all.sh fan2d yoloface
set -uo pipefail
FF=${FF:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
MODELS=${*:-"arcface fan2d yoloface hyperswap"}

for m in $MODELS; do
  echo "================================ $m ================================"
  date '+%H:%M:%S'
  bash "$FF/work/qnn/convert.sh" "$m" 2>&1 \
    | grep -E "mode:|converter exit|Total MACs|RuntimeError|ERROR|^-rw" | head -8
done
echo
echo "================================ done ================================"
ls -la "$FF"/work/device/*_v79.bin
