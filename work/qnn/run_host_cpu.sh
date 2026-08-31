#!/usr/bin/env bash
# Run a converted graph on the HOST CPU backend.
#
# This is the test that separates "the quantised graph is wrong" from "the device runs it
# wrong".  It needs no phone, and it is the only cheap way to know which half to debug.
#
#   ./run_host_cpu.sh <name> [buildsuffix]
set -uo pipefail
source "$HOME/npuconvert/qnn_env.sh"

FF=${FF:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
NAME=${1:?usage: run_host_cpu.sh <name> [buildsuffix]}
SUF=${2:-}
BUILD=$HOME/ff-build/${NAME}${SUF}
LIB=$BUILD/lib/x86_64-linux-clang/lib${NAME}.so

if [ ! -f "$LIB" ]; then
  echo "no $LIB -- building model lib"
  qnn-model-lib-generator -c "$BUILD/$NAME.cpp" -b "$BUILD/$NAME.bin" \
      -t x86_64-linux-clang -o "$BUILD/lib" 2>&1 | tail -3
fi
[ -f "$LIB" ] || { echo "ERROR: still no $LIB"; exit 1; }

LIST=/tmp/${NAME}${SUF}_host_list.txt
sed "s|/data/local/tmp/ff|$FF/work/device|g" "$FF/work/device/io/$NAME/input_list.txt" > "$LIST"
OUT=/tmp/${NAME}${SUF}_cpu
rm -rf "$OUT"; mkdir -p "$OUT"

qnn-net-run --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnCpu.so" \
    --model "$LIB" --input_list "$LIST" --output_dir "$OUT" 2>&1 | tail -8

DEST=$FF/work/device/out/${NAME}${SUF}_cpu
rm -rf "$DEST"; cp -r "$OUT" "$DEST"
echo "-> work/device/out/${NAME}${SUF}_cpu  ($(find "$DEST" -name '*.raw' | wc -l) tensors)"
