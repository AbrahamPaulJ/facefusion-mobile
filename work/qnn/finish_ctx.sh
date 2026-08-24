#!/usr/bin/env bash
# Run just lib-generator + context-binary-generator for an already-converted graph.
#
# Exists because the converter is the long step (hyperswap: ~5 min of CPU-backend
# calibration passes) and there is no reason to repeat it when only the tail failed --
# which happened twice, once from editing convert.sh mid-run (docs/traps.md #5).
#
# ARCH picks the target; the htp_*.json it reads are the arch-suffixed ones convert.sh and
# emit_arch.sh write.  If the lib already exists, emit_arch.sh is the better tool -- this
# one exists for the case where step 2 still has to run.
set -uo pipefail
source "$HOME/npuconvert/qnn_env.sh"
FF=${FF:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
NAME=${1:?usage: finish_ctx.sh <name> [suffix]}
SUF=${2:-}
ARCH=${ARCH:-v79}
OUT=$HOME/ff-build/${NAME}${SUF}
LIB=$OUT/lib/x86_64-linux-clang/lib${NAME}.so

cd "$OUT" || exit 1
if [ ! -f "$LIB" ]; then
  echo "== lib-generator =="
  qnn-model-lib-generator -c "$OUT/$NAME.cpp" -b "$OUT/$NAME.bin" \
      -t x86_64-linux-clang -o "$OUT/lib" 2>&1 | tail -4
fi
[ -f "$LIB" ] || { echo "ERROR: no $LIB"; exit 1; }
ls -la "$LIB"

echo "== context-binary-generator =="
date '+%H:%M:%S'
qnn-context-binary-generator --model "$LIB" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$OUT/ctx" --binary_file "${NAME}${SUF}_${ARCH}" \
    --config_file "$FF/work/qnn/htp_backend_${NAME}_${ARCH}.json" 2>&1 \
  | grep -viE '^\s*\[[# ]*\]' | tail -12
date '+%H:%M:%S'

if [ -f "$OUT/ctx/${NAME}${SUF}_${ARCH}.bin" ]; then
  cp "$OUT/ctx/${NAME}${SUF}_${ARCH}.bin" "$FF/work/device/"
  ls -la "$FF/work/device/${NAME}${SUF}_${ARCH}.bin"
else
  echo "ERROR: context binary was not produced"
  exit 1
fi
