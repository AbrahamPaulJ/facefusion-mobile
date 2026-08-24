#!/system/bin/sh
# On-device runner.  Push to /data/local/tmp/ff and run:
#     sh /data/local/tmp/ff/run_model.sh <name> [acc|perf]
#
# acc  -- one pass over every held-out input, outputs written for host comparison
# perf -- repeated runs with detailed profiling, for latency
#
# trap #8: these are QUANTISED graphs with UFIXED_POINT_16 I/O, so --use_native_input_files
# is deliberately NOT passed.  Passing it makes every input parse identically and every
# output plausible-but-constant.
set -u
FF=/data/local/tmp/ff
NAME=${1:?usage: run_model.sh <name> [acc|perf]}
MODE=${2:-acc}
CTX=${3:-$NAME}          # context binary basename; lets arcfacef reuse io/arcface

export LD_LIBRARY_PATH=$FF/lib:${LD_LIBRARY_PATH:-}
export ADSP_LIBRARY_PATH=$FF/dsp

cd "$FF" || exit 1
OUT=$FF/out/$CTX
rm -rf "$OUT"; mkdir -p "$OUT"

COMMON="--backend $FF/lib/libQnnHtp.so \
        --retrieve_context $FF/ctx/${CTX}_v79.bin \
        --input_list $FF/io/$NAME/input_list.txt \
        --output_dir $OUT"

case "$MODE" in
  perf)
    ./qnn-net-run $COMMON --profiling_level detailed --num_inferences 10 \
        --perf_profile burst 2>&1 | tail -25
    ;;
  *)
    ./qnn-net-run $COMMON --perf_profile burst 2>&1 | tail -15
    ;;
esac

echo "---- outputs ----"
ls "$OUT" | head -5
find "$OUT" -name '*.raw' | wc -l
