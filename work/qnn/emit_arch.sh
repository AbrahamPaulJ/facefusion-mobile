#!/usr/bin/env bash
# Emit an extra HTP context binary from an ALREADY-CONVERTED graph.
#
# Steps 1/3 (quantise) and 2/3 (lib-generator) are arch-independent: only
# qnn-context-binary-generator reads dsp_arch and vtcm_mb, and it consumes the host
# lib<NAME>.so that step 2 produced.  So a second target costs seconds, not another
# conversion.  Measured 2026-08-24: yoloface 2-3 s, hyperswap 12-16 s.
#
#   ./emit_arch.sh v68 2                    # every converted graph, v68 / 2 MB VTCM
#   ./emit_arch.sh v73 8 arcface hyperswap  # just these
#   ./emit_arch.sh all                      # every graph, every shipping tier
#
# WARNING: depends on $HOME/ff-build/<name>/lib/x86_64-linux-clang/lib<name>.so.  A rerun of
# convert.sh for that graph rebuilds it; deleting ~/ff-build means a full reconversion.
#
# WARNING: this cannot cross SDK versions.  A 2.49 lib emits 2.49 contexts only -- 2.28's
# generator rejects a 2.49 lib at the first Conv (docs/roadmap.md 1.7).  To emit 2.28
# contexts, first build the graph with QSDK=228 and point BUILD_TAG at it.
set -uo pipefail

QSDK=${QSDK:-249}
case "$QSDK" in
  249) source "$HOME/npuconvert/qnn_env.sh";     BUILD_TAG="" ;;
  228) source "$HOME/npuconvert/qnn_env_228.sh"; BUILD_TAG="_228" ;;
  *) echo "unknown QSDK: $QSDK (want 249 or 228)"; exit 1 ;;
esac

FF=${FF:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# The shipping tiers, mirroring LocalDream's _min / _8gen1 / _8gen2 split plus our v79.
# soc_model is emitted only for v79: it pins a context to one SoC, and below v79 the whole
# point of the build is breadth.
#   v68/2  -- SD 888 and up.  Runs forward onto every later HTP; 2 MB fits anywhere.
#   v69/8  -- 8 Gen 1 and up with full VTCM.
#   v73/8  -- 8 Gen 2 and up.
#   v79/8  -- 8 Elite.  Pinned to soc_model 69, so it is NOT the answer for newer parts.
#   v81/8  -- 8 Elite Gen 5 and up.  Added because pickTier was sending every v81 part
#             (S26 Ultra) to the v73 build: the v79 context is soc-pinned, so `arch >= 79`
#             fell through to `arch >= 73`.  Deliberately NOT soc-pinned -- below v79 the
#             point of a tier is breadth, and the same applies above it.
#             ⚠ vtcm_mb 8 is the CONSERVATIVE value carried over from v73/v79, not a
#             measurement: no v81 part has been on the bench.  If one reports more VTCM,
#             raising this is free performance and needs re-emitting nothing else.
TIERS="v68:2 v69:8 v73:8 v79:8 v81:8"

emit() {
  local name=$1 arch=$2 vtcm=$3
  local src=$HOME/ff-build/${name}${BUILD_TAG}
  # $name is the BUILD DIRECTORY, which is not always the graph name: a variant build like
  # `nsfwq` or `hyperswap_fp16src` carries the suffix in its directory while the graph
  # inside it -- the lib, and the `graph_names` the context declares -- keeps the bare name.
  # Deriving the graph from the lib that is actually there fixes both at once.  Hard-coding
  # lib${name}.so silently SKIPPED every variant, and hard-coding graph_names[$name] would
  # have emitted a context declaring a graph nothing could look up.
  local lib
  lib=$(ls "$src"/lib/x86_64-linux-clang/lib*.so 2>/dev/null | head -1)
  if [ -z "$lib" ]; then
    printf '  %-11s %-4s vtcm%-2s  SKIP (no lib in %s)\n' \
        "$name" "$arch" "$vtcm" "${src#$HOME/}/lib"
    return
  fi
  local graph
  graph=$(basename "$lib" .so); graph=${graph#lib}

  local cfg=$FF/work/qnn/htp_config_${name}_${arch}.json
  local be=$FF/work/qnn/htp_backend_${name}_${arch}.json
  local soc=""
  [ "$arch" = "v79" ] && soc='"soc_model": 69,'
  cat > "$cfg" <<EOF
{ "graphs": [ { "vtcm_mb": $vtcm, "graph_names": ["$graph"], "O": 3.0 } ],
  "devices": [ { "dsp_arch": "$arch", $soc
                 "cores": [ { "perf_profile": "burst", "rpc_control_latency": 100 } ] } ] }
EOF
  cat > "$be" <<EOF
{ "backend_extensions": { "shared_library_path": "libQnnHtpNetRunExtensions.so",
                          "config_file_path": "$cfg" } }
EOF

  local out=$src/ctx
  local bin=${name}${BUILD_TAG}_${arch}
  local t0=$(date +%s)
  qnn-context-binary-generator --model "$lib" \
      --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
      --output_dir "$out" --binary_file "$bin" --config_file "$be" \
      > "$out/${bin}.log" 2>&1
  local rc=$? secs=$(( $(date +%s) - $t0 ))

  if [ $rc -eq 0 ] && [ -f "$out/${bin}.bin" ]; then
    cp "$out/${bin}.bin" "$FF/work/device/"
    local mb=$(( $(stat -c%s "$out/${bin}.bin") / 1048576 ))
    # spillFill is the number that matters when VTCM shrinks -- report it, do not assume it
    local spill=$(python3 - "$out/${bin}.bin" 2>/dev/null <<'PY'
import subprocess, sys, json, os, tempfile
util = os.environ["QNN_SDK_ROOT"] + "/bin/x86_64-linux-clang/qnn-context-binary-utility"
j = tempfile.mktemp(suffix=".json")
if subprocess.run([util, "--context_binary", sys.argv[1], "--json_file", j],
                  capture_output=True).returncode == 0 and os.path.exists(j):
    g = json.load(open(j))["info"]["graphs"][0]["info"]
    gb = g.get("graphBlobInfo")
    inf = gb.get("info") if isinstance(gb, dict) else (gb[0].get("info") if gb else {})
    print("%.1f MB" % ((inf or {}).get("spillFillBufferSize", 0) / 1048576.0))
PY
)
    printf '  %-11s %-4s vtcm%-2s  %4s MB  spill %-9s %3ds\n' \
        "$name" "$arch" "$vtcm" "$mb" "${spill:-?}" "$secs"
  else
    printf '  %-11s %-4s vtcm%-2s  FAILED (%ds) -- see %s\n' \
        "$name" "$arch" "$vtcm" "$secs" "$out/${bin}.log"
  fi
}

if [ "${1:-}" = "all" ]; then
  GRAPHS="yoloface fan2d arcface hyperswap nsfw"
  echo "=== QAIRT $QSDK: every graph, every tier ==="
  for t in $TIERS; do
    echo "-- dsp_arch ${t%%:*}, vtcm ${t##*:} MB"
    for g in $GRAPHS; do emit "$g" "${t%%:*}" "${t##*:}"; done
  done
else
  ARCH=${1:?usage: emit_arch.sh <arch> <vtcm_mb> [graphs...]   |   emit_arch.sh all}
  VTCM=${2:?usage: emit_arch.sh <arch> <vtcm_mb> [graphs...]}
  shift 2
  GRAPHS=${*:-"yoloface fan2d arcface hyperswap nsfw"}
  echo "=== QAIRT $QSDK: dsp_arch $ARCH, vtcm ${VTCM} MB ==="
  for g in $GRAPHS; do emit "$g" "$ARCH" "$VTCM"; done
fi
