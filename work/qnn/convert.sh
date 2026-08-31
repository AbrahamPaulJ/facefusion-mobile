#!/usr/bin/env bash
# W8A16 conversion for the FaceFusion MVP graphs.  Cloned from
# ../Neodragon/work/qnn/convert_quicksrnet.sh, which is the recipe that works.
#
#   ./convert.sh <name> --layout   # layout screen only: no calibration, fast, and a
#                                  # faithful proxy for the quantised build (trap #24)
#   ./convert.sh <name>            # full W8A16 build -> context binary
#
# names: arcface | fan2d | yoloface | hyperswap | inswapper | nsfw | gpen | wav2lip
#
# Every graph here is a conv model, so --preserve_io layout on the image tensors is
# MANDATORY (trap #7: omitting it measured -0.75 dB with nothing looking broken).
# hyperswap's `source` is a rank-2 embedding with no image semantics and is deliberately
# left out of preserve_io (trap #23).
#
# Builds under $HOME (ext4).  qnn-model-lib-generator explodes the weight blob into ~900
# .raw files in a temp dir under the CWD; doing that on /mnt/c crosses the 9p mount for
# every one of them (trap #30).
set -uo pipefail

# QSDK selects the toolchain.  249 (default) is the shipping build; 228 exists because a
# 2.49-built context declares an fp16 requirement that chips lacking fp16 execution refuse
# at load, and rebuilding under 2.28 is the only known way to drop it (docs/roadmap.md 1.3).
#
# WARNING: the two are NOT interchangeable at the context step.  2.28's
# context-binary-generator rejects a 2.49 libmodel.so at the first Conv with
# MODEL_GRAPH_OP_VALIDATION_ERROR, so a 2.28 build re-quantises from ONNX -- it is not a
# re-emit.  Verified by LocalDream 2026-08-14 (npuconvertv2/build_vae_228.sh header).
# Separate output dirs keep the 2.49 intermediates intact, since they are the only record
# of the shipping build.
QSDK=${QSDK:-249}
case "$QSDK" in
  249) source "$HOME/npuconvert/qnn_env.sh";     BUILD_TAG="" ;;
  228) source "$HOME/npuconvert/qnn_env_228.sh"; BUILD_TAG="_228" ;;
  *) echo "unknown QSDK: $QSDK (want 249 or 228)"; exit 1 ;;
esac

# ARCH/VTCM target the context binary only -- step 3/3 is the sole consumer of dsp_arch and
# vtcm_mb, and it eats the arch-independent lib<NAME>.so from step 2.  So a second arch is a
# re-emit of step 3, not another conversion (measured: 2-3 s yoloface, 12-16 s hyperswap).
ARCH=${ARCH:-v79}
VTCM=${VTCM:-8}

FF=${FF:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
NAME=${1:?usage: convert.sh <name> [--layout]}
MODE=${2:-full}

case "$NAME" in
  arcface)
    ONNX=$FF/work/onnx/arcface_w600k_r50_b1_sim.onnx
    DIMS=(--input_dim input 1,3,112,112)
    PRESERVE=(--preserve_io layout input)
    ;;
  fan2d)
    ONNX=$FF/work/onnx/2dfan4_heatmaps_sim.onnx
    DIMS=(--input_dim input 1,3,256,256)
    PRESERVE=(--preserve_io layout input heatmaps)
    ;;
  yoloface)
    ONNX=$FF/work/onnx/yoloface_8n_b1_sim.onnx
    DIMS=(--input_dim input 1,3,640,640)
    PRESERVE=(--preserve_io layout input)
    ;;
  hyperswap)
    # The SHIPPING graph is the fp32-demoted one.  hyperswap is the only natively-fp16
    # model we ship (306 fp16 initialisers); demoting the weights to fp32 before
    # quantisation measured 3.5% FASTER at bit-identical accuracy (30.87 dB deploy SNR,
    # 47.87/41.68 dB e2e, both unchanged), and it is also the only way 2.28 will convert
    # this graph at all.  docs/roadmap.md 1.7.  The fp16-sourced intermediates are
    # archived at ~/ff-build/hyperswap_fp16src.
    ONNX=$FF/work/onnx/hyperswap_1a_256_fp32.onnx
    DIMS=(--input_dim target 1,3,256,256 --input_dim source 1,512)
    PRESERVE=(--preserve_io layout target output)
    ;;
  nsfw)
    # The content gate.  A ViT-Small: the whole surgery is constant folding, which also
    # removes the Expand and every Slice (prepare_onnx.py:do_nsfw).
    ONNX=$FF/work/onnx/nsfw_2_sim.onnx
    DIMS=(--input_dim input 1,3,384,384)
    PRESERVE=(--preserve_io layout input)
    ;;
  gpen)
    # The face enhancer, after prepare_onnx.py:do_gpen made every modulated conv kernel
    # static -- 20 of its 45 convs took a COMPUTED kernel as exported, which HTP cannot map,
    # and that was 87% of the graph's 8.57 GMAC. The surgery is exact to 107.9 dB.
    ONNX=$FF/work/onnx/gpen_bfr_256_sim.onnx
    DIMS=(--input_dim input 1,3,256,256)
    PRESERVE=(--preserve_io layout input output)
    ;;
  edtalk)
    # The 256x256 lip syncer. Screened and REJECTED once on a raw-graph op census, which
    # was wrong: onnxsim with the shapes pinned folds 4772 nodes to 1170 and both `If`
    # nodes away. This case exists to settle the rest the way trap #2 says to -- by
    # building it float and reading the converter's own complaint.
    ONNX=$FF/work/onnx/edtalk_256_sim.onnx
    DIMS=(--input_dim source 1,1,80,16 --input_dim target 1,3,256,256 --input_dim weight 1)
    PRESERVE=(--preserve_io layout target output)
    ;;
  wav2lip)
    # The lip syncer (upstream's default is the GAN variant; the two are the same graph
    # to the node).  Two inputs: `source` is the 80x16 mel window, `target` is the masked
    # crop concatenated with the reference crop on the channel axis.
    #
    # `source` is rank-4 with C=1, so NCHW and NHWC are the SAME bytes for it -- it is in
    # preserve_io only so that every tensor the app fills is NCHW, with no special case.
    ONNX=$FF/work/onnx/wav2lip_gan_96_b1_sim.onnx
    DIMS=(--input_dim source 1,1,80,16 --input_dim target 1,6,96,96)
    PRESERVE=(--preserve_io layout source target output)
    ;;
  inswapper)
    ONNX=$FF/work/onnx/inswapper_128_split_sim.onnx
    DIMS=(--input_dim target 1,3,128,128 --input_dim source 1,512)
    PRESERVE=(--preserve_io layout target output)
    ;;
  *) echo "unknown model: $NAME"; exit 1 ;;
esac

# float and W8A16 builds must not share an output dir: the float run was overwriting
# ${NAME}_net.json, so an encoding scan silently read bitwidth-0 tensors.
#
# nsfw INVERTS that: fp32 is its SHIPPING build and W8A16 is the experiment, so it is the
# float build that gets the bare name and the quantised one that is suffixed.  Measured
# 2026-08-24: W8A16 shifts the gate's decision statistic +0.075 mean / +0.153 max, 16 of 16
# held-out frames toward flagging, against a 0.25 threshold, while the fp32 context tracks
# the host at -0.012.  The gate is SAMPLED (~11 calls per video, not per frame), so
# quantising it buys 7.3 MB and 10 ms and costs false refusals (docs/roadmap.md 2).
NAME_SUFFIX=""
if [ "$NAME" = "nsfw" ]; then
  # "q2", not "q": the quantised gate's encodings are calibrated for the INPUT RANGE the
  # app feeds it, and that range changed on 2026-08-30 from [0,1] to [-1,1] (the range
  # facefusion actually uses). An app on the new range with a `nsfwq_` built for the old
  # one is silently wrong -- the input lands outside the calibrated interval and clamps.
  # A new FILENAME makes that incompatibility impossible to hit by accident: an app that
  # wants `nsfwq2_` simply does not find `nsfwq_`, reports the gate missing, and offers
  # the download. Renaming is the cheapest way to turn a silent mismatch into a prompt.
  [ "$MODE" != "--float" ] && [ "$MODE" != "--layout" ] && NAME_SUFFIX="q2"
else
  [ "$MODE" = "--float" ] && NAME_SUFFIX="f"
fi
[ "${WBW:-8}" != "8" ] && NAME_SUFFIX="${NAME_SUFFIX}w${WBW}"
[ "${ABW:-16}" != "16" ] && NAME_SUFFIX="${NAME_SUFFIX}a${ABW}"
[ "${QOPT:-pc}" != "pc" ] && NAME_SUFFIX="${NAME_SUFFIX}${QOPT}"
OUT=$HOME/ff-build/${NAME}${NAME_SUFFIX}${BUILD_TAG}
mkdir -p "$OUT"
[ -f "$ONNX" ] || { echo "ERROR: no $ONNX -- run work/export/prepare_onnx.py first"; exit 1; }

QFLAGS=()
n=0
if [ "$MODE" = "--layout" ] || [ "$MODE" = "--float" ]; then
  echo "mode: ${MODE#--} (fp32, no calibration)"
else
  LIST=$FF/work/calib/${NAME}_list.txt
  [ -f "$LIST" ] || { echo "ERROR: no $LIST -- run work/qnn/make_calib_lists.py"; exit 1; }
  cp "$LIST" "$OUT/calib_list.txt"
  n=$(wc -l < "$OUT/calib_list.txt")
  missing=$(sed 's/[a-z_]*:=//g' "$OUT/calib_list.txt" | tr ' ' '\n' | while read -r f; do
              [ -z "$f" ] || [ -f "$f" ] || echo x; done | wc -l)
  [ "$missing" -eq 0 ] || { echo "ERROR: $missing calibration files missing"; exit 1; }
  QFLAGS=(--input_list "$OUT/calib_list.txt"
          --act_bitwidth ${ABW:-16} --weights_bitwidth ${WBW:-8} --bias_bitwidth 32)
  # QOPT selects the extra quantiser options, so they can be bisected:
  #   pc (default) = per-channel only ;  full = per-channel + per-row ;  min = neither
  #
  # *** --use_per_row_quantization DESTROYS these graphs. ***  Measured on arcface, 16
  # held-out crops, against the fp32 host embedding:
  #     fp32                       56.65 dB   cosine 0.999999
  #     per-channel + per-row      -7.45 dB   cosine -0.0155     <- garbage
  #     no options                 21.37 dB   cosine 0.9964
  #     per-channel only           32.23 dB   cosine 0.99970     <- the default now
  # Neodragon used per-row happily, so this is model-specific, not an SDK-wide rule --
  # suspected to be the [512, 25088] FullyConnected.  Bisect with QOPT before assuming
  # a quantisation problem is calibration.
  case "${QOPT:-pc}" in
    full) QFLAGS+=(--use_per_channel_quantization
                   --use_per_row_quantization --enable_per_row_quantized_bias) ;;
    pc)   QFLAGS+=(--use_per_channel_quantization) ;;
    min)  : ;;
  esac
  # CLE rescales per-channel weights assuming positively-homogeneous activations.
  # Set CLE=1 to enable; it is OFF by default because it broke arcface (25x PRelu with
  # per-channel slopes): device output saturated at the encoding bounds with 2.1x the
  # reference std and cosine ~0.01 against the host embedding.
  [ "${CLE:-0}" = "1" ] && QFLAGS+=(--algorithms cle)
  echo "mode: FULL W8A16, $n calibration samples"
fi

CFG=$FF/work/qnn/htp_config_${NAME}_${ARCH}.json
BE=$FF/work/qnn/htp_backend_${NAME}_${ARCH}.json
# soc_model is emitted only for v79.  LocalDream's lower tiers omit it deliberately: it
# pins the context to one SoC, and below v79 the point of the build is breadth.
SOC_LINE=""
[ "$ARCH" = "v79" ] && SOC_LINE='"soc_model": 69,'
cat > "$CFG" <<EOF
{ "graphs": [ { "vtcm_mb": $VTCM, "graph_names": ["$NAME"], "O": 3.0 } ],
  "devices": [ { "dsp_arch": "$ARCH", $SOC_LINE
                 "cores": [ { "perf_profile": "burst", "rpc_control_latency": 100 } ] } ] }
EOF
cat > "$BE" <<EOF
{ "backend_extensions": { "shared_library_path": "libQnnHtpNetRunExtensions.so",
                          "config_file_path": "$CFG" } }
EOF

step() { echo; echo "########## $* ##########"; date '+%H:%M:%S'; }

echo "toolchain: QAIRT $QSDK  ->  $QNN_SDK_ROOT"
echo "target   : dsp_arch $ARCH, vtcm ${VTCM}MB"
step "1/3 converter ($NAME, mode=$MODE, $n samples)"
cd "$OUT" || exit 1
qnn-onnx-converter --input_network "$ONNX" \
    --output_path "$OUT/$NAME.cpp" \
    "${DIMS[@]}" "${PRESERVE[@]}" "${QFLAGS[@]}" \
    2>&1 | tee "$OUT/converter.log" | grep -viE '^\s*\[[# ]*\]' | tail -14
rc=${PIPESTATUS[0]}
echo "converter exit $rc"
[ "$rc" -eq 0 ] || exit "$rc"

cp "$OUT/${NAME}_net.json" "$FF/work/device/${NAME}${NAME_SUFFIX}${BUILD_TAG}_net.json" 2>/dev/null \
  && echo "-> work/device/${NAME}${NAME_SUFFIX}${BUILD_TAG}_net.json"

if [ "$MODE" = "--layout" ]; then
  echo; echo "LAYOUT screen done -- now run:"
  echo "  py -3.10 work/device/analyze_net.py work/device/${NAME}${NAME_SUFFIX}_net.json"
  exit 0
fi

step "2/3 lib-generator"
qnn-model-lib-generator -c "$OUT/$NAME.cpp" -b "$OUT/$NAME.bin" \
    -t x86_64-linux-clang -o "$OUT/lib" 2>&1 | tail -3

step "3/3 context-binary-generator"
qnn-context-binary-generator --model "$OUT/lib/x86_64-linux-clang/lib$NAME.so" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$OUT/ctx" --binary_file "${NAME}${NAME_SUFFIX}${BUILD_TAG}_${ARCH}" \
    --config_file "$BE" 2>&1 | grep -viE '^\s*\[[# ]*\]' | tail -8

cp "$OUT/ctx/${NAME}${NAME_SUFFIX}${BUILD_TAG}_${ARCH}.bin" "$FF/work/device/" \
  && ls -la "$FF/work/device/${NAME}${NAME_SUFFIX}${BUILD_TAG}_${ARCH}.bin"
