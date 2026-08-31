#!/usr/bin/env bash
# Bisect why quantised arcface returns garbage on HTP while the fp32 build is 56.65 dB.
#
# Known so far (all measured, 16 held-out crops, cosine vs the host embedding):
#   fp32   56.65 dB   cosine 0.999999
#   W8A16  -7.45 dB   cosine -0.0155
#   W16A16 -7.46 dB   cosine -0.0158     -> weight bitwidth is NOT the cause
#   input fed as float32 / native u16 / NHWC: all identical -> input handling is correct
#
# So the remaining suspects are the quantiser OPTIONS and the activation bitwidth.
set -uo pipefail
FF=${FF:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

run() {
  echo "===== $* ====="
  env "$@" bash "$FF/work/qnn/convert.sh" arcface 2>&1 \
    | grep -E "mode:|converter exit|ERROR" | head -4
}

run QOPT=min
run QOPT=pc
run QOPT=min ABW=32

echo
echo "built context binaries:"
ls -la "$HOME"/ff-build/arcface*/ctx/*.bin 2>/dev/null
