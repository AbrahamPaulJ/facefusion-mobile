#!/usr/bin/env bash
# Build the geometry library for the HOST and check every op against OpenCV.
# Run this before trusting any change to ffcv.cpp -- the app and the on-device CLI both
# depend on it reproducing FaceFusion's geometry, not merely approximating it.
set -uo pipefail
source "$HOME/npuconvert/qnn_env.sh"
HERE=${HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
OUT=$HOME/ff-native
mkdir -p "$OUT"

g++ -O2 -fPIC -shared -std=c++17 \
    "$HERE/ffcv_capi.cpp" \
    "$HERE/../android/app/src/main/cpp/ffcv.cpp" \
    -o "$OUT/libffcv.so" 2>&1 | head -30
[ -f "$OUT/libffcv.so" ] || { echo "BUILD FAILED"; exit 1; }
cp "$OUT/libffcv.so" "$HERE/libffcv.so"
echo "built $(stat -c%s "$OUT/libffcv.so") bytes"

cd "$HERE" && python test_ffcv.py
geometry=$?

# The lip syncer's audio front end. Separate .so because it shares no code with ffcv and
# a failure here should name the audio port rather than the geometry one.
g++ -O2 -fPIC -shared -std=c++17 \
    "$HERE/ffaudio_capi.cpp" \
    "$HERE/../android/app/src/main/cpp/ffaudio.cpp" \
    -o "$OUT/libffaudio.so" 2>&1 | head -30
[ -f "$OUT/libffaudio.so" ] || { echo "ffaudio BUILD FAILED"; exit 1; }
cp "$OUT/libffaudio.so" "$HERE/libffaudio.so"
echo "built $(stat -c%s "$OUT/libffaudio.so") bytes (ffaudio)"

cd "$HERE" && python test_ffaudio.py
audio=$?

exit $(( geometry || audio ))
