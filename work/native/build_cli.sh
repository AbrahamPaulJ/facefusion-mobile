#!/usr/bin/env bash
# Cross-compile the on-device CLI (aarch64-android) with the NDK.
#
# The CLI exists so the whole pipeline -- QNN + geometry + video -- can be tested headlessly
# over adb with the phone's screen locked, before any APK packaging is involved.  It lives
# in /data/local/tmp, where a binary IS allowed to reach the DSP (trap #33 only bites
# executables launched out of an APK).
set -uo pipefail
FF=${FF:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# The Windows SDK NDK only ships windows-x86_64 prebuilts, and this cross-compile runs
# under WSL, so it needs the Linux NDK Neodragon installed for the same reason.
NDK=${NDK:-$HOME/ndk/android-ndk-r26d}
TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64
CXX=$TC/bin/aarch64-linux-android31-clang++
OUT=$HOME/ff-native/android
mkdir -p "$OUT"

[ -x "$CXX" ] || { echo "no NDK clang++ at $CXX"; ls "$TC/bin" 2>/dev/null | head; exit 1; }

CPP=$FF/work/android/app/src/main/cpp
"$CXX" -O2 -std=c++17 -fPIC \
  -I"$CPP/include" -I"$CPP/include/QNN" \
  "$FF/work/native/ffswap_main.cpp" "$CPP/ffqnn.cpp" "$CPP/ffcv.cpp" "$CPP/ffpipe.cpp" \
  "$CPP/ffnn.cpp" "$CPP/ffnn_qnn.cpp" \
  -o "$OUT/ffswap" -llog -ldl -lm -static-libstdc++ 2>&1 | head -40

[ -f "$OUT/ffswap" ] || { echo "BUILD FAILED"; exit 1; }
ls -la "$OUT/ffswap"
cp "$OUT/ffswap" "$FF/work/device/ffswap"
echo "-> work/device/ffswap"
