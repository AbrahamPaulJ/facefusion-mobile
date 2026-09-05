#!/usr/bin/env bash
# The NEON half of the ffcv gate: build test_warp_bounds for arm64 under AddressSanitizer
# and run it on the device.
#
# build_and_test.sh checks ffcv against OpenCV but compiles for x86_64, where FFCV_NEON is
# undefined -- so it cannot see a bug in a NEON path at all. This one compiles the real
# thing for the real ISA and lets ASAN watch the loads.
#
# Run from WSL:  work/native/build_asan_test.sh
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NDK=${NDK:-$HOME/ndk/android-ndk-r26d}
TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64
CXX=$TC/bin/aarch64-linux-android30-clang++
ASAN_RT=$(find "$TC" -name 'libclang_rt.asan-aarch64-android.so' | head -1)
# In the tree, not in ~: adb.exe is a Windows binary and cannot stat a WSL path, so the
# push has to name something both sides can see. Gitignored.
OUT=$HERE/asan-build
DEV=/data/local/tmp/ffasan
ADB=${ADB:-$HERE/../../../LocalDream/mvp/platform-tools/adb.exe}
SERIAL=${SERIAL:-}

[ -x "$CXX" ] || { echo "no NDK clang++ at $CXX"; exit 1; }
[ -n "$ASAN_RT" ] || { echo "no asan runtime in $TC"; exit 1; }

mkdir -p "$OUT"
# -static-libstdc++ so the binary needs nothing on the device but the ASAN runtime; the
# NDK links libc++_shared by default and /data/local/tmp has no library path.
"$CXX" -O2 -std=c++17 -fsanitize=address -fno-omit-frame-pointer -g -static-libstdc++ \
    "$HERE/test_warp_bounds.cpp" \
    "$HERE/../android/app/src/main/cpp/ffcv.cpp" \
    -o "$OUT/test_warp_bounds" || { echo "BUILD FAILED"; exit 1; }
echo "built $(stat -c%s "$OUT/test_warp_bounds") bytes for arm64+asan"

# adb lives on the Windows side and the device attaches twice, so -s is not optional
# (trap #14). MSYS_NO_PATHCONV keeps Git Bash from rewriting /data/local/tmp (trap #8).
if [ -z "$SERIAL" ]; then
  # adb.exe is a Windows binary, so every line it prints ends \r\n -- without the strip
  # `$2=="device"` compares against "device\r" and silently finds no device.
  SERIAL=$(MSYS_NO_PATHCONV=1 "$ADB" devices | tr -d '\r' | awk 'NR>1 && $2=="device" {print $1; exit}')
fi
[ -n "$SERIAL" ] || { echo "no device"; exit 1; }
echo "device $SERIAL"

adbs() { MSYS_NO_PATHCONV=1 "$ADB" -s "$SERIAL" "$@"; }

# adb.exe is a WINDOWS binary: it cannot stat /mnt/c/... or /home/... , so every LOCAL
# path handed to `adb push` has to be converted. (Remote paths are the opposite problem --
# trap #8 -- and are left alone.)
win() { wslpath -w "$1"; }

# The ASAN runtime lives under ~/ndk, which adb cannot see at all; stage it beside the
# binary first so both pushes name a Windows-visible path.
cp -f "$ASAN_RT" "$OUT/" || exit 1

adbs shell mkdir -p $DEV >/dev/null
adbs push "$(win "$OUT/test_warp_bounds")" $DEV/ >/dev/null || exit 1
adbs push "$(win "$OUT/$(basename "$ASAN_RT")")" $DEV/ >/dev/null || exit 1
adbs shell chmod 755 $DEV/test_warp_bounds

# Verify the push by hash, never by its own output (trap #8).
want=$(sha256sum "$OUT/test_warp_bounds" | cut -d' ' -f1)
got=$(adbs shell sha256sum $DEV/test_warp_bounds | cut -d' ' -f1 | tr -d '\r')
[ "$want" = "$got" ] || { echo "PUSH MISMATCH: $want != $got"; exit 1; }
echo "pushed, sha256 $got"

RUN="cd $DEV && LD_PRELOAD=./libclang_rt.asan-aarch64-android.so ASAN_OPTIONS=detect_leaks=0"

echo
echo "== asan =="
adbs shell "$RUN ./test_warp_bounds" 2>&1 | tr -d '\r'
rc=${PIPESTATUS[0]}

# Bit-exactness: NEON against FFCVSCALAR=1. The env var is read once into a function-local
# static, so the two configurations cannot share a process -- hence two runs and a diff.
echo
echo "== neon vs scalar =="
adbs shell "$RUN FFDUMP=1 ./test_warp_bounds > $DEV/neon.bin" >/dev/null 2>&1
adbs shell "$RUN FFDUMP=1 FFCVSCALAR=1 ./test_warp_bounds > $DEV/scalar.bin" >/dev/null 2>&1
n=$(adbs shell "sha256sum $DEV/neon.bin $DEV/scalar.bin; cmp -l $DEV/neon.bin $DEV/scalar.bin | wc -l" | tr -d '\r')
echo "$n"

exit $rc
