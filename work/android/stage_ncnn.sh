#!/usr/bin/env bash
# Stage the ncnn Android build for the APK's CMake, which runs under WINDOWS Gradle.
#
# ncnn is built in WSL with the Linux NDK (`~/ncnn/build-android-vulkan`, arm64-v8a,
# NCNN_VULKAN=ON NCNN_SIMPLEVK=ON) because that is where the converter and the bench live.
# Gradle, the Android SDK and the CMake that compiles `libffnative.so` all run on Windows.
# A Windows CMake cannot reliably reach `\\wsl$\...` -- ninja hands UNC paths to the
# compiler and the build fails in ways that read as missing headers -- so the archives and
# headers are COPIED to `work/android/ncnn/`, which is gitignored (~162 MB, derived).
#
# The APK build turns itself on when this directory exists; see build.gradle.kts. Delete
# the directory and the app builds QNN-only, exactly as it did before 0.4.0.
#
# Run from WSL:  bash work/android/stage_ncnn.sh
set -euo pipefail

NCNN_ROOT=${NCNN_ROOT:-$HOME/ncnn}
NCNN_BUILD=${NCNN_BUILD:-$NCNN_ROOT/build-android-vulkan}
DEST=${DEST:-$(cd "$(dirname "$0")" && pwd)/ncnn}

[ -f "$NCNN_BUILD/src/libncnn.a" ] || {
    echo "no libncnn.a under $NCNN_BUILD -- build ncnn for android-vulkan first" >&2
    exit 1
}

mkdir -p "$DEST/lib" "$DEST/include"

# libncnn.a first: the link order below mirrors it, and glslang must follow ncnn because
# ncnn is what references it.
cp "$NCNN_BUILD/src/libncnn.a" "$DEST/lib/"
for a in glslang/glslang/libglslang.a \
         glslang/glslang/libMachineIndependent.a \
         glslang/glslang/libGenericCodeGen.a \
         glslang/glslang/OSDependent/Unix/libOSDependent.a \
         glslang/SPIRV/libSPIRV.a \
         glslang/glslang/libglslang-default-resource-limits.a; do
    [ -f "$NCNN_BUILD/$a" ] && cp "$NCNN_BUILD/$a" "$DEST/lib/"
done

# Both header roots. `net.h` and friends are in the source tree; `platform.h`,
# `ncnn_export.h` and the generated layer registries are in the BUILD tree, and ncnn's own
# headers include them unqualified. Flattening the two into one directory is what lets the
# APK's CMake carry a single include path.
cp "$NCNN_ROOT"/src/*.h "$DEST/include/"
cp "$NCNN_BUILD"/src/*.h "$DEST/include/"

# Record what this came from. A staged tree with no provenance is the thing that costs an
# afternoon when a measurement stops reproducing.
{
    echo "ncnn      $(git -C "$NCNN_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "built     $NCNN_BUILD"
    echo "staged    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    grep -E '^(NCNN_(VULKAN|SIMPLEVK|OPENMP|THREADS|SHARED_LIB|INT8|BF16)|ANDROID_(ABI|PLATFORM)|CMAKE_TOOLCHAIN_FILE)' \
        "$NCNN_BUILD/CMakeCache.txt" 2>/dev/null || true
} > "$DEST/STAGED.txt"

echo "staged to $DEST"
cat "$DEST/STAGED.txt"
du -sh "$DEST"
