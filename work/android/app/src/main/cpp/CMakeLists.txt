cmake_minimum_required(VERSION 3.22.1)
project(ffnative)

# The same four objects the headless CLI links (work/native/ffswap_main.cpp), plus the JNI
# bridge.  Keeping them identical is what makes the CLI a valid harness for the app.
add_library(ffnative SHARED ffjni.cpp ffpipe.cpp ffcv.cpp ffqnn.cpp
                            ffnn.cpp ffnn_qnn.cpp)

# Both roots are needed: ffqnn.cpp includes "QNN/QnnBackend.h", while the SDK's own
# sub-headers include their siblings unqualified.
target_include_directories(ffnative PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/QNN)

# The QNN backend is dlopen'd at runtime from the app's nativeLibraryDir, never linked, so
# no QNN .so is needed at build time and the build is independent of the HTP variant.
target_link_libraries(ffnative android log dl)
target_compile_options(ffnative PRIVATE -O3 -fvisibility=hidden -Wall)

# ---------------------------------------------------------------------------
# Optional ncnn backend (roadmap 6): CPU (NEON) or GPU (Vulkan), for parts with no Hexagon.
# ---------------------------------------------------------------------------
# Turned on by build.gradle.kts when `work/android/ncnn/` exists, which is a STAGED copy of
# the WSL build -- see work/android/stage_ncnn.sh for why it is copied rather than
# referenced in place. Delete that directory and the app builds QNN-only, unchanged.
#
# `NCNN_DIR` holds the flattened staging: lib/*.a and one include/ carrying BOTH ncnn header
# roots (the source tree's net.h and the build tree's generated platform.h / layer
# registries, which ncnn's own headers include unqualified).
option(FF_NCNN "Link the ncnn backend for non-Qualcomm parts" OFF)
set(NCNN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../ncnn" CACHE PATH "staged ncnn tree")
if(FF_NCNN)
    if(NOT EXISTS "${NCNN_DIR}/lib/libncnn.a")
        message(FATAL_ERROR "FF_NCNN=ON but no ${NCNN_DIR}/lib/libncnn.a -- run work/android/stage_ncnn.sh from WSL")
    endif()
    target_sources(ffnative PRIVATE ffnn_ncnn.cpp)
    target_compile_definitions(ffnative PRIVATE FFNN_HAVE_NCNN)
    target_include_directories(ffnative PRIVATE "${NCNN_DIR}/include")
    # ORDER MATTERS: libncnn.a references glslang, so glslang follows it. A static link
    # resolves left to right and reversing these produces undefined glslang symbols.
    target_link_libraries(ffnative
        "${NCNN_DIR}/lib/libncnn.a"
        "${NCNN_DIR}/lib/libglslang.a"
        "${NCNN_DIR}/lib/libMachineIndependent.a"
        "${NCNN_DIR}/lib/libGenericCodeGen.a"
        "${NCNN_DIR}/lib/libOSDependent.a"
        "${NCNN_DIR}/lib/libSPIRV.a")
    # ncnn is built NCNN_OPENMP=ON, so its archive carries unresolved `__kmpc_*` symbols;
    # -static-openmp links libomp into this .so rather than expecting one on the device,
    # where there is none. The same two flags the headless CLI uses (work/native/build_cli.sh).
    # -landroid is for ncnn's DataReaderFromAndroidAsset, which libncnn.a always contains
    # even though nothing here calls it.
    #
    # NCNN_SIMPLEVK=ON, so ncnn dlopen's the Vulkan loader itself: no -lvulkan, and a device
    # with no Vulkan fails at get_gpu_count() rather than at load.
    target_link_options(ffnative PRIVATE -fopenmp -static-openmp)
    target_link_libraries(ffnative android)
endif()
