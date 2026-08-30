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
# Optional ncnn backend (roadmap 6). -DFF_NCNN=ON at configure time.
# ---------------------------------------------------------------------------
option(FF_NCNN "Link the ncnn backend for non-Qualcomm parts" OFF)
if(FF_NCNN)
    set(NCNN_ROOT "$ENV{HOME}/ncnn" CACHE PATH "ncnn source root")
    set(NCNN_BUILD "${NCNN_ROOT}/build-android-vulkan" CACHE PATH "ncnn android build")
    target_sources(ffnative PRIVATE ffnn_ncnn.cpp)
    target_compile_definitions(ffnative PRIVATE FFNN_HAVE_NCNN)
    target_include_directories(ffnative PRIVATE "${NCNN_ROOT}/src" "${NCNN_BUILD}/src")
    target_link_libraries(ffnative
        "${NCNN_BUILD}/src/libncnn.a"
        "${NCNN_BUILD}/glslang/glslang/libglslang.a"
        "${NCNN_BUILD}/glslang/glslang/libMachineIndependent.a"
        "${NCNN_BUILD}/glslang/glslang/libGenericCodeGen.a"
        "${NCNN_BUILD}/glslang/glslang/OSDependent/Unix/libOSDependent.a"
        "${NCNN_BUILD}/glslang/SPIRV/libSPIRV.a")
endif()
