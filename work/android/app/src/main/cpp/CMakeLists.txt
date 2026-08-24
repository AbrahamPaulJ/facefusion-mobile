cmake_minimum_required(VERSION 3.22.1)
project(ffnative)

# The same four objects the headless CLI links (work/native/ffswap_main.cpp), plus the JNI
# bridge.  Keeping them identical is what makes the CLI a valid harness for the app.
add_library(ffnative SHARED ffjni.cpp ffpipe.cpp ffcv.cpp ffqnn.cpp)

# Both roots are needed: ffqnn.cpp includes "QNN/QnnBackend.h", while the SDK's own
# sub-headers include their siblings unqualified.
target_include_directories(ffnative PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/QNN)

# The QNN backend is dlopen'd at runtime from the app's nativeLibraryDir, never linked, so
# no QNN .so is needed at build time and the build is independent of the HTP variant.
target_link_libraries(ffnative android log dl)
target_compile_options(ffnative PRIVATE -O3 -fvisibility=hidden -Wall)
