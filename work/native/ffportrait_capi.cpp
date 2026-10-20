// Flat C ABI over ffportrait, so the host test can drive it from Python with ctypes and
// compare against the ONNX oracle.  Not used by the app.
#include "../android/app/src/main/cpp/ffportrait.h"

extern "C" {

// lp_gencut's hg_input, channels-last: kVoxels * kHourglassChannels floats into dst.
void ff_lp_hourglass_input(const float* featureVolume, const float* source,
                           const float* target, float* dst) {
  ffportrait::hourglassInput(featureVolume, source, target, dst);
}

}  // extern "C"
