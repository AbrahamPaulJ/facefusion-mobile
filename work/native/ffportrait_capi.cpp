// Flat C ABI over ffportrait, so the host test can drive it from Python with ctypes and
// compare against the ONNX oracle.  Not used by the app.
#include <cstring>
#include <string>

#include "../android/app/src/main/cpp/ffportrait.h"

extern "C" {

// lp_gencut's hg_input, channels-last: kVoxels * kHourglassChannels floats into dst.
void ff_lp_hourglass_input(const float* featureVolume, const float* source,
                           const float* target, float* dst) {
  ffportrait::hourglassInput(featureVolume, source, target, dst);
}

// face_editor's keypoints. `motion` packs the motion extractor's outputs as pitch, yaw,
// roll, scale, translation[3], expression[63], points[63] (133 floats). Returns 0 when the
// MLP file cannot be loaded.
int ff_lp_edit_keypoints(const float* motion, const float* sliders, const float* landmark68,
                         const char* mlpPath, float* source, float* target) {
  static ffportrait::Mlps mlps;
  static std::string loaded;
  if (loaded != mlpPath) {
    if (!mlps.load(mlpPath, nullptr)) return 0;
    loaded = mlpPath;
  }
  ffportrait::Motion m;
  m.pitch = motion[0]; m.yaw = motion[1]; m.roll = motion[2]; m.scale = motion[3];
  std::memcpy(m.translation, motion + 4, sizeof m.translation);
  std::memcpy(m.expression, motion + 7, sizeof m.expression);
  std::memcpy(m.points, motion + 70, sizeof m.points);
  ffportrait::editKeypoints(m, sliders, landmark68, mlps, source, target);
  return 1;
}

}  // extern "C"
