// LivePortrait's dense-motion front end on the CPU: the input of the hourglass that the
// NPU graph `lp_gencut` starts from.
//
// The full generator spent 1.7 s of its 1.67 s on the HTP building this tensor -- 22
// trilinear warps of the 4-channel compressed volume through a decomposed grid_sample, and
// 21 keypoint Gaussians -- and 118 ms on everything after it. On the CPU it is small: every
// sparse motion is the identity grid plus ONE constant 3D shift (target_k - source_k), so
// each warp is separable into three 1-D linear interpolations. The oracle is the fp32
// lp_generator.onnx itself (its `/model/dense_motion_network/Reshape_12_output_0`), and
// `work/native/test_ffportrait.py` measures this against it.
//
// Upstream's conventions, which a "sensible" reimplementation would get wrong:
//
//   * grid_sample with align_corners=False and ZERO padding: pixel = ((g + 1) * N) / 2 - 0.5,
//     and a corner outside [0, N-1] contributes nothing. Motion 0 is the identity grid,
//     which under align_corners=False is NOT the identity -- it resamples by N/(N-1).
//   * the identity grid is upstream's fp16 constant, not a float32 linspace; copied, in
//     ffportrait.cpp, by work/export/lp_constants.py.
//   * `source` is subtracted and `target` added (grid - source + target), and the heatmap
//     is Gaussian(source) - Gaussian(target): the ONNX's input NAMES, which are the reverse
//     of face_editor's. Pass the graph's own `source` and `target` here, unchanged.

#pragma once
#include <string>
#include <vector>

namespace ffportrait {

constexpr int kKeypoints = 21;
constexpr int kMotions = kKeypoints + 1;              // the identity motion first
constexpr int kFeatureChannels = 32;                  // feature_volume
constexpr int kCompress = 4;                          // compressed volume
constexpr int kD = 16, kH = 64, kW = 64;
constexpr int kVoxels = kD * kH * kW;
constexpr int kHourglassChannels = kMotions * (1 + kCompress);   // 110

/**
 * lp_gencut's `hg_input`, CHANNELS-LAST [kD][kH][kW][110]: channel k*5 is motion k's
 * heatmap (zero for the identity), k*5+1..k*5+4 its warped compressed volume. Channels-last
 * because that is what the graph takes (fed NCDHW, the HTP's own transpose garbled it).
 *
 * featureVolume: the feature extractor's output, NCDHW [32][16][64][64].
 * source, target: the generator's `source` and `target` keypoints, [21][3] as (x, y, z).
 * out: kVoxels * kHourglassChannels floats.
 */
void hourglassInput(const float* featureVolume, const float* source, const float* target,
                    float* out);

// ------------------------------------------------------------------ face_editor
//
// face_editor/core.py:apply_edit's keypoint arithmetic, op for op. The oracle is
// work/pipeline/lp_editor.py, which test_ffportrait.py checks this against.

// face_editor's sliders, each in [-1, 1], in this order (lp_editor.SLIDERS).
enum Slider {
  kEyebrowDirection, kEyeGazeHorizontal, kEyeGazeVertical, kEyeOpenRatio, kLipOpenRatio,
  kMouthGrim, kMouthPout, kMouthPurse, kMouthSmile, kMouthPositionHorizontal,
  kMouthPositionVertical, kHeadPitch, kHeadYaw, kHeadRoll, kSliderCount
};

// The motion extractor's seven outputs for one face.
struct Motion {
  float pitch = 0, yaw = 0, roll = 0, scale = 0;
  float translation[3] = {};
  float expression[kKeypoints * 3] = {};
  float points[kKeypoints * 3] = {};
};

// The eye retargeter, the lip retargeter and the stitcher: Gemm+ReLU MLPs, from the flat
// file work/export/lp_mlp_export.py writes (lp_mlp_<tier>.bin).
class Mlps {
 public:
  bool load(const std::string& path, std::string* err);
  bool loaded() const { return !nets_[2].empty(); }
  // net 0 eye, 1 lip, 2 stitcher. `in` is that net's input width.
  std::vector<float> run(int net, const float* in) const;

 private:
  struct Layer { int out = 0, in = 0; std::vector<float> w, b; };
  std::vector<Layer> nets_[3];
};

// calculate_distance_ratio on the 68-point landmarks (x, y pairs, any scale).
float distanceRatio(const float* landmark68, int top, int bottom, int left, int right);

// True when at least one slider is non-zero -- with none set the edit is the identity and
// the whole generator can be skipped.
bool anyEdit(const float* sliders);

/**
 * apply_edit's keypoints: `source` is the EDITED face and `target` the face as it is,
 * each [21][3] -- the generator's own `source` / `target` inputs (see hourglassInput).
 * `landmark68` is the face's 68 points, for the eye and lip openers' ratios.
 */
void editKeypoints(const Motion& m, const float* sliders, const float* landmark68,
                   const Mlps& mlps, float* source, float* target);

}  // namespace ffportrait
