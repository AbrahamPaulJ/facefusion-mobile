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

}  // namespace ffportrait
