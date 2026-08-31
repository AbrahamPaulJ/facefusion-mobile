// Geometry and imaging for the FaceFusion swap chain.
//
// Every function here replaces one OpenCV call on FaceFusion's critical path, and each is
// verified against that call in work/native/test_ffcv.py -- run it before trusting any of
// this.  Nothing in here depends on Android, so the same objects build for the host test,
// the on-device CLI and the APK.
//
// Two facts that shaped the implementation, both measured rather than assumed:
//
//   * cv2.warpAffine SILENTLY IGNORES INTER_AREA. FaceFusion passes flags=INTER_AREA in
//     warp_face_by_face_landmark_5 (face_helper.py:79), but warpAffine only supports
//     INTER_LINEAR and INTER_NEAREST; measured, INTER_AREA and INTER_LINEAR are
//     bit-identical outputs. So only bilinear is needed here.
//
//   * cv2.estimateAffinePartial2D(..., RANSAC, thresh=100) on 5 points is deterministic
//     (0.0 spread over 20 calls) and equals a closed-form Umeyama similarity fit to
//     2.6e-06, so umeyama() below replaces it exactly and without RANSAC.
//
// OpenCV's warpAffine/resize use fixed-point arithmetic with 5 fractional bits; this uses
// float. The difference is measured by the test rather than argued about.

#pragma once
#include <cstdint>
#include <vector>

namespace ffcv {

// ---------------------------------------------------------------- images

// Interleaved 8-bit BGR, the layout every FaceFusion frame uses.
struct Image {
  int w = 0, h = 0, c = 3;
  std::vector<uint8_t> data;
  Image() = default;
  Image(int w_, int h_, int c_ = 3) : w(w_), h(h_), c(c_), data((size_t)w_ * h_ * c_) {}
  uint8_t* row(int y) { return data.data() + (size_t)y * w * c; }
  const uint8_t* row(int y) const { return data.data() + (size_t)y * w * c; }
};

// Single-channel float, used for masks and the swapper's output crop.
struct MatF {
  int w = 0, h = 0, c = 1;
  std::vector<float> data;
  MatF() = default;
  MatF(int w_, int h_, int c_ = 1) : w(w_), h(h_), c(c_), data((size_t)w_ * h_ * c_) {}
  float* row(int y) { return data.data() + (size_t)y * w * c; }
  const float* row(int y) const { return data.data() + (size_t)y * w * c; }
};

// 2x3 affine, row-major, same convention as cv2.
struct Affine {
  double m[6] = {1, 0, 0, 0, 1, 0};
  double& operator()(int r, int cc) { return m[r * 3 + cc]; }
  double operator()(int r, int cc) const { return m[r * 3 + cc]; }
};

enum Border { BORDER_CONSTANT = 0, BORDER_REPLICATE = 1 };

// ---------------------------------------------------------------- transforms

Affine invertAffine(const Affine& a);                       // cv2.invertAffineTransform
void transformPoints(const float* pts, int n, const Affine& a, float* out);  // cv2.transform
Affine getRotationMatrix2D(double cx, double cy, double angleDeg, double scale);
// cv2.estimateAffinePartial2D(src, dst, RANSAC, 100) for the 5-point face case
Affine umeyama(const float* src, const float* dst, int n);

// ---------------------------------------------------------------- resampling

// cv2.warpAffine(src, M, (dw,dh), flags=INTER_LINEAR, borderMode=...)
Image warpAffine(const Image& src, const Affine& M, int dw, int dh, Border border);
MatF warpAffineF(const MatF& src, const Affine& M, int dw, int dh, Border border);
/**
 * warpAffine that fills only [x0, x1) x [y0, y1) of the destination.
 *
 * Everything outside the rectangle is left BLACK, not warped. That is only ever correct
 * when the caller knows those pixels are never read -- the lip syncer's are multiplied by
 * a mask that is zero there -- so this is a sharp tool: a caller that is wrong about the
 * rectangle gets a black band rather than a slow frame.
 */
Image warpAffineRoi(const Image& src, const Affine& M, int dw, int dh, Border border,
                    int x0, int y0, int x1, int y1);

// cv2.resize(src, (dw,dh), interpolation=INTER_LINEAR)
Image resizeLinear(const Image& src, int dw, int dh);

// ---------------------------------------------------------------- detection

// cv2.dnn.NMSBoxes over [x1,y1,x2,y2] boxes; returns kept indices, score-sorted.
std::vector<int> nmsBoxes(const std::vector<std::array<float, 4>>& boxes,
                          const std::vector<float>& scores,
                          float scoreThreshold, float nmsThreshold);

// ---------------------------------------------------------------- landmarks

// The twelve nodes 2dfan4 runs between `heatmaps` and `landmarks`, on the host.
// heatmaps is [68,64,64]; writes 68 (x,y) pairs in heatmap coordinates plus per-point peak.
void decodeHeatmaps(const float* heatmaps, int n, int hh, int hw,
                    float* outXY, float* outPeak);

// facefusion/face_helper.py:220 -- snapped to {0,90,180,270}
int estimateFaceAngle(const float* landmark68);
// facefusion/face_helper.py:208
void toLandmark5(const float* landmark68, float* out5);

// ---------------------------------------------------------------- masking

// facefusion/face_masker.py:188 -- box mask with a Gaussian-blurred border
MatF createBoxMask(int w, int h, float blur, const int padding[4]);
// cv2.GaussianBlur(src, (0,0), sigma) on a single-channel float image
MatF gaussianBlur(const MatF& src, double sigma);

// facefusion/face_helper.py:101 -- paste the swapped crop back through the mask
void pasteBack(Image& frame, const MatF& crop /*HxWx3, 0..255*/, const MatF& mask,
               const Affine& affine);

// ---------------------------------------------------------------- lip syncer crop

// The lip syncer does NOT reuse the swap crop. Upstream warps to ffhq_512 by landmark-5
// exactly as the swapper does, but then takes its own box from the 68 landmarks
// TRANSFORMED INTO that crop, and warps THAT to 96x96. So these operate in crop space,
// not frame space, and the 68 points must have been through transformPoints first.

// cv2.getAffineTransform -- the exact affine through three correspondences.
// `src` and `dst` are 3 (x, y) pairs each.
Affine getAffineTransform(const float* src, const float* dst);

// facefusion/face_helper.py:150 -- min/max over all 68 points, then sorted so x1 <= x2.
// Writes [x1, y1, x2, y2].
void createBoundingBox(const float* landmark68, float* outBox);

// facefusion/face_helper.py:83 -- warp a bounding box to crop_size. Upstream picks
// INTER_AREA when the box is larger than the crop, but cv2.warpAffine SILENTLY IGNORES
// INTER_AREA (measured, see the header comment), so bilinear is both branches.
Image warpFaceByBoundingBox(const Image& src, const float* box, int size, Affine* outAffine);

enum FaceMaskArea { AREA_UPPER_FACE = 0, AREA_LOWER_FACE = 1, AREA_MOUTH = 2 };

// facefusion/face_masker.py:226 -- convex hull of the area's landmark subset, filled,
// blurred at sigma 5, then (clip(0.5, 1) - 0.5) * 2. The landmarks are cast to int32
// FIRST, which truncates toward zero; that is upstream's and it moves the hull.
MatF createAreaMask(int w, int h, const float* landmark68, FaceMaskArea area);

// ---------------------------------------------------------------- warp templates

// facefusion/face_helper.py:10. `which`: 0=arcface_112_v2, 1=arcface_128, 2=ffhq_512
const float* warpTemplate(int which);

}  // namespace ffcv
