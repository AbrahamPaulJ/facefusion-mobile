// Flat C ABI over ffcv, so the host test can drive it from Python with ctypes and compare
// against OpenCV in the same process.  Not used by the app.
#include <cstring>
#include <array>
#include <vector>

#include "../android/app/src/main/cpp/ffcv.h"

using namespace ffcv;

static Affine mkAffine(const double* m) {
  Affine a;
  for (int i = 0; i < 6; ++i) a.m[i] = m[i];
  return a;
}

extern "C" {

void ff_warp_affine_u8(const uint8_t* src, int sw, int sh, const double* m, int dw, int dh,
                       int border, uint8_t* dst) {
  Image in(sw, sh, 3);
  std::memcpy(in.data.data(), src, (size_t)sw * sh * 3);
  Image out = warpAffine(in, mkAffine(m), dw, dh, (Border)border);
  std::memcpy(dst, out.data.data(), (size_t)dw * dh * 3);
}

void ff_resize_linear_u8(const uint8_t* src, int sw, int sh, int dw, int dh, uint8_t* dst) {
  Image in(sw, sh, 3);
  std::memcpy(in.data.data(), src, (size_t)sw * sh * 3);
  Image out = resizeLinear(in, dw, dh);
  std::memcpy(dst, out.data.data(), (size_t)dw * dh * 3);
}

void ff_invert_affine(const double* m, double* out) {
  Affine r = invertAffine(mkAffine(m));
  for (int i = 0; i < 6; ++i) out[i] = r.m[i];
}

void ff_get_affine_transform(const float* src, const float* dst, double* out) {
  Affine r = getAffineTransform(src, dst);
  for (int i = 0; i < 6; ++i) out[i] = r.m[i];
}

void ff_create_bounding_box(const float* landmark68, float* outBox) {
  createBoundingBox(landmark68, outBox);
}

void ff_warp_face_by_bbox(const uint8_t* srcPix, int sw, int sh, const float* box, int size,
                          uint8_t* dst, double* outAffine) {
  Image in(sw, sh, 3);
  std::memcpy(in.data.data(), srcPix, (size_t)sw * sh * 3);
  Affine m;
  Image out = warpFaceByBoundingBox(in, box, size, &m);
  std::memcpy(dst, out.data.data(), (size_t)size * size * 3);
  for (int i = 0; i < 6; ++i) outAffine[i] = m.m[i];
}

void ff_create_area_mask(int w, int h, const float* landmark68, int area, float* dst) {
  MatF m = createAreaMask(w, h, landmark68, (FaceMaskArea)area);
  std::memcpy(dst, m.data.data(), (size_t)w * h * sizeof(float));
}

void ff_umeyama(const float* src, const float* dst, int n, double* out) {
  Affine r = umeyama(src, dst, n);
  for (int i = 0; i < 6; ++i) out[i] = r.m[i];
}

void ff_rot_matrix(double cx, double cy, double ang, double scale, double* out) {
  Affine r = getRotationMatrix2D(cx, cy, ang, scale);
  for (int i = 0; i < 6; ++i) out[i] = r.m[i];
}

int ff_nms(const float* boxes, const float* scores, int n, float st, float nt, int* keep) {
  std::vector<std::array<float, 4>> b(n);
  std::vector<float> s(scores, scores + n);
  for (int i = 0; i < n; ++i)
    b[i] = {boxes[4 * i], boxes[4 * i + 1], boxes[4 * i + 2], boxes[4 * i + 3]};
  auto k = nmsBoxes(b, s, st, nt);
  for (size_t i = 0; i < k.size(); ++i) keep[i] = k[i];
  return (int)k.size();
}

void ff_decode_heatmaps(const float* hm, int n, int hh, int hw, float* xy, float* peak) {
  decodeHeatmaps(hm, n, hh, hw, xy, peak);
}

int ff_face_angle(const float* lm68) { return estimateFaceAngle(lm68); }

void ff_to_landmark5(const float* lm68, float* out5) { toLandmark5(lm68, out5); }

void ff_gaussian_blur(const float* src, int w, int h, double sigma, float* dst) {
  MatF in(w, h, 1);
  std::memcpy(in.data.data(), src, (size_t)w * h * sizeof(float));
  MatF out = gaussianBlur(in, sigma);
  std::memcpy(dst, out.data.data(), (size_t)w * h * sizeof(float));
}

void ff_box_mask(int w, int h, float blur, const int* pad, float* dst) {
  MatF m = createBoxMask(w, h, blur, pad);
  std::memcpy(dst, m.data.data(), (size_t)w * h * sizeof(float));
}

void ff_paste_back(uint8_t* frame, int fw, int fh, const float* crop, int cw, int ch,
                   const float* mask, const double* affine) {
  Image f(fw, fh, 3);
  std::memcpy(f.data.data(), frame, (size_t)fw * fh * 3);
  MatF c(cw, ch, 3);
  std::memcpy(c.data.data(), crop, (size_t)cw * ch * 3 * sizeof(float));
  MatF m(cw, ch, 1);
  std::memcpy(m.data.data(), mask, (size_t)cw * ch * sizeof(float));
  pasteBack(f, c, m, mkAffine(affine));
  std::memcpy(frame, f.data.data(), (size_t)fw * fh * 3);
}

const float* ff_warp_template(int which) { return warpTemplate(which); }

}  // extern "C"
