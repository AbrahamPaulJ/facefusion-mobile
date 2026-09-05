#include "ffcv.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define FFCV_NEON 1
#endif

namespace ffcv {
namespace {

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline int iclamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Bilinear fetch with the border policy applied to the SAMPLE COORDINATES, which is what
// cv2 does: BORDER_REPLICATE clamps the integer taps, BORDER_CONSTANT returns 0 outside.
template <typename T, int CH>
inline void sampleBilinear(const T* src, int sw, int sh, float x, float y, Border border,
                           float* out) {
  int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
  float ax = x - x0, ay = y - y0;

  int xs[2] = {x0, x0 + 1}, ys[2] = {y0, y0 + 1};
  bool ok[2][2];
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 2; ++i) ok[j][i] = true;

  if (border == BORDER_REPLICATE) {
    for (int i = 0; i < 2; ++i) {
      xs[i] = iclamp(xs[i], 0, sw - 1);
      ys[i] = iclamp(ys[i], 0, sh - 1);
    }
  } else {
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i)
        ok[j][i] = xs[i] >= 0 && xs[i] < sw && ys[j] >= 0 && ys[j] < sh;
  }

  float wgt[2][2] = {{(1 - ax) * (1 - ay), ax * (1 - ay)}, {(1 - ax) * ay, ax * ay}};
  for (int ch = 0; ch < CH; ++ch) out[ch] = 0.f;
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 2; ++i) {
      if (!ok[j][i]) continue;
      const T* p = src + ((size_t)ys[j] * sw + xs[i]) * CH;
      for (int ch = 0; ch < CH; ++ch) out[ch] += wgt[j][i] * (float)p[ch];
    }
}

#ifdef FFCV_NEON
// A zero pixel for BORDER_CONSTANT's rejected taps. They are given a weight of ZERO rather
// than being skipped, so the sum keeps the order the scalar sampler used; the pointer only
// has to be four readable bytes.
const uint8_t kZeroPx[4] = {0, 0, 0, 0};

/**
 * The four bilinear taps of a 3-channel uint8 image, accumulated into b, g, r in lanes 0-2.
 *
 * This is the shape of nearly all the remaining CPU geometry: warpAffine into the three
 * model crops and resizeToCHW into the detector's input are the same twelve scattered byte
 * loads, twelve multiplies and nine adds per pixel, differing only in how they arrive at
 * the taps. One lane per channel collapses that to four widening loads and four
 * multiply-accumulates.
 *
 * ⚠ Reads FOUR bytes from a THREE-byte pixel. In bounds everywhere except the image's
 * final pixel, so every caller must test for that tap and fall back to scalar.
 *
 * ⚠ Test the COORDINATES, not p11. p11 is the largest of the four only when all four are
 * real pixels; under BORDER_CONSTANT a rejected tap becomes kZeroPx, so the final pixel can
 * arrive as p00, p10 or p01 with p11 pointing at the zero constant -- which is a check that
 * passes while the load runs off the end of the image. That was a live SIGSEGV: 0.6.0
 * crashed in the landmarker crop whenever a face reached the frame's bottom-right corner.
 * The 2x2 window touches the final pixel exactly when x0 >= sw-2 && y0 >= sh-2, under
 * either border rule, provided sw and sh are both >= 2 (at 1 a clamp can pull an
 * out-of-range coordinate onto the last pixel from below).
 */
inline float32x4_t tapsU8x3(const uint8_t* p00, const uint8_t* p10,
                            const uint8_t* p01, const uint8_t* p11,
                            float w00, float w10, float w01, float w11) {
  uint32_t a, b, c, d;
  std::memcpy(&a, p00, 4);
  std::memcpy(&b, p10, 4);
  std::memcpy(&c, p01, 4);
  std::memcpy(&d, p11, 4);
  const uint32x4_t packed = {a, b, c, d};
  const uint8x16_t bytes = vreinterpretq_u8_u32(packed);
  const uint16x8_t lo = vmovl_u8(vget_low_u8(bytes));    // taps 00, 10
  const uint16x8_t hi = vmovl_u8(vget_high_u8(bytes));   // taps 01, 11
  float32x4_t acc = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))), w00);
  acc = vfmaq_n_f32(acc, vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo))), w10);
  acc = vfmaq_n_f32(acc, vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))), w01);
  acc = vfmaq_n_f32(acc, vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi))), w11);
  return acc;
}

// vcvtaq_s32_f32 rounds to nearest with ties away from zero, which is exactly std::lround
// -- the rounding every uint8 output in this file uses. Clamped to 0..255 in the vector.
inline void storeU8x3(uint8_t* out, float32x4_t v) {
  int32x4_t i = vcvtaq_s32_f32(v);
  i = vminq_s32(vmaxq_s32(i, vdupq_n_s32(0)), vdupq_n_s32(255));
  out[0] = (uint8_t)vgetq_lane_s32(i, 0);
  out[1] = (uint8_t)vgetq_lane_s32(i, 1);
  out[2] = (uint8_t)vgetq_lane_s32(i, 2);
}

// FFCVSCALAR=1 disables every NEON path in this file at once, for the A/B.
inline bool neonDisabled() {
  static const bool v = getenv("FFCVSCALAR") != nullptr;
  return v;
}
#endif  // FFCV_NEON

}  // namespace

// ---------------------------------------------------------------- transforms

Affine invertAffine(const Affine& a) {
  double det = a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0);
  det = (det != 0.0) ? 1.0 / det : 0.0;
  Affine r;
  r(0, 0) = a(1, 1) * det;
  r(0, 1) = -a(0, 1) * det;
  r(1, 0) = -a(1, 0) * det;
  r(1, 1) = a(0, 0) * det;
  r(0, 2) = -(r(0, 0) * a(0, 2) + r(0, 1) * a(1, 2));
  r(1, 2) = -(r(1, 0) * a(0, 2) + r(1, 1) * a(1, 2));
  return r;
}

void transformPoints(const float* pts, int n, const Affine& a, float* out) {
  for (int i = 0; i < n; ++i) {
    double x = pts[2 * i], y = pts[2 * i + 1];
    out[2 * i] = (float)(a(0, 0) * x + a(0, 1) * y + a(0, 2));
    out[2 * i + 1] = (float)(a(1, 0) * x + a(1, 1) * y + a(1, 2));
  }
}

Affine getRotationMatrix2D(double cx, double cy, double angleDeg, double scale) {
  double a = angleDeg * M_PI / 180.0;
  double alpha = std::cos(a) * scale, beta = std::sin(a) * scale;
  Affine m;
  m(0, 0) = alpha;  m(0, 1) = beta;   m(0, 2) = (1 - alpha) * cx - beta * cy;
  m(1, 0) = -beta;  m(1, 1) = alpha;  m(1, 2) = beta * cx + (1 - alpha) * cy;
  return m;
}

// Closed-form similarity (rotation + uniform scale + translation) least-squares fit.
// Umeyama 1991.  Replaces cv2.estimateAffinePartial2D for the 5-point face case.
Affine umeyama(const float* src, const float* dst, int n) {
  double sx = 0, sy = 0, dx = 0, dy = 0;
  for (int i = 0; i < n; ++i) {
    sx += src[2 * i]; sy += src[2 * i + 1];
    dx += dst[2 * i]; dy += dst[2 * i + 1];
  }
  sx /= n; sy /= n; dx /= n; dy /= n;

  // 4-DOF similarity (uniform scale + rotation + translation) -- exactly the model
  // cv2.estimateAffinePartial2D fits.  For centred sets a_i (src) and b_i (dst) the
  // least-squares solution is closed form and needs no SVD:
  //
  //     dot   = sum(a_i . b_i)          cross = sum(a_i x b_i)
  //     s*cos = dot / sum|a_i|^2        s*sin = cross / sum|a_i|^2
  //
  // which is the minimiser of sum |b_i - sR a_i|^2 over scale and rotation together.
  double dot = 0, cross = 0, saa = 0;
  for (int i = 0; i < n; ++i) {
    double ax = src[2 * i] - sx, ay = src[2 * i + 1] - sy;
    double bx = dst[2 * i] - dx, by = dst[2 * i + 1] - dy;
    dot += ax * bx + ay * by;
    cross += ax * by - ay * bx;
    saa += ax * ax + ay * ay;
  }
  double sc = (saa > 0) ? dot / saa : 1.0;      // s * cos(theta)
  double ss = (saa > 0) ? cross / saa : 0.0;    // s * sin(theta)

  Affine m;
  m(0, 0) = sc;  m(0, 1) = -ss;
  m(1, 0) = ss;  m(1, 1) = sc;
  m(0, 2) = dx - (m(0, 0) * sx + m(0, 1) * sy);
  m(1, 2) = dy - (m(1, 0) * sx + m(1, 1) * sy);
  return m;
}

// ---------------------------------------------------------------- resampling

Image warpAffine(const Image& src, const Affine& M, int dw, int dh, Border border) {
  Image dst(dw, dh, src.c);
  Affine inv = invertAffine(M);
#ifdef FFCV_NEON
  // The sample COORDINATES stay in double and keep the exact expression the scalar path
  // uses -- deliberately not hoisted per row. Only the accumulate is vectorised, so the
  // only arithmetic difference is vfma rounding a multiply-add once where a scalar pair
  // rounds twice, and these crops feed three graphs whose inputs are worth not drifting.
  if (src.c == 3 && src.w >= 2 && src.h >= 2 && !neonDisabled()) {
    const int sw = src.w, sh = src.h;
    const uint8_t* s = src.data.data();
    for (int y = 0; y < dh; ++y) {
      uint8_t* out = dst.row(y);
      for (int x = 0; x < dw; ++x) {
        const float sxf = (float)(inv(0, 0) * x + inv(0, 1) * y + inv(0, 2));
        const float syf = (float)(inv(1, 0) * x + inv(1, 1) * y + inv(1, 2));
        const int x0 = (int)std::floor(sxf), y0 = (int)std::floor(syf);
        // The tail guard, on the coordinates -- see tapsU8x3. A window that can touch the
        // image's final pixel goes scalar; that is a handful of destination pixels at the
        // bottom-right corner, and the alternative is a one-byte read off the end.
        if (x0 >= sw - 2 && y0 >= sh - 2) {
          float px[4];
          sampleBilinear<uint8_t, 3>(s, sw, sh, sxf, syf, border, px);
          for (int ch = 0; ch < 3; ++ch)
            out[x * 3 + ch] = (uint8_t)iclamp((int)std::lround(px[ch]), 0, 255);
          continue;
        }
        const float ax = sxf - x0, ay = syf - y0;
        float w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay);
        float w01 = (1 - ax) * ay,       w11 = ax * ay;
        const uint8_t *p00, *p10, *p01, *p11;
        if (border == BORDER_REPLICATE) {
          const int ix0 = iclamp(x0, 0, sw - 1), ix1 = iclamp(x0 + 1, 0, sw - 1);
          const int iy0 = iclamp(y0, 0, sh - 1), iy1 = iclamp(y0 + 1, 0, sh - 1);
          p00 = s + ((size_t)iy0 * sw + ix0) * 3;
          p10 = s + ((size_t)iy0 * sw + ix1) * 3;
          p01 = s + ((size_t)iy1 * sw + ix0) * 3;
          p11 = s + ((size_t)iy1 * sw + ix1) * 3;
        } else {
          const bool x0ok = (unsigned)x0 < (unsigned)sw, x1ok = (unsigned)(x0 + 1) < (unsigned)sw;
          const bool y0ok = (unsigned)y0 < (unsigned)sh, y1ok = (unsigned)(y0 + 1) < (unsigned)sh;
          p00 = (x0ok && y0ok) ? s + ((size_t)y0 * sw + x0) * 3 : (w00 = 0.f, kZeroPx);
          p10 = (x1ok && y0ok) ? s + ((size_t)y0 * sw + x0 + 1) * 3 : (w10 = 0.f, kZeroPx);
          p01 = (x0ok && y1ok) ? s + ((size_t)(y0 + 1) * sw + x0) * 3 : (w01 = 0.f, kZeroPx);
          p11 = (x1ok && y1ok) ? s + ((size_t)(y0 + 1) * sw + x0 + 1) * 3 : (w11 = 0.f, kZeroPx);
        }
        storeU8x3(out + x * 3, tapsU8x3(p00, p10, p01, p11, w00, w10, w01, w11));
      }
    }
    return dst;
  }
#endif
  for (int y = 0; y < dh; ++y) {
    uint8_t* out = dst.row(y);
    for (int x = 0; x < dw; ++x) {
      float sxf = (float)(inv(0, 0) * x + inv(0, 1) * y + inv(0, 2));
      float syf = (float)(inv(1, 0) * x + inv(1, 1) * y + inv(1, 2));
      float px[4];
      sampleBilinear<uint8_t, 3>(src.data.data(), src.w, src.h, sxf, syf, border, px);
      for (int ch = 0; ch < 3; ++ch)
        out[x * 3 + ch] = (uint8_t)iclamp((int)std::lround(px[ch]), 0, 255);
    }
  }
  return dst;
}

Image warpAffineRoi(const Image& src, const Affine& M, int dw, int dh, Border border,
                    int x0, int y0, int x1, int y1) {
  Image dst(dw, dh, src.c);
  x0 = iclamp(x0, 0, dw); x1 = iclamp(x1, 0, dw);
  y0 = iclamp(y0, 0, dh); y1 = iclamp(y1, 0, dh);
  Affine inv = invertAffine(M);
  for (int y = y0; y < y1; ++y) {
    uint8_t* out = dst.row(y);
    for (int x = x0; x < x1; ++x) {
      float sxf = (float)(inv(0, 0) * x + inv(0, 1) * y + inv(0, 2));
      float syf = (float)(inv(1, 0) * x + inv(1, 1) * y + inv(1, 2));
      float px[4];
      sampleBilinear<uint8_t, 3>(src.data.data(), src.w, src.h, sxf, syf, border, px);
      for (int ch = 0; ch < 3; ++ch)
        out[x * 3 + ch] = (uint8_t)iclamp((int)std::lround(px[ch]), 0, 255);
    }
  }
  return dst;
}

MatF warpAffineF(const MatF& src, const Affine& M, int dw, int dh, Border border) {
  MatF dst(dw, dh, src.c);
  Affine inv = invertAffine(M);
  for (int y = 0; y < dh; ++y) {
    float* out = dst.row(y);
    for (int x = 0; x < dw; ++x) {
      float sxf = (float)(inv(0, 0) * x + inv(0, 1) * y + inv(0, 2));
      float syf = (float)(inv(1, 0) * x + inv(1, 1) * y + inv(1, 2));
      float px[4] = {0, 0, 0, 0};
      if (src.c == 1)
        sampleBilinear<float, 1>(src.data.data(), src.w, src.h, sxf, syf, border, px);
      else
        sampleBilinear<float, 3>(src.data.data(), src.w, src.h, sxf, syf, border, px);
      for (int ch = 0; ch < src.c; ++ch) out[x * src.c + ch] = px[ch];
    }
  }
  return dst;
}

Image resizeLinear(const Image& src, int dw, int dh) {
  Image dst(dw, dh, src.c);
  double fx = (double)src.w / dw, fy = (double)src.h / dh;
  for (int y = 0; y < dh; ++y) {
    // cv2's INTER_LINEAR maps destination pixel CENTRES back to the source
    float sy = (float)((y + 0.5) * fy - 0.5);
    uint8_t* out = dst.row(y);
    for (int x = 0; x < dw; ++x) {
      float sx = (float)((x + 0.5) * fx - 0.5);
      float px[4];
      sampleBilinear<uint8_t, 3>(src.data.data(), src.w, src.h, sx, sy, BORDER_REPLICATE, px);
      for (int ch = 0; ch < 3; ++ch)
        out[x * 3 + ch] = (uint8_t)iclamp((int)std::lround(px[ch]), 0, 255);
    }
  }
  return dst;
}

void resizeToCHW(const Image& src, int dw, int dh, int S, float* dst) {
  if (dw <= 0 || dh <= 0) return;
  const int sw = src.w, sh = src.h;
  const uint8_t* sdata = src.data.data();
  const double fx = (double)sw / dw, fy = (double)sh / dh;

  // The taps and weights for a pure scale are a function of the destination COLUMN alone,
  // so they are built once per row of the output rather than once per pixel of it. Same
  // expressions cv2 uses -- centres mapped back, then BORDER_REPLICATE clamping the taps.
  std::vector<int> xs0(dw), xs1(dw);
  std::vector<float> axs(dw);
  for (int x = 0; x < dw; ++x) {
    const float sx = (float)((x + 0.5) * fx - 0.5);
    const int x0 = (int)std::floor(sx);
    axs[x] = sx - x0;
    xs0[x] = iclamp(x0, 0, sw - 1);
    xs1[x] = iclamp(x0 + 1, 0, sw - 1);
  }

  // The value is a uint8 by the time it is scaled, so /255 has 256 possible answers.
  // A table is both EXACT -- `v * (1.f/255.f)` is not `v / 255.0f`, since 1/255 has no
  // exact float -- and cheaper than 647 K divides.
  static const float* kInv255 = [] {
    static float t[256];
    for (int i = 0; i < 256; ++i) t[i] = i / 255.0f;
    return t;
  }();
  const size_t plane = (size_t)S * S;
  for (int y = 0; y < dh; ++y) {
    const float sy = (float)((y + 0.5) * fy - 0.5);
    const int y0 = (int)std::floor(sy);
    const float ay = sy - y0, omay = 1.f - ay;
    const uint8_t* r0 = sdata + (size_t)iclamp(y0, 0, sh - 1) * sw * 3;
    const uint8_t* r1 = sdata + (size_t)iclamp(y0 + 1, 0, sh - 1) * sw * 3;

    float* o0 = dst + (size_t)y * S;
    float* o1 = o0 + plane;
    float* o2 = o1 + plane;
    for (int x = 0; x < dw; ++x) {
      const float ax = axs[x], omax = 1.f - ax;
      // The four weights in the order the shared sampler accumulated them, so the sum is
      // bit-for-bit the one resizeLinear produced.
      const float w00 = omax * omay, w10 = ax * omay;
      const float w01 = omax * ay,   w11 = ax * ay;
      const uint8_t* p00 = r0 + (size_t)xs0[x] * 3;
      const uint8_t* p10 = r0 + (size_t)xs1[x] * 3;
      const uint8_t* p01 = r1 + (size_t)xs0[x] * 3;
      const uint8_t* p11 = r1 + (size_t)xs1[x] * 3;

#ifdef FFCV_NEON
      // Same four taps, same order, one lane per channel. The tail guard is the last
      // pixel of the source, exactly as in warpAffine.
      if (!neonDisabled() && (size_t)(p11 - sdata) + 4 <= (size_t)sw * sh * 3) {
        int32x4_t vi = vcvtaq_s32_f32(tapsU8x3(p00, p10, p01, p11, w00, w10, w01, w11));
        vi = vminq_s32(vmaxq_s32(vi, vdupq_n_s32(0)), vdupq_n_s32(255));
        o0[x] = kInv255[vgetq_lane_s32(vi, 0)];
        o1[x] = kInv255[vgetq_lane_s32(vi, 1)];
        o2[x] = kInv255[vgetq_lane_s32(vi, 2)];
        continue;
      }
#endif
      const float b = w00 * p00[0] + w10 * p10[0] + w01 * p01[0] + w11 * p11[0];
      const float g = w00 * p00[1] + w10 * p10[1] + w01 * p01[1] + w11 * p11[1];
      const float r = w00 * p00[2] + w10 * p10[2] + w01 * p01[2] + w11 * p11[2];
      // uint8 first, exactly as the two-step version did -- see the header note.
      o0[x] = kInv255[iclamp((int)std::lround(b), 0, 255)];
      o1[x] = kInv255[iclamp((int)std::lround(g), 0, 255)];
      o2[x] = kInv255[iclamp((int)std::lround(r), 0, 255)];
    }
  }
}

// ---------------------------------------------------------------- detection

std::vector<int> nmsBoxes(const std::vector<std::array<float, 4>>& boxes,
                          const std::vector<float>& scores, float scoreThreshold,
                          float nmsThreshold) {
  std::vector<int> order;
  for (size_t i = 0; i < boxes.size(); ++i)
    if (scores[i] > scoreThreshold) order.push_back((int)i);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return scores[a] > scores[b]; });

  std::vector<int> keep;
  std::vector<char> dead(boxes.size(), 0);
  for (size_t oi = 0; oi < order.size(); ++oi) {
    int i = order[oi];
    if (dead[i]) continue;
    keep.push_back(i);
    const auto& A = boxes[i];
    float areaA = (A[2] - A[0]) * (A[3] - A[1]);
    for (size_t oj = oi + 1; oj < order.size(); ++oj) {
      int j = order[oj];
      if (dead[j]) continue;
      const auto& B = boxes[j];
      float xx1 = std::max(A[0], B[0]), yy1 = std::max(A[1], B[1]);
      float xx2 = std::min(A[2], B[2]), yy2 = std::min(A[3], B[3]);
      float w = std::max(0.f, xx2 - xx1), h = std::max(0.f, yy2 - yy1);
      float inter = w * h;
      float areaB = (B[2] - B[0]) * (B[3] - B[1]);
      float denom = areaA + areaB - inter;
      if (denom > 0 && inter / denom > nmsThreshold) dead[j] = 1;
    }
  }
  return keep;
}

// ---------------------------------------------------------------- landmarks

void decodeHeatmaps(const float* hm, int n, int hh, int hw, float* outXY, float* outPeak) {
  // Constants read out of 2dfan4's own initializers, not fitted:
  //   Greater_867 threshold 6.4 ; Clip_899 floor 0.0 ; Clip_901 m00 floor 1.1920929e-07
  //   x_indices/y_indices are PIXEL CENTRES (0.5 .. 63.5), not 0-based
  const float kWindow = 6.4f, kWindow2 = kWindow * kWindow, kEps = 1.1920929e-07f;
  const int r = (int)std::ceil(kWindow);

  for (int k = 0; k < n; ++k) {
    const float* p = hm + (size_t)k * hh * hw;
    int best = 0;
    float bv = p[0];
    for (int i = 1; i < hh * hw; ++i)
      if (p[i] > bv) { bv = p[i]; best = i; }
    int py = best / hw, px = best % hw;
    outPeak[k] = bv;

    // Only a radius-6.4 disc can contribute, so scan a 13x13 neighbourhood, not 64x64.
    double m00 = 0, mx = 0, my = 0;
    int y0 = std::max(py - r, 0), y1 = std::min(py + r, hh - 1);
    int x0 = std::max(px - r, 0), x1 = std::min(px + r, hw - 1);
    for (int y = y0; y <= y1; ++y) {
      int dy = y - py;
      for (int x = x0; x <= x1; ++x) {
        int dx = x - px;
        if ((float)(dx * dx + dy * dy) > kWindow2) continue;
        float v = p[(size_t)y * hw + x];
        if (v < 0.f) v = 0.f;              // Clip lower bound
        m00 += v;
        mx += (double)v * (x + 0.5);
        my += (double)v * (y + 0.5);
      }
    }
    if (m00 < kEps) m00 = kEps;
    outXY[2 * k] = (float)(mx / m00);
    outXY[2 * k + 1] = (float)(my / m00);
  }
}

int estimateFaceAngle(const float* lm68) {
  double x1 = lm68[0], y1 = lm68[1], x2 = lm68[16 * 2], y2 = lm68[16 * 2 + 1];
  double theta = std::atan2(y2 - y1, x2 - x1) * 180.0 / M_PI;
  theta = std::fmod(theta, 360.0);
  if (theta < 0) theta += 360.0;
  const double angles[5] = {0, 90, 180, 270, 360};
  int bi = 0;
  double bd = 1e18;
  for (int i = 0; i < 5; ++i) {
    double d = std::fabs(angles[i] - theta);
    if (d < bd) { bd = d; bi = i; }
  }
  return (int)std::fmod(angles[bi], 360.0);
}

void toLandmark5(const float* lm68, float* out5) {
  auto mean = [&](int a, int b, float* o) {
    double sx = 0, sy = 0;
    for (int i = a; i < b; ++i) { sx += lm68[2 * i]; sy += lm68[2 * i + 1]; }
    o[0] = (float)(sx / (b - a));
    o[1] = (float)(sy / (b - a));
  };
  mean(36, 42, out5 + 0);
  mean(42, 48, out5 + 2);
  out5[4] = lm68[30 * 2];  out5[5] = lm68[30 * 2 + 1];
  out5[6] = lm68[48 * 2];  out5[7] = lm68[48 * 2 + 1];
  out5[8] = lm68[54 * 2];  out5[9] = lm68[54 * 2 + 1];
}

// ---------------------------------------------------------------- masking

MatF gaussianBlur(const MatF& src, double sigma) {
  // cv2 derives the kernel size from sigma: for a float image, ksize = round(sigma*4*2+1)|1
  int ks = (int)std::lround(sigma * 4.0 * 2.0 + 1.0) | 1;
  if (ks < 3) ks = 3;
  int rad = ks / 2;
  std::vector<double> k(ks);
  double sum = 0, s2 = 2.0 * sigma * sigma;
  for (int i = 0; i < ks; ++i) {
    double d = i - rad;
    k[i] = std::exp(-(d * d) / s2);
    sum += k[i];
  }
  for (double& v : k) v /= sum;

  // ⚠ Both passes below accumulate the SAME products in the SAME order as the obvious
  // nested loop, in double, so this is exact -- test_ffcv.py holds it to 0.0 against cv2.
  // What changed is only where the work happens:
  //
  //   * the border reflection is hoisted out of the tap loop.  At sigma 5 the kernel is 41
  //     taps, and every one of them used to cost three branches and a clamp to discover
  //     that it was nowhere near an edge.  The interior is now a straight dot product.
  //   * the vertical pass walks ROWS, not columns.  It used to read tmp.row(y+i-rad)[x]
  //     with i innermost, so each output pixel touched 41 rows -- a stride of w floats per
  //     tap, which misses cache on every one.  Accumulating one tap row across the whole
  //     output row instead touches each input row once.
  //
  // This function is 42% of the lip syncer's geometry (measured 9.53 ms/frame of 21.9),
  // because syncLip calls it once per FRAME where the swapper calls it once per face.
  MatF tmp(src.w, src.h, 1), dst(src.w, src.h, 1);

  auto reflect = [](int i, int n) {
    if (n == 1) return 0;
    if (i < 0) i = -i;
    if (i >= n) i = 2 * (n - 1) - i;
    return iclamp(i, 0, n - 1);
  };

  // ---- horizontal, BORDER_REFLECT_101
  const int xlo = std::min(rad, src.w), xhi = std::max(xlo, src.w - rad);
  for (int y = 0; y < src.h; ++y) {
    const float* srow = src.row(y);
    float* trow = tmp.row(y);
    for (int x = 0; x < xlo; ++x) {                   // left edge
      double a = 0;
      for (int i = 0; i < ks; ++i) a += k[i] * srow[reflect(x + i - rad, src.w)];
      trow[x] = (float)a;
    }
    for (int x = xlo; x < xhi; ++x) {                 // interior: no edge can be reached
      double a = 0;
      const float* p = srow + (x - rad);
      for (int i = 0; i < ks; ++i) a += k[i] * p[i];
      trow[x] = (float)a;
    }
    for (int x = xhi; x < src.w; ++x) {               // right edge
      double a = 0;
      for (int i = 0; i < ks; ++i) a += k[i] * srow[reflect(x + i - rad, src.w)];
      trow[x] = (float)a;
    }
  }

  // ---- vertical, one tap row at a time
  std::vector<double> acc(src.w);
  for (int y = 0; y < src.h; ++y) {
    std::fill(acc.begin(), acc.end(), 0.0);
    for (int i = 0; i < ks; ++i) {
      const int yy = reflect(y + i - rad, src.h);
      const float* trow = tmp.row(yy);
      const double ki = k[i];
      for (int x = 0; x < src.w; ++x) acc[x] += ki * trow[x];
    }
    float* drow = dst.row(y);
    for (int x = 0; x < src.w; ++x) drow[x] = (float)acc[x];
  }
  return dst;
}

MatF createBoxMask(int w, int h, float blur, const int padding[4]) {
  int blurAmount = (int)(w * 0.5f * blur);
  int blurArea = std::max(blurAmount / 2, 1);
  MatF m(w, h, 1);
  std::fill(m.data.begin(), m.data.end(), 1.0f);

  int top = std::max(blurArea, (int)(h * padding[0] / 100.0));
  int bottom = std::max(blurArea, (int)(h * padding[2] / 100.0));
  int left = std::max(blurArea, (int)(w * padding[3] / 100.0));
  int right = std::max(blurArea, (int)(w * padding[1] / 100.0));

  for (int y = 0; y < std::min(top, h); ++y)
    std::fill(m.row(y), m.row(y) + w, 0.f);
  for (int y = std::max(0, h - bottom); y < h; ++y)
    std::fill(m.row(y), m.row(y) + w, 0.f);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < std::min(left, w); ++x) m.row(y)[x] = 0.f;
    for (int x = std::max(0, w - right); x < w; ++x) m.row(y)[x] = 0.f;
  }
  if (blurAmount > 0) m = gaussianBlur(m, blurAmount * 0.25);
  return m;
}

void pasteBack(Image& frame, const MatF& crop, const MatF& mask, const Affine& affine) {
  pasteBackRoi(frame, crop, mask, affine, 0, 0, crop.w, crop.h);
}

// The pre-0.5.2 paste: two full bilinear warps into two float buffers, then a blend pass
// over them. Kept ONLY as the A/B reference for the fused version below, selected by
// FFPASTELEGACY=1, so one binary can measure both inside a single thermal session --
// which is the only kind of comparison this device gives a trustworthy answer to.
// Delete once the fused path has been exact and faster across a release.
static void pasteBackRoiLegacy(Image& frame, const MatF& crop, const MatF& mask,
                               const Affine& affine, int rx0, int ry0, int rx1, int ry1) {
  Affine inv = invertAffine(affine);
  // The destination box is the projection of the caller's RECTANGLE, not of the whole
  // crop. The lip syncer's mask is non-zero only over the lower face, so projecting all
  // 512x512 corners warped roughly five times the area that the blend could touch, and
  // both warps below are bilinear over every pixel of it. Outside the rectangle the mask
  // is zero and the blend is the identity, so shrinking the box changes no output pixel.
  rx0 = iclamp(rx0, 0, crop.w); rx1 = iclamp(rx1, 0, crop.w);
  ry0 = iclamp(ry0, 0, crop.h); ry1 = iclamp(ry1, 0, crop.h);
  if (rx1 <= rx0 || ry1 <= ry0) return;
  float corners[8] = {(float)rx0, (float)ry0, (float)rx1, (float)ry0,
                      (float)rx1, (float)ry1, (float)rx0, (float)ry1};
  float out[8];
  transformPoints(corners, 4, inv, out);
  float minx = out[0], maxx = out[0], miny = out[1], maxy = out[1];
  for (int i = 1; i < 4; ++i) {
    minx = std::min(minx, out[2 * i]);   maxx = std::max(maxx, out[2 * i]);
    miny = std::min(miny, out[2 * i + 1]); maxy = std::max(maxy, out[2 * i + 1]);
  }
  int x1 = iclamp((int)std::floor(minx), 0, frame.w);
  int y1 = iclamp((int)std::floor(miny), 0, frame.h);
  int x2 = iclamp((int)std::ceil(maxx), 0, frame.w);
  int y2 = iclamp((int)std::ceil(maxy), 0, frame.h);
  int pw = x2 - x1, ph = y2 - y1;
  if (pw <= 0 || ph <= 0) return;

  Affine paste = inv;
  paste(0, 2) -= x1;
  paste(1, 2) -= y1;

  MatF im = warpAffineF(mask, paste, pw, ph, BORDER_CONSTANT);
  MatF ic = warpAffineF(crop, paste, pw, ph, BORDER_REPLICATE);

  for (int y = 0; y < ph; ++y) {
    uint8_t* dst = frame.row(y1 + y) + (size_t)x1 * 3;
    const float* mrow = im.row(y);
    const float* crow = ic.row(y);
    for (int x = 0; x < pw; ++x) {
      float a = clampf(mrow[x], 0.f, 1.f);
      for (int ch = 0; ch < 3; ++ch) {
        float v = dst[x * 3 + ch] * (1.f - a) + crow[x * 3 + ch] * a;
        dst[x * 3 + ch] = (uint8_t)iclamp((int)v, 0, 255);   // numpy astype = truncate
      }
    }
  }
}

/**
 * paste_back, fused: sample the mask, sample the crop and blend in ONE pass.
 *
 * The three-pass version above measured 11.02 ms/frame on a 1366x720 clip -- the largest
 * single item in the CPU geometry -- and almost none of it was arithmetic. Per destination
 * pixel it called a templated `sampleBilinear` twice through a runtime border argument, so
 * every pixel re-ran a 2x2 validity array, two `std::floor`s and four branches it could not
 * hoist, and it wrote 2.5 MB of float intermediates only to read them straight back.
 *
 * Fusing removes all of that at once:
 *
 *   - ONE inverse affine instead of two (warpAffineF inverted its argument internally, so
 *     the same matrix was inverted on both calls),
 *   - no `im`/`ic` buffers, so no allocation, no zero-fill, and no 2.5 MB round trip to
 *     memory between producing a value and consuming it,
 *   - the border policy resolved per CALL SITE rather than per pixel: the mask is
 *     BORDER_CONSTANT and the crop BORDER_REPLICATE, both known here,
 *   - and an early-out on a zero mask sample, which is the algorithmic part. The box is the
 *     axis-aligned bound of a ROTATED crop, so a third of it maps outside the mask entirely
 *     and a further ring falls in the mask's zero padding; a == 0 makes the blend the exact
 *     identity, so those pixels can skip the 3-channel sampler without changing a byte.
 *
 * ⚠ Bit-exactness is deliberate, not incidental. Weights are accumulated in the same
 * w00, w10, w01, w11 order the templated sampler used, and taps that the border rejects
 * contribute a zero term rather than being skipped, because `+= 0.f` is an identity on
 * floats and reordering the sum is not. The one liberty taken is hoisting `W(0,1)*y +
 * W(0,2)` out of the x loop, which re-associates one double add -- worth a last bit far
 * below the test's 2 LSB bound, and it removes two multiplies per pixel.
 */
void pasteBackRoi(Image& frame, const MatF& crop, const MatF& mask, const Affine& affine,
                  int rx0, int ry0, int rx1, int ry1) {
  if (getenv("FFPASTELEGACY"))
    return pasteBackRoiLegacy(frame, crop, mask, affine, rx0, ry0, rx1, ry1);

  Affine inv = invertAffine(affine);
  rx0 = iclamp(rx0, 0, crop.w); rx1 = iclamp(rx1, 0, crop.w);
  ry0 = iclamp(ry0, 0, crop.h); ry1 = iclamp(ry1, 0, crop.h);
  if (rx1 <= rx0 || ry1 <= ry0) return;
  float corners[8] = {(float)rx0, (float)ry0, (float)rx1, (float)ry0,
                      (float)rx1, (float)ry1, (float)rx0, (float)ry1};
  float out[8];
  transformPoints(corners, 4, inv, out);
  float minx = out[0], maxx = out[0], miny = out[1], maxy = out[1];
  for (int i = 1; i < 4; ++i) {
    minx = std::min(minx, out[2 * i]);   maxx = std::max(maxx, out[2 * i]);
    miny = std::min(miny, out[2 * i + 1]); maxy = std::max(maxy, out[2 * i + 1]);
  }
  int x1 = iclamp((int)std::floor(minx), 0, frame.w);
  int y1 = iclamp((int)std::floor(miny), 0, frame.h);
  int x2 = iclamp((int)std::ceil(maxx), 0, frame.w);
  int y2 = iclamp((int)std::ceil(maxy), 0, frame.h);
  int pw = x2 - x1, ph = y2 - y1;
  if (pw <= 0 || ph <= 0) return;

  Affine paste = inv;
  paste(0, 2) -= x1;
  paste(1, 2) -= y1;
  // What warpAffineF computed twice, internally: box pixel -> crop/mask coordinates.
  Affine W = invertAffine(paste);

  const int mw = mask.w, mh = mask.h, cw = crop.w, ch_ = crop.h;
  const float* mdata = mask.data.data();
  const float* cdata = crop.data.data();

  // TILED, and this is where the time actually was. The affine is a rotation, so walking
  // the destination in raster order walks the crop DIAGONALLY: a destination row touches a
  // new crop row every few pixels, uses one or two of the five pixels in each 64-byte line,
  // and by the time the next destination row wants that line it has been evicted. Measured
  // at ~300 cycles per pixel for about 30 flops and 16 loads -- the arithmetic was never
  // the cost, and no amount of SIMD would have found that out.
  //
  // A 64x64 destination tile touches roughly 64x64 of the crop, ~49 KB of float RGB, which
  // stays resident while the tile is worked. Every pixel's computation is UNCHANGED and
  // independent of every other, so this reorders nothing arithmetically: the output is
  // bit-identical to the untiled loop, not merely within tolerance.
  //
  // FFPASTETILE overrides the size, 0 disabling tiling, so the sweep that chose 64 can be
  // rerun on other silicon without a rebuild.
  static const int kTile = [] {
    const char* e = getenv("FFPASTETILE");
    return e ? atoi(e) : 64;
  }();
  const int tile = kTile > 0 ? kTile : (pw > ph ? pw : ph);

  // FFPASTESCALAR=1 keeps the scalar blend, so the NEON path is A/B-able in one binary.
  //
  // FFCVSCALAR is documented as turning off every NEON path in this file, and until 0.5.2
  // it did not reach this one -- so a whole-file A/B ran the NEON blend in BOTH arms and
  // reported its cost as noise (-0.45 ms/frame, sign flipping with the run order). It is
  // ORed in here so the switch means what the docs and the working rules say it means.
#ifdef FFCV_NEON
  static const bool neonOff = getenv("FFPASTESCALAR") != nullptr || neonDisabled();
#endif

  // FFPASTENOFULL=1 keeps the unconditional blend, for the A/B.
  //
  // MEASURED before it was written, because the last two guesses at this loop were both
  // wrong: 33.0% of blended pixels carry alpha EXACTLY 1.0f (29.4% of the whole box), so a
  // third of the work loads the destination, multiplies it by zero and adds it back. The
  // mask is a box blurred at sigma 9.6, and its interior really is a flat 1.0.
  //
  // Bit-exact rather than close: with a == 1, ia == 0, and the blend it replaces is
  // `d*0 + cv*1` -- +0 added to a finite cv, which IEEE returns as cv unchanged. Only
  // EXACT equality qualifies, so the interior pixels whose four weights sum to 0.99999994
  // instead take the ordinary path and are unaffected. That is also why this is 33% and
  // not the ~60% the mask's flat interior would suggest geometrically.
  static const bool noFull = getenv("FFPASTENOFULL") != nullptr;

  // FFPASTEDBG reports the box actually being blended. Guessing at it produced two wrong
  // hypotheses in a row -- a working set that would thrash the cache, then a tile size to
  // fix it -- and the tiling measured as nothing at all.
  static const bool kDbg = getenv("FFPASTEDBG") != nullptr;
  static int dbgLeft = 3;
  const bool dbg = kDbg && dbgLeft > 0;
  long nOutside = 0, nZero = 0, nBlend = 0, nFull = 0;

  for (int ty = 0; ty < ph; ty += tile) {
  const int tyEnd = ty + tile < ph ? ty + tile : ph;
  for (int tx = 0; tx < pw; tx += tile) {
  const int txEnd = tx + tile < pw ? tx + tile : pw;
  for (int y = ty; y < tyEnd; ++y) {
    uint8_t* dst = frame.row(y1 + y) + (size_t)x1 * 3;
    const double bx = W(0, 1) * y + W(0, 2);
    const double by = W(1, 1) * y + W(1, 2);
    for (int x = tx; x < txEnd; ++x) {
      const float sxf = (float)(W(0, 0) * x + bx);
      const float syf = (float)(W(1, 0) * x + by);
      const int px0 = (int)std::floor(sxf), py0 = (int)std::floor(syf);

      // Wholly outside the mask: alpha is 0 and the blend is the identity.
      if (px0 + 1 < 0 || px0 >= mw || py0 + 1 < 0 || py0 >= mh) { if (dbg) ++nOutside; continue; }

      const float ax = sxf - px0, ay = syf - py0;
      const float w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay);
      const float w01 = (1 - ax) * ay,       w11 = ax * ay;
      const int px1 = px0 + 1, py1 = py0 + 1;

      // BORDER_CONSTANT: a rejected tap contributes a zero term, in the original order.
      float a = 0.f;
      const bool mx0ok = (unsigned)px0 < (unsigned)mw, mx1ok = (unsigned)px1 < (unsigned)mw;
      if ((unsigned)py0 < (unsigned)mh) {
        const float* r = mdata + (size_t)py0 * mw;
        a += (mx0ok ? w00 * r[px0] : 0.f);
        a += (mx1ok ? w10 * r[px1] : 0.f);
      } else {
        a += 0.f; a += 0.f;
      }
      if ((unsigned)py1 < (unsigned)mh) {
        const float* r = mdata + (size_t)py1 * mw;
        a += (mx0ok ? w01 * r[px0] : 0.f);
        a += (mx1ok ? w11 * r[px1] : 0.f);
      } else {
        a += 0.f; a += 0.f;
      }
      a = clampf(a, 0.f, 1.f);
      if (a == 0.f) { if (dbg) ++nZero; continue; }   // exact identity: skip the crop sampler
      if (dbg) { ++nBlend; if (a == 1.f) ++nFull; }

      // BORDER_REPLICATE: clamp the taps, which is what cv2 does and what the templated
      // sampler did for this border.
      const int ix0 = iclamp(px0, 0, cw - 1), ix1 = iclamp(px1, 0, cw - 1);
      const int iy0 = iclamp(py0, 0, ch_ - 1), iy1 = iclamp(py1, 0, ch_ - 1);
      const float* p00 = cdata + ((size_t)iy0 * cw + ix0) * 3;
      const float* p10 = cdata + ((size_t)iy0 * cw + ix1) * 3;
      const float* p01 = cdata + ((size_t)iy1 * cw + ix0) * 3;
      const float* p11 = cdata + ((size_t)iy1 * cw + ix1) * 3;

      const float ia = 1.f - a;
      uint8_t* d = dst + x * 3;
#ifdef FFCV_NEON
      // The three channels ARE the vector. A destination pixel needs 12 loads, 12
      // multiplies and 9 adds to gather its four taps, and every one of them is the same
      // operation on b, g and r -- so one lane per channel turns that into four loads and
      // four multiply-accumulates, with the fourth lane simply carried and discarded.
      //
      // ⚠ `vld1q_f32` reads FOUR floats from a three-float pixel. That is deliberate and
      // in bounds everywhere except the final pixel of the crop, where it would run one
      // float past the vector's storage -- so that one case falls back to scalar. It is
      // reached only by a sample landing on the extreme bottom-right corner.
      if (!neonOff && (size_t)iy1 * cw + ix1 + 1 < (size_t)cw * ch_) {
        float32x4_t acc = vmulq_n_f32(vld1q_f32(p00), w00);
        acc = vfmaq_n_f32(acc, vld1q_f32(p10), w10);
        acc = vfmaq_n_f32(acc, vld1q_f32(p01), w01);
        acc = vfmaq_n_f32(acc, vld1q_f32(p11), w11);
        float32x4_t res;
        if (a == 1.f && !noFull) {
          res = acc;                    // d*0 + acc*1, without touching d
        } else {
          float32x4_t dv = {(float)d[0], (float)d[1], (float)d[2], 0.f};
          res = vmulq_n_f32(dv, ia);
          res = vfmaq_n_f32(res, acc, a);
        }
        // vcvtq_s32_f32 truncates toward zero, which is the numpy astype this mirrors.
        int32x4_t vi = vcvtq_s32_f32(res);
        vi = vminq_s32(vmaxq_s32(vi, vdupq_n_s32(0)), vdupq_n_s32(255));
        d[0] = (uint8_t)vgetq_lane_s32(vi, 0);
        d[1] = (uint8_t)vgetq_lane_s32(vi, 1);
        d[2] = (uint8_t)vgetq_lane_s32(vi, 2);
      } else
#endif
      for (int c = 0; c < 3; ++c) {
        const float cv = w00 * p00[c] + w10 * p10[c] + w01 * p01[c] + w11 * p11[c];
        const float v = (a == 1.f && !noFull) ? cv : d[c] * ia + cv * a;
        d[c] = (uint8_t)iclamp((int)v, 0, 255);   // numpy astype = truncate
      }
    }
  }
  }
  }
  if (dbg) {
    --dbgLeft;
    const long tot = (long)pw * ph;
    fprintf(stderr,
            "[paste] box %dx%d = %ld px  crop %dx%d  |  outside %ld (%.1f%%)  "
            "zero-alpha %ld (%.1f%%)  blended %ld (%.1f%%)  of which FULL-alpha "
            "%ld (%.1f%% of frame, %.1f%% of blended)\n",
            pw, ph, tot, cw, ch_,
            nOutside, 100.0 * nOutside / tot, nZero, 100.0 * nZero / tot,
            nBlend, 100.0 * nBlend / tot,
            nFull, 100.0 * nFull / tot, nBlend ? 100.0 * nFull / nBlend : 0.0);
  }
}

// ---------------------------------------------------------------- warp templates

// ---------------------------------------------------------------- lip syncer crop

Affine getAffineTransform(const float* src, const float* dst) {
  // Two independent 3x3 solves, one per output row, by Cramer's rule. The system is
  //   [x0 y0 1][a]   [u0]
  //   [x1 y1 1][b] = [u1]
  //   [x2 y2 1][c]   [u2]
  const double x0 = src[0], y0 = src[1], x1 = src[2], y1 = src[3], x2 = src[4], y2 = src[5];
  const double det = x0 * (y1 - y2) - y0 * (x1 - x2) + (x1 * y2 - x2 * y1);
  Affine out;
  if (det == 0.0) return out;   // degenerate: three collinear points, identity is honest
  for (int r = 0; r < 2; ++r) {
    const double u0 = dst[0 + r], u1 = dst[2 + r], u2 = dst[4 + r];
    const double da = u0 * (y1 - y2) - y0 * (u1 - u2) + (u1 * y2 - u2 * y1);
    const double db = x0 * (u1 - u2) - u0 * (x1 - x2) + (x1 * u2 - x2 * u1);
    const double dc = x0 * (y1 * u2 - y2 * u1) - y0 * (x1 * u2 - x2 * u1)
                    + u0 * (x1 * y2 - x2 * y1);
    out.m[r * 3 + 0] = da / det;
    out.m[r * 3 + 1] = db / det;
    out.m[r * 3 + 2] = dc / det;
  }
  return out;
}

void createBoundingBox(const float* landmark68, float* outBox) {
  float x1 = landmark68[0], y1 = landmark68[1], x2 = landmark68[0], y2 = landmark68[1];
  for (int i = 1; i < 68; ++i) {
    x1 = std::min(x1, landmark68[i * 2]);
    x2 = std::max(x2, landmark68[i * 2]);
    y1 = std::min(y1, landmark68[i * 2 + 1]);
    y2 = std::max(y2, landmark68[i * 2 + 1]);
  }
  // normalize_bounding_box sorts each pair; min/max already did, but upstream applies it
  // and a caller may hand this an arbitrary box later.
  outBox[0] = std::min(x1, x2);
  outBox[1] = std::min(y1, y2);
  outBox[2] = std::max(x1, x2);
  outBox[3] = std::max(y1, y2);
}

Image warpFaceByBoundingBox(const Image& src, const float* box, int size, Affine* outAffine) {
  const float srcPts[6] = {box[0], box[1], box[2], box[1], box[0], box[3]};
  const float dstPts[6] = {0.f, 0.f, (float)size, 0.f, 0.f, (float)size};
  Affine m = getAffineTransform(srcPts, dstPts);
  if (outAffine) *outAffine = m;
  return warpAffine(src, m, size, size, BORDER_CONSTANT);
}

namespace {

// facefusion/choices.py:26. Indices into the 68-point set.
const std::vector<int>& areaPoints(FaceMaskArea area) {
  static const std::vector<int> kUpper = {0, 1, 2, 31, 32, 33, 34, 35, 14, 15, 16,
                                          26, 25, 24, 23, 22, 21, 20, 19, 18, 17};
  static const std::vector<int> kLower = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                          35, 34, 33, 32, 31};
  static const std::vector<int> kMouth = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                                          60, 61, 62, 63, 64, 65, 66, 67};
  switch (area) {
    case AREA_UPPER_FACE: return kUpper;
    case AREA_MOUTH: return kMouth;
    default: return kLower;
  }
}

// cv2.convexHull over integer points: Andrew's monotone chain, counter-clockwise, with
// collinear points dropped the way OpenCV drops them.
std::vector<std::array<int, 2>> convexHull(std::vector<std::array<int, 2>> pts) {
  std::sort(pts.begin(), pts.end(), [](const std::array<int, 2>& a, const std::array<int, 2>& b) {
    return a[0] != b[0] ? a[0] < b[0] : a[1] < b[1];
  });
  pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
  if (pts.size() < 3) return pts;

  auto cross = [](const std::array<int, 2>& o, const std::array<int, 2>& a,
                  const std::array<int, 2>& b) {
    return (long long)(a[0] - o[0]) * (b[1] - o[1]) - (long long)(a[1] - o[1]) * (b[0] - o[0]);
  };
  std::vector<std::array<int, 2>> hull(pts.size() * 2);
  size_t k = 0;
  for (size_t i = 0; i < pts.size(); ++i) {
    while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) --k;
    hull[k++] = pts[i];
  }
  const size_t lower = k + 1;
  for (size_t i = pts.size() - 1; i-- > 0;) {
    while (k >= lower && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) --k;
    hull[k++] = pts[i];
  }
  hull.resize(k - 1);
  return hull;
}

}  // namespace

MatF createAreaMask(int w, int h, const float* landmark68, FaceMaskArea area) {
  MatF mask(w, h, 1);
  std::fill(mask.data.begin(), mask.data.end(), 0.f);

  // astype(numpy.int32) truncates TOWARD ZERO, which is not floor for negatives. The
  // hull is built on those truncated points, so doing it any other way moves the mask.
  std::vector<std::array<int, 2>> pts;
  for (int index : areaPoints(area)) {
    pts.push_back({(int)landmark68[index * 2], (int)landmark68[index * 2 + 1]});
  }
  std::vector<std::array<int, 2>> hull = convexHull(pts);
  if (hull.size() >= 3) {
    // cv2.fillConvexPoly: one span per scanline between the polygon's left and right
    // edges. Convexity is what makes a single span correct.
    int top = hull[0][1], bottom = hull[0][1];
    for (const std::array<int, 2>& p : hull) {
      top = std::min(top, p[1]);
      bottom = std::max(bottom, p[1]);
    }
    top = std::max(top, 0);
    bottom = std::min(bottom, h - 1);
    for (int y = top; y <= bottom; ++y) {
      double left = 1e30, right = -1e30;
      for (size_t i = 0; i < hull.size(); ++i) {
        const std::array<int, 2>& a = hull[i];
        const std::array<int, 2>& b = hull[(i + 1) % hull.size()];
        const double ay = a[1], by = b[1];
        const double yLo = std::min(ay, by), yHi = std::max(ay, by);
        if (ay == by) {
          if ((int)ay == y) {
            left = std::min(left, (double)std::min(a[0], b[0]));
            right = std::max(right, (double)std::max(a[0], b[0]));
          }
          continue;
        }
        // cv2.fillConvexPoly does NOT sample the edge at the row centre. Its span covers
        // everything the edge SWEEPS THROUGH between y - 0.5 and y + 0.5, so a shallow
        // edge widens the row by half its dx/dy. Derived by probing cv2 with a known
        // triangle, then confirmed on the real hull: at a row where the true edge sits at
        // 206.59 with slope -3.41/row, cv2 fills from 205, which is the sweep's minimum
        // rounded, not the centre's. Sampling the centre instead loses 1.26% of the mask
        // area, all of it around the boundary where the mouth is blended.
        const double s0 = std::max((double)y - 0.5, yLo);
        const double s1 = std::min((double)y + 0.5, yHi);
        if (s0 > s1) continue;
        for (double sy : {s0, s1}) {
          const double x = a[0] + (sy - ay) / (by - ay) * (b[0] - a[0]);
          left = std::min(left, x);
          right = std::max(right, x);
        }
      }
      if (left > right) continue;
      // ...and the ends are CEIL on the left, FLOOR on the right, not rounded: the span
      // is the integer pixels lying inside the swept interval. Rounding both ends instead
      // still lost 0.66% of the area. Both halves of this rule were derived by probing
      // cv2 rather than read off its source, then checked row by row against the real
      // hull -- 42 of 71 rows disagreed on rounding alone.
      const int x0 = std::max(0, (int)std::ceil(left));
      const int x1 = std::min(w - 1, (int)std::floor(right));
      for (int x = x0; x <= x1; ++x) mask.row(y)[x] = 1.f;
    }
  }

  // The blur is the expensive half of this function -- measured 20.7 ms of 21.3 on a
  // 512x512 host, and the lip syncer calls it once per FRAME, where the swap's box mask
  // was once per face. It is restricted to the hull's bounding box grown by the kernel
  // radius, which is EXACT rather than an approximation: outside the hull the fill is 0,
  // a blur of 0 is 0, and (clip(0, 0.5, 1) - 0.5) * 2 is 0 too. So every pixel this skips
  // was going to be zero.
  //
  // The box comes from the HULL, not from scanning the canvas for non-zero pixels. Both
  // give the same answer and the scan cost 262144 reads per frame to rediscover something
  // already in hand -- 42% of the lip syncer's geometry was this function (measured,
  // 9.28 ms/frame of 21.90). It is exact rather than close: the span for a row is swept
  // along the hull's own edges, so every x it fills lies between two hull vertices, and
  // the rows are the hull's own y range. A box that is a superset would be fine anyway,
  // since blurring zeros yields zeros -- but this one is not even a superset, it is equal.
  const int rad = (int)std::lround(5.0 * 4.0 * 2.0 + 1.0) / 2;
  if (hull.size() < 3) return mask;   // nothing filled: an all-zero mask is already correct
  int bx0 = hull[0][0], by0 = hull[0][1], bx1 = hull[0][0], by1 = hull[0][1];
  for (const std::array<int, 2>& p : hull) {
    bx0 = std::min(bx0, p[0]); bx1 = std::max(bx1, p[0]);
    by0 = std::min(by0, p[1]); by1 = std::max(by1, p[1]);
  }
  bx0 = std::max(bx0, 0); by0 = std::max(by0, 0);
  bx1 = std::min(bx1, w - 1); by1 = std::min(by1, h - 1);
  if (bx1 < bx0 || by1 < by0) return mask;   // hull entirely off-canvas

  bx0 = std::max(0, bx0 - 2 * rad); by0 = std::max(0, by0 - 2 * rad);
  bx1 = std::min(w - 1, bx1 + 2 * rad); by1 = std::min(h - 1, by1 + 2 * rad);
  const int rw = bx1 - bx0 + 1, rh = by1 - by0 + 1;

  MatF window(rw, rh, 1);
  for (int y = 0; y < rh; ++y)
    std::memcpy(window.row(y), mask.row(by0 + y) + bx0, (size_t)rw * sizeof(float));
  window = gaussianBlur(window, 5.0);

  // No second std::fill: the canvas was zeroed on entry, the only pixels written since are
  // the hull's 1s, and the hull is inside this window by construction -- so everything
  // outside it is ALREADY zero and the loop below overwrites everything inside.
  for (int y = 0; y < rh; ++y) {
    const float* srow = window.row(y);
    float* drow = mask.row(by0 + y) + bx0;
    for (int x = 0; x < rw; ++x) {
      drow[x] = (std::min(std::max(srow[x], 0.5f), 1.0f) - 0.5f) * 2.0f;
    }
  }
  return mask;
}

const float* warpTemplate(int which) {
  static const float kArc112v2[10] = {
      0.34191607f, 0.46157411f, 0.65653393f, 0.45983393f, 0.50022500f,
      0.64050536f, 0.37097589f, 0.82469196f, 0.63151696f, 0.82325089f};
  static const float kArc128[10] = {
      0.36167656f, 0.40387734f, 0.63696719f, 0.40235469f, 0.50019687f,
      0.56044219f, 0.38710391f, 0.72160547f, 0.61507734f, 0.72034453f};
  static const float kFfhq512[10] = {
      0.37691676f, 0.46864664f, 0.62285697f, 0.46912813f, 0.50123859f,
      0.61331904f, 0.39308822f, 0.72541100f, 0.61150205f, 0.72490465f};
  switch (which) {
    case 1: return kArc128;
    case 2: return kFfhq512;
    default: return kArc112v2;
  }
}

}  // namespace ffcv
