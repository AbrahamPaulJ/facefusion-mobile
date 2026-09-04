#include "ffcv.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>

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

void pasteBackRoi(Image& frame, const MatF& crop, const MatF& mask, const Affine& affine,
                  int rx0, int ry0, int rx1, int ry1) {
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
