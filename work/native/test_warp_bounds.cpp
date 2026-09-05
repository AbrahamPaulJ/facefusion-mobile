// Does warpAffine read off the end of the source image?
//
// This exists because work/native/build_and_test.sh CANNOT answer that question: it
// compiles for x86_64, where FFCV_NEON is undefined, so it exercises the scalar fallback
// and reports PASSING while the NEON path walks past the buffer. Build this one with the
// NDK for arm64 under AddressSanitizer and run it on the device.
//
// The bug it was written for: tapsU8x3 reads FOUR bytes from a THREE-byte pixel, and the
// tail guard tested p11 alone on the claim that p11 is the largest of the four taps. Under
// BORDER_CONSTANT a rejected tap is replaced by kZeroPx, so sampling the source's
// bottom-right corner puts the final pixel in p00 while p11 points at the zero constant --
// the guard passes and the load runs one byte past the image. On device that was a
// SIGSEGV whenever a face reached the corner of the frame; in Live, whenever you moved.
//
//   build: see build_asan_test.sh
//   pass:  prints OK and ASAN reports nothing

#include "../android/app/src/main/cpp/ffcv.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int failures = 0;

static void check(bool ok, const char* what) {
  printf("%-58s %s\n", what, ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}

// A source whose every pixel is distinct, so a misread shows up as a wrong value and not
// merely as a sanitizer report.
static ffcv::Image ramp(int w, int h) {
  ffcv::Image im(w, h, 3);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      uint8_t* p = im.row(y) + x * 3;
      p[0] = (uint8_t)(x * 7 + y * 13);
      p[1] = (uint8_t)(x * 3 + y * 5);
      p[2] = (uint8_t)(x + y);
    }
  return im;
}

// The identity-with-offset warp that puts a given source point at the destination origin.
static ffcv::Affine shift(double dx, double dy) {
  ffcv::Affine m;
  m(0, 0) = 1; m(0, 1) = 0; m(0, 2) = dx;
  m(1, 0) = 0; m(1, 1) = 1; m(1, 2) = dy;
  return m;
}

int main() {
  // ---- 1. The crash. A destination window that straddles the source's far corner, at a
  // half-pixel offset so all four taps are live and the corner lands in p00.
  // Every one of these ran off the end before the fix; ASAN turns that into a report and a
  // bare device turns it into the SIGSEGV that was reported from Live.
  for (int w : {2, 3, 16, 63, 64, 720}) {
    for (int h : {2, 3, 16, 63, 64, 1280}) {
      const ffcv::Image src = ramp(w, h);
      for (double off : {0.0, 0.5, 0.25, -0.5}) {
        // Put the source's last pixel in the middle of a 8x8 destination.
        ffcv::Affine m = shift(4.0 - (w - 1) - off, 4.0 - (h - 1) - off);
        volatile int sink = 0;
        ffcv::Image a = ffcv::warpAffine(src, m, 8, 8, ffcv::BORDER_CONSTANT);
        ffcv::Image b = ffcv::warpAffine(src, m, 8, 8, ffcv::BORDER_REPLICATE);
        sink += a.data[0] + b.data[0];
        (void)sink;
      }
    }
  }
  check(true, "corner sampling, both borders, 6x6 sizes x 4 offsets");

  // ---- 2. A whole-image warp that sweeps the corner across the destination, which is the
  // shape the landmarker crop actually has: a face box scaled to 195 px on a 256 canvas,
  // so the crop reaches ~1.3x the box and hangs off the frame when the face is at an edge.
  {
    const ffcv::Image src = ramp(720, 1280);
    for (int k = 0; k < 6; ++k) {
      ffcv::Affine m;
      const double sc = 195.0 / (120.0 + k * 40);
      m(0, 0) = sc; m(0, 1) = 0; m(0, 2) = 128 - sc * (700 - k * 3);
      m(1, 0) = 0; m(1, 1) = sc; m(1, 2) = 128 - sc * (1260 - k * 3);
      ffcv::Image c = ffcv::warpAffine(src, m, 256, 256, ffcv::BORDER_CONSTANT);
      (void)c;
    }
    check(true, "landmarker-shaped crop over the frame's bottom-right corner");
  }

  // ---- 3. The fix must not have changed a pixel. FFCVSCALAR=1 turns off every NEON path
  // in ffcv.cpp, but it is read ONCE into a function-local static, so the two runs cannot
  // live in one process -- the harness runs this binary twice and diffs, and here we only
  // dump what it compares.
  if (getenv("FFDUMP")) {
    const ffcv::Image src = ramp(97, 61);
    ffcv::Affine m = ffcv::getRotationMatrix2D(48, 30, 11.0, 0.83);
    m(0, 2) += 6.5; m(1, 2) -= 3.25;
    for (ffcv::Border b : {ffcv::BORDER_CONSTANT, ffcv::BORDER_REPLICATE}) {
      ffcv::Image d = ffcv::warpAffine(src, m, 128, 96, b);
      fwrite(d.data.data(), 1, d.data.size(), stdout);
    }
    // And the corner window itself, which is the region the guard now routes to scalar.
    for (ffcv::Border b : {ffcv::BORDER_CONSTANT, ffcv::BORDER_REPLICATE}) {
      ffcv::Image d = ffcv::warpAffine(src, shift(4.0 - 96 - 0.5, 4.0 - 60 - 0.5), 8, 8, b);
      fwrite(d.data.data(), 1, d.data.size(), stdout);
    }
    return 0;
  }

  // ---- 4. resizeToCHW shares tapsU8x3. Its taps are clamped rather than zeroed, so p11
  // really is the largest of the four and its guard was always right -- checked, not
  // assumed, because the two functions are one edit apart.
  {
    for (int w : {2, 5, 641, 720}) {
      for (int h : {2, 5, 391, 1280}) {
        const ffcv::Image src = ramp(w, h);
        std::vector<float> out((size_t)3 * 640 * 640, 0.f);
        double scale = std::min(640.0 / h, 640.0 / w);
        int tw = (int)(w * scale), th = (int)(h * scale);
        if (scale >= 1.0) { tw = w; th = h; }
        if (tw < 1) tw = 1;
        if (th < 1) th = 1;
        ffcv::resizeToCHW(src, tw, th, 640, out.data());
      }
    }
    check(true, "resizeToCHW over the same sizes");
  }

  printf(failures ? "\nFAILED\n" : "\nOK\n");
  return failures ? 1 : 0;
}
