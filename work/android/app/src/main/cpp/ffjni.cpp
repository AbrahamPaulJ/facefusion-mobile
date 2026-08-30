// JNI surface for the app.
//
// The pipeline, the geometry and the QNN runner are the SAME objects the headless CLI
// links (work/native/ffswap_main.cpp), which is what makes the CLI a valid harness for
// the app: if a swap is right over adb it is right here.
//
// Colour conversion lives here rather than in Kotlin because it is per-pixel work on every
// frame -- 2.95 M pixels at 720p -- and a Kotlin loop over that is not viable.

#include <jni.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "ffcv.h"
#include "ffpipe.h"
#include "ffnn.h"
#include "ffqnn.h"

namespace {

std::unique_ptr<ffpipe::Pipeline> g_pipe;
std::string g_err;

std::string jstr(JNIEnv* env, jstring s) {
  if (!s) return {};
  const char* c = env->GetStringUTFChars(s, nullptr);
  std::string out(c ? c : "");
  env->ReleaseStringUTFChars(s, c);
  return out;
}

inline uint8_t clamp8(int v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }

}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_facefusion_mobile_NativePipe_lastError(JNIEnv* env, jclass) {
  return env->NewStringUTF(g_err.c_str());
}

JNIEXPORT jboolean JNICALL
Java_com_facefusion_mobile_NativePipe_initEx(JNIEnv* env, jclass, jstring jLib, jstring jSkel,
                                             jstring jModels, jstring jSwapper,
                                             jfloat weight, jfloat maskBlur,
                                             jintArray jPadding, jfloat detScore,
                                             jfloat lmkScore, jint pixelBoost,
                                             jboolean largestOnly,
                                             jboolean faceEnhance, jfloat enhanceBlend) {
  g_pipe.reset(new ffpipe::Pipeline());
  ffpipe::Config cfg;
  std::string swapper = jstr(env, jSwapper);
  if (swapper == "inswapper") {
    cfg.swapSize = 128; cfg.swapMean = 0.f; cfg.swapStd = 1.f;
    cfg.swapDenorm = false; cfg.swapperIsHyperswap = false;
  }
  // Clamped here rather than trusted from Kotlin: these come from sliders, and a bad
  // pixelBoost would allocate a crop of pixelBoost^2 the area and run that many graph
  // invocations per face.
  cfg.swapperWeight = std::fmin(1.f, std::fmax(0.f, weight));
  cfg.maskBlur = std::fmin(1.f, std::fmax(0.f, maskBlur));
  cfg.detectorScore = std::fmin(1.f, std::fmax(0.f, detScore));
  cfg.landmarkerScore = std::fmin(1.f, std::fmax(0.f, lmkScore));
  cfg.pixelBoost = pixelBoost < 1 ? 1 : (pixelBoost > 4 ? 4 : pixelBoost);
  cfg.swapLargestOnly = largestOnly == JNI_TRUE;
  // Asking for the enhancer is not the same as having it: hasEnhancer() decides, and
  // the stage is skipped silently when gpen_<tier>.bin was not there. A stale saved
  // preference from a build that had the model must not become a failed run.
  cfg.faceEnhance = faceEnhance == JNI_TRUE;
  cfg.faceEnhancerBlend = std::fmin(1.f, std::fmax(0.f, enhanceBlend));
  if (jPadding && env->GetArrayLength(jPadding) == 4) {
    jint pad[4];
    env->GetIntArrayRegion(jPadding, 0, 4, pad);
    for (int i = 0; i < 4; ++i)
      cfg.maskPadding[i] = pad[i] < 0 ? 0 : (pad[i] > 100 ? 100 : (int)pad[i]);
  }

  if (!g_pipe->init(jstr(env, jLib), jstr(env, jSkel), jstr(env, jModels), swapper, cfg)) {
    g_err = g_pipe->error();
    g_pipe.reset();
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_facefusion_mobile_NativePipe_release(JNIEnv*, jclass) { g_pipe.reset(); }

// The content gate on one BGR frame.  Returns upstream's decision statistic,
// `logit[0] - logit[1]`, or NaN if the graph did not run.
//
// A raw score rather than a boolean: the threshold is a policy constant that belongs with
// the policy, and the caller needs the number to log how much margin there was.  NaN for
// failure because there is no in-band float that could be mistaken for a real score --
// returning `false` on error would silently ALLOW everything the moment the gate broke.
JNIEXPORT jfloat JNICALL
Java_com_facefusion_mobile_NativePipe_contentScore(JNIEnv* env, jclass, jbyteArray jBgr,
                                                   jint w, jint h) {
  if (!g_pipe) { g_err = "pipeline not initialised"; return NAN; }
  ffcv::Image img(w, h, 3);
  if ((size_t)env->GetArrayLength(jBgr) != img.data.size()) {
    g_err = "contentScore: frame is not w*h*3 bytes";
    return NAN;
  }
  env->GetByteArrayRegion(jBgr, 0, (jsize)img.data.size(), (jbyte*)img.data.data());
  ffpipe::ContentVerdict v = g_pipe->checkContent(img);
  if (!v.ok) { g_err = g_pipe->error(); return NAN; }
  return v.score;
}

JNIEXPORT jboolean JNICALL
Java_com_facefusion_mobile_NativePipe_contentGateIsQuantised(JNIEnv*, jclass) {
  return (g_pipe && g_pipe->contentGateIsQuantised()) ? JNI_TRUE : JNI_FALSE;
}

// Whether gpen_<tier>.bin was present at init. Drives whether the switch is OFFERED, so it
// is only meaningful after a successful init -- before that it is false, which is the safe
// direction: the UI hides a control rather than showing one that cannot work.
JNIEXPORT jboolean JNICALL
Java_com_facefusion_mobile_NativePipe_hasEnhancer(JNIEnv*, jclass) {
  return (g_pipe && g_pipe->hasEnhancer()) ? JNI_TRUE : JNI_FALSE;
}

// Which context-binary tier this chip needs. Callable BEFORE any model exists, which is
// the point: the app has to know which files it is looking for before it can complain
// that they are missing.
JNIEXPORT jstring JNICALL
Java_com_facefusion_mobile_NativePipe_probeTier(JNIEnv* env, jclass, jstring jLib,
                                                jstring jSkel) {
  std::string lib = jstr(env, jLib);
  if (!ffqnn::init(lib + "/libQnnHtp.so", lib + "/libQnnSystem.so", jstr(env, jSkel))) {
    g_err = ffqnn::lastError();
    // Not an exception: pickTier's own fallback is the right answer when the backend
    // cannot come up, and the caller still gets a usable suffix.
    return env->NewStringUTF(ffqnn::pickTier(ffqnn::DeviceInfo{}).c_str());
  }
  return env->NewStringUTF(ffqnn::pickTier(ffqnn::deviceInfo()).c_str());
}

// Every tier this chip can load, best first, comma-joined: "v81,v73,v68".
//
// The DOWNLOADER needs this, not just probeTier. pickTier names the tier the hardware
// deserves, which the hosted manifest may not carry yet -- asking for a tier that is not
// published is an error, and on a brand-new arch it would be the error every user of that
// chip hits. Handing Kotlin the whole chain keeps the rule in one place: C++ decides what
// is loadable, the downloader decides what is available.
JNIEXPORT jstring JNICALL
Java_com_facefusion_mobile_NativePipe_probeTierChain(JNIEnv* env, jclass, jstring jLib,
                                                     jstring jSkel) {
  std::string lib = jstr(env, jLib);
  ffqnn::DeviceInfo d{};
  if (!ffqnn::init(lib + "/libQnnHtp.so", lib + "/libQnnSystem.so", jstr(env, jSkel)))
    g_err = ffqnn::lastError();   // leave d unmeasured; the chain's own fallback applies
  else
    d = ffqnn::deviceInfo();
  std::string out;
  for (const std::string& t : ffqnn::tierChain(d)) {
    if (!out.empty()) out += ",";
    out += t;
  }
  return env->NewStringUTF(out.c_str());
}

// "yes" | "no" | "unknown".  A String rather than a tri-state enum because "unknown" has
// to be impossible to confuse with "no" at the call site -- a boolean here would make the
// control's whole purpose unrepresentable.
JNIEXPORT jstring JNICALL
Java_com_facefusion_mobile_NativePipe_probeFp16(JNIEnv* env, jclass, jstring jLib,
                                                jstring jSkel, jstring jCanaryDir) {
  std::string lib = jstr(env, jLib);
  if (!ffqnn::init(lib + "/libQnnHtp.so", lib + "/libQnnSystem.so", jstr(env, jSkel))) {
    g_err = ffqnn::lastError();
    return env->NewStringUTF("unknown");
  }
  switch (ffqnn::fp16Canary(jstr(env, jCanaryDir))) {
    case ffqnn::Fp16::Yes: return env->NewStringUTF("yes");
    case ffqnn::Fp16::No:  return env->NewStringUTF("no");
    default:
      g_err = ffqnn::lastError();
      return env->NewStringUTF("unknown");
  }
}

// What the HTP actually reports, as `key=value;` pairs.
//
// A string rather than a struct because there is no cheap way to hand a struct across JNI
// and this is read once, for a settings screen. `ok=0` means the probe FAILED and every
// other field is meaningless -- it does not mean the chip is old, which is the same
// distinction pickTier and the fp16 canary both have to make.
JNIEXPORT jstring JNICALL
Java_com_facefusion_mobile_NativePipe_probeDeviceInfo(JNIEnv* env, jclass, jstring jLib,
                                                       jstring jSkel) {
  std::string lib = jstr(env, jLib);
  if (!ffqnn::init(lib + "/libQnnHtp.so", lib + "/libQnnSystem.so", jstr(env, jSkel))) {
    g_err = ffqnn::lastError();
    return env->NewStringUTF("ok=0");
  }
  ffqnn::DeviceInfo d = ffqnn::deviceInfo();
  if (!d.ok) {
    g_err = ffqnn::lastError();
    return env->NewStringUTF("ok=0");
  }
  std::string s = "ok=1";
  s += ";arch=" + std::to_string(d.arch);
  s += ";vtcm=" + std::to_string((unsigned long long)d.vtcmMb);
  s += ";soc=" + std::to_string((unsigned long)d.socModel);
  s += ";signedPd=" + std::to_string(d.signedPd ? 1 : 0);
  s += ";dlbc=" + std::to_string(d.dlbc ? 1 : 0);
  s += ";tier=" + ffqnn::pickTier(d);
  return env->NewStringUTF(s.c_str());
}

// Which RUNTIME will this device use, asked before anything is downloaded.
//
// "qnn" or "ncnn". The answer decides which MODEL SET to fetch -- a Hexagon part wants
// context binaries, everything else wants the ncnn pair -- so the download screen has to
// know it before a single file exists.
//
// ⚠ It has to be asked by TRYING, not by probing. `QnnDevice_getPlatformInfo` needs QNN
// already dlopen'd with the skels on ADSP_LIBRARY_PATH, so "probe, then choose" reports
// no-NPU on every device (ffnn.cpp carries the same warning; an earlier draft shipped it
// and failed on the bench it was written on).
JNIEXPORT jstring JNICALL
Java_com_facefusion_mobile_NativePipe_probeBackend(JNIEnv* env, jclass, jstring jLib,
                                                   jstring jSkel) {
  std::string lib = jstr(env, jLib);
  ffnn::InitSpec spec;
  spec.libDir = lib;
  spec.skelDir = jstr(env, jSkel);
  spec.modelDir = lib;   // unused by init; models are opened later
  if (!ffnn::init(ffnn::Backend::Auto, spec)) {
    g_err = ffnn::lastError();
    return env->NewStringUTF("none");
  }
  return env->NewStringUTF(ffnn::active() == ffnn::Backend::Qnn ? "qnn" : "ncnn");
}

JNIEXPORT jboolean JNICALL
Java_com_facefusion_mobile_NativePipe_setSource(JNIEnv* env, jclass, jbyteArray jBgr,
                                                jint w, jint h) {
  if (!g_pipe) { g_err = "pipeline not initialised"; return JNI_FALSE; }
  ffcv::Image img(w, h, 3);
  env->GetByteArrayRegion(jBgr, 0, (jsize)img.data.size(), (jbyte*)img.data.data());
  if (!g_pipe->setSource(img)) { g_err = g_pipe->error(); return JNI_FALSE; }
  return JNI_TRUE;
}

/** Swap every face in a BGR frame, in place.  Returns the face count, or -1 on error. */
JNIEXPORT jint JNICALL
Java_com_facefusion_mobile_NativePipe_processFrame(JNIEnv* env, jclass, jbyteArray jBgr,
                                                   jint w, jint h) {
  if (!g_pipe) { g_err = "pipeline not initialised"; return -1; }
  ffcv::Image img(w, h, 3);
  env->GetByteArrayRegion(jBgr, 0, (jsize)img.data.size(), (jbyte*)img.data.data());
  auto faces = g_pipe->analyse(img);
  if (!faces.empty() && !g_pipe->swapAll(img, faces)) { g_err = g_pipe->error(); return -1; }
  if (!faces.empty())
    env->SetByteArrayRegion(jBgr, 0, (jsize)img.data.size(), (const jbyte*)img.data.data());
  return (jint)faces.size();
}

/** Bitmap ARGB_8888 ints -> packed BGR bytes. */
JNIEXPORT jbyteArray JNICALL
Java_com_facefusion_mobile_NativePipe_argbToBgr(JNIEnv* env, jclass, jintArray jArgb,
                                                jint w, jint h) {
  jsize n = (jsize)w * h;
  std::vector<int32_t> px((size_t)n);
  env->GetIntArrayRegion(jArgb, 0, n, (jint*)px.data());
  jbyteArray out = env->NewByteArray(n * 3);
  std::vector<uint8_t> bgr((size_t)n * 3);
  for (jsize i = 0; i < n; ++i) {
    uint32_t v = (uint32_t)px[i];
    bgr[i * 3 + 0] = (uint8_t)(v & 0xFF);           // B
    bgr[i * 3 + 1] = (uint8_t)((v >> 8) & 0xFF);    // G
    bgr[i * 3 + 2] = (uint8_t)((v >> 16) & 0xFF);   // R
  }
  env->SetByteArrayRegion(out, 0, n * 3, (const jbyte*)bgr.data());
  return out;
}

/**
 * Rotate a packed BGR frame by 0/90/180/270 degrees clockwise.
 *
 * A portrait video is stored as LANDSCAPE frames plus a rotation flag in the container.
 * MediaMetadataRetriever applies that flag, which is why the preview looks upright, but
 * MediaCodec does not -- so the swap path was detecting faces on their side and, when it
 * found none, producing a sideways video with no flag set either.
 *
 * Native because it is 2.95 M pixels per frame at 720p: the same reason yuvToBgr is here.
 * A pure index remap, so it is exact -- no resampling and nothing to verify numerically.
 * 90 and 270 SWAP the dimensions; the caller must size everything downstream to match.
 */
JNIEXPORT jbyteArray JNICALL
Java_com_facefusion_mobile_NativePipe_rotateBgr(JNIEnv* env, jclass, jbyteArray jBgr,
                                                jint w, jint h, jint degrees) {
  const jsize n = (jsize)w * h * 3;
  if (env->GetArrayLength(jBgr) != n) {
    g_err = "rotateBgr: buffer is not w*h*3 bytes";
    return nullptr;
  }
  std::vector<uint8_t> src((size_t)n);
  env->GetByteArrayRegion(jBgr, 0, n, (jbyte*)src.data());

  int deg = ((degrees % 360) + 360) % 360;
  if (deg == 0) {
    jbyteArray out = env->NewByteArray(n);
    env->SetByteArrayRegion(out, 0, n, (const jbyte*)src.data());
    return out;
  }

  const int dw = (deg == 180) ? w : h;      // destination width
  const int dh = (deg == 180) ? h : w;      // destination height
  std::vector<uint8_t> dst((size_t)n);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int dx, dy;
      switch (deg) {
        case 90:  dx = dw - 1 - y; dy = x;             break;   // clockwise
        case 180: dx = w - 1 - x;  dy = h - 1 - y;     break;
        default:  dx = y;          dy = dh - 1 - x;    break;   // 270
      }
      const uint8_t* sp = &src[((size_t)y * w + x) * 3];
      uint8_t* dp = &dst[((size_t)dy * dw + dx) * 3];
      dp[0] = sp[0];
      dp[1] = sp[1];
      dp[2] = sp[2];
    }
  }

  jbyteArray out = env->NewByteArray(n);
  env->SetByteArrayRegion(out, 0, n, (const jbyte*)dst.data());
  return out;
}

/**
 * YUV_420_888 planes -> packed BGR.  MediaCodec hands back arbitrary row/pixel strides
 * and semi-planar (NV12/NV21) chroma is expressed as pixelStride 2, so both strides have
 * to be honoured -- assuming tightly packed I420 gives a green-and-magenta image.
 */
JNIEXPORT jbyteArray JNICALL
Java_com_facefusion_mobile_NativePipe_yuvToBgr(JNIEnv* env, jclass,
                                               jbyteArray jY, jint yRow,
                                               jbyteArray jU, jint uRow, jint uPix,
                                               jbyteArray jV, jint vRow, jint vPix,
                                               jint w, jint h) {
  std::vector<uint8_t> Y((size_t)env->GetArrayLength(jY));
  std::vector<uint8_t> U((size_t)env->GetArrayLength(jU));
  std::vector<uint8_t> V((size_t)env->GetArrayLength(jV));
  env->GetByteArrayRegion(jY, 0, (jsize)Y.size(), (jbyte*)Y.data());
  env->GetByteArrayRegion(jU, 0, (jsize)U.size(), (jbyte*)U.data());
  env->GetByteArrayRegion(jV, 0, (jsize)V.size(), (jbyte*)V.data());

  std::vector<uint8_t> bgr((size_t)w * h * 3);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      size_t yi = (size_t)y * yRow + x;
      size_t ci = (size_t)(y / 2) * uRow + (size_t)(x / 2) * uPix;
      size_t vi = (size_t)(y / 2) * vRow + (size_t)(x / 2) * vPix;
      int Yv = (yi < Y.size() ? Y[yi] : 16) - 16;
      int Uv = (ci < U.size() ? U[ci] : 128) - 128;
      int Vv = (vi < V.size() ? V[vi] : 128) - 128;
      int c = 298 * Yv;
      uint8_t* p = &bgr[((size_t)y * w + x) * 3];
      p[0] = clamp8((c + 516 * Uv + 128) >> 8);              // B
      p[1] = clamp8((c - 100 * Uv - 208 * Vv + 128) >> 8);   // G
      p[2] = clamp8((c + 409 * Vv + 128) >> 8);              // R
    }
  }
  jbyteArray out = env->NewByteArray((jsize)bgr.size());
  env->SetByteArrayRegion(out, 0, (jsize)bgr.size(), (const jbyte*)bgr.data());
  return out;
}

/**
 * Packed BGR -> ARGB_8888 ints, box-downsampled to dstW x dstH for the live preview.
 *
 * Downsampling here rather than with Bitmap.createScaledBitmap avoids allocating a
 * full-resolution Bitmap per previewed frame -- 3.9 MB at 720p, every frame, purely to
 * throw most of it away.
 */
JNIEXPORT jintArray JNICALL
Java_com_facefusion_mobile_NativePipe_bgrToArgb(JNIEnv* env, jclass, jbyteArray jBgr,
                                                jint w, jint h, jint dstW, jint dstH) {
  std::vector<uint8_t> bgr((size_t)w * h * 3);
  env->GetByteArrayRegion(jBgr, 0, (jsize)bgr.size(), (jbyte*)bgr.data());
  if (dstW <= 0 || dstH <= 0) { dstW = w; dstH = h; }

  std::vector<int32_t> out((size_t)dstW * dstH);
  for (int y = 0; y < dstH; ++y) {
    int sy0 = (int)((int64_t)y * h / dstH), sy1 = (int)((int64_t)(y + 1) * h / dstH);
    if (sy1 <= sy0) sy1 = sy0 + 1;
    for (int x = 0; x < dstW; ++x) {
      int sx0 = (int)((int64_t)x * w / dstW), sx1 = (int)((int64_t)(x + 1) * w / dstW);
      if (sx1 <= sx0) sx1 = sx0 + 1;
      uint32_t B = 0, G = 0, R = 0, n = 0;
      for (int sy = sy0; sy < sy1 && sy < h; ++sy)
        for (int sx = sx0; sx < sx1 && sx < w; ++sx) {
          const uint8_t* p = &bgr[((size_t)sy * w + sx) * 3];
          B += p[0]; G += p[1]; R += p[2]; ++n;
        }
      if (!n) n = 1;
      out[(size_t)y * dstW + x] =
          (int32_t)(0xFF000000u | ((R / n) << 16) | ((G / n) << 8) | (B / n));
    }
  }
  jintArray ja = env->NewIntArray((jsize)out.size());
  env->SetIntArrayRegion(ja, 0, (jsize)out.size(), (const jint*)out.data());
  return ja;
}

/**
 * Packed BGR -> the encoder's OWN input planes, honouring its strides.
 *
 * COLOR_FormatYUV420Flexible does not mean I420.  On this device the AVC encoder is
 * semi-planar (NV12): chroma is interleaved in one plane with pixelStride 2.  Writing
 * planar I420 into it puts luma in the right place and chroma in the wrong one, which
 * renders as a greyscale image with green and pink blobs -- luma is fine, so it looks
 * "nearly working", which is the misleading part.
 *
 * Taking the planes from MediaCodec.getInputImage() and respecting rowStride/pixelStride
 * is correct for planar and semi-planar alike, so the layout never has to be guessed.
 * Buffers must be direct (MediaCodec's are).
 */
JNIEXPORT jboolean JNICALL
Java_com_facefusion_mobile_NativePipe_bgrToImagePlanes(
    JNIEnv* env, jclass, jbyteArray jBgr, jint w, jint h,
    jobject yBuf, jint yRow, jint yPix,
    jobject uBuf, jint uRow, jint uPix,
    jobject vBuf, jint vRow, jint vPix) {

  auto* Y = (uint8_t*)env->GetDirectBufferAddress(yBuf);
  auto* U = (uint8_t*)env->GetDirectBufferAddress(uBuf);
  auto* V = (uint8_t*)env->GetDirectBufferAddress(vBuf);
  if (!Y || !U || !V) return JNI_FALSE;
  jlong yCap = env->GetDirectBufferCapacity(yBuf);
  jlong uCap = env->GetDirectBufferCapacity(uBuf);
  jlong vCap = env->GetDirectBufferCapacity(vBuf);

  std::vector<uint8_t> bgr((size_t)w * h * 3);
  env->GetByteArrayRegion(jBgr, 0, (jsize)bgr.size(), (jbyte*)bgr.data());

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const uint8_t* p = &bgr[((size_t)y * w + x) * 3];
      int B = p[0], G = p[1], R = p[2];
      jlong off = (jlong)y * yRow + (jlong)x * yPix;
      if (off >= 0 && off < yCap)
        Y[off] = clamp8(((66 * R + 129 * G + 25 * B + 128) >> 8) + 16);
    }
  }
  for (int y = 0; y < h; y += 2) {
    for (int x = 0; x < w; x += 2) {
      // average the 2x2 block; point-sampling shows as chroma crawl on the swapped edge
      int R = 0, G = 0, B = 0, n = 0;
      for (int dy = 0; dy < 2 && y + dy < h; ++dy)
        for (int dx = 0; dx < 2 && x + dx < w; ++dx) {
          const uint8_t* p = &bgr[(((size_t)y + dy) * w + x + dx) * 3];
          B += p[0]; G += p[1]; R += p[2]; ++n;
        }
      R /= n; G /= n; B /= n;
      jlong uo = (jlong)(y / 2) * uRow + (jlong)(x / 2) * uPix;
      jlong vo = (jlong)(y / 2) * vRow + (jlong)(x / 2) * vPix;
      if (uo >= 0 && uo < uCap)
        U[uo] = clamp8(((-38 * R - 74 * G + 112 * B + 128) >> 8) + 128);
      if (vo >= 0 && vo < vCap)
        V[vo] = clamp8(((112 * R - 94 * G - 18 * B + 128) >> 8) + 128);
    }
  }
  return JNI_TRUE;
}

/** Packed BGR -> I420 (planar).  Kept for reference; the encoder path uses the planes. */
JNIEXPORT jbyteArray JNICALL
Java_com_facefusion_mobile_NativePipe_bgrToI420(JNIEnv* env, jclass, jbyteArray jBgr,
                                                jint w, jint h) {
  std::vector<uint8_t> bgr((size_t)w * h * 3);
  env->GetByteArrayRegion(jBgr, 0, (jsize)bgr.size(), (jbyte*)bgr.data());
  size_t ySize = (size_t)w * h, cSize = ySize / 4;
  std::vector<uint8_t> out(ySize + 2 * cSize);
  uint8_t* Y = out.data();
  uint8_t* U = Y + ySize;
  uint8_t* V = U + cSize;

  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const uint8_t* p = &bgr[((size_t)y * w + x) * 3];
      int B = p[0], G = p[1], R = p[2];
      Y[(size_t)y * w + x] = clamp8(((66 * R + 129 * G + 25 * B + 128) >> 8) + 16);
    }
  for (int y = 0; y < h; y += 2)
    for (int x = 0; x < w; x += 2) {
      // average the 2x2 block rather than point-sampling: point sampling shows as
      // chroma crawl on the swapped edge
      int R = 0, G = 0, B = 0, n = 0;
      for (int dy = 0; dy < 2 && y + dy < h; ++dy)
        for (int dx = 0; dx < 2 && x + dx < w; ++dx) {
          const uint8_t* p = &bgr[(((size_t)y + dy) * w + x + dx) * 3];
          B += p[0]; G += p[1]; R += p[2]; ++n;
        }
      R /= n; G /= n; B /= n;
      size_t ci = (size_t)(y / 2) * (w / 2) + (x / 2);
      U[ci] = clamp8(((-38 * R - 74 * G + 112 * B + 128) >> 8) + 128);
      V[ci] = clamp8(((112 * R - 94 * G - 18 * B + 128) >> 8) + 128);
    }
  jbyteArray ja = env->NewByteArray((jsize)out.size());
  env->SetByteArrayRegion(ja, 0, (jsize)out.size(), (const jbyte*)out.data());
  return ja;
}

}  // extern "C"
