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
#include <cstdlib>   // setenv/unsetenv, for setForcedBackend
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
std::string g_rejectedTier;
std::vector<std::string> g_skipTiers;

std::string jstr(JNIEnv* env, jstring s) {
  if (!s) return {};
  const char* c = env->GetStringUTFChars(s, nullptr);
  std::string out(c ? c : "");
  env->ReleaseStringUTFChars(s, c);
  return out;
}

inline uint8_t clamp8(int v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }

/**
 * The per-frame tunables, clamped, into `cfg`.
 *
 * Shared by initEx and setOptions BECAUSE it is the clamping: these come from sliders, and
 * a bad pixelBoost allocates a crop of pixelBoost^2 the area and runs that many graph
 * invocations per face. Two copies of that would be one copy away from a path that trusts
 * Kotlin, and the live-update path is exactly the one that could be reached most often.
 *
 * Touches nothing derived from WHICH SWAPPER is loaded -- swapSize and the normalisation
 * belong to the model, not to the sliders.
 */
void tunables(JNIEnv* env, ffpipe::Config& cfg, jfloat weight, jfloat maskBlur,
              jintArray jPadding, jfloat detScore, jfloat lmkScore, jint pixelBoost,
              jboolean largestOnly, jboolean faceEnhance, jfloat enhanceBlend) {
  cfg.swapperWeight = std::fmin(1.f, std::fmax(0.f, weight));
  cfg.maskBlur = std::fmin(1.f, std::fmax(0.f, maskBlur));
  cfg.detectorScore = std::fmin(1.f, std::fmax(0.f, detScore));
  cfg.landmarkerScore = std::fmin(1.f, std::fmax(0.f, lmkScore));
  cfg.pixelBoost = pixelBoost < 1 ? 1 : (pixelBoost > 4 ? 4 : pixelBoost);
  cfg.swapLargestOnly = largestOnly == JNI_TRUE;
  // Asking for the enhancer is not the same as having it: hasEnhancer() decides, and the
  // stage is skipped silently when gpen_<tier>.bin was not there. A stale saved preference
  // from a build that had the model must not become a failed run.
  cfg.faceEnhance = faceEnhance == JNI_TRUE;
  cfg.faceEnhancerBlend = std::fmin(1.f, std::fmax(0.f, enhanceBlend));
  if (jPadding && env->GetArrayLength(jPadding) == 4) {
    jint pad[4];
    env->GetIntArrayRegion(jPadding, 0, 4, pad);
    for (int i = 0; i < 4; ++i)
      cfg.maskPadding[i] = pad[i] < 0 ? 0 : (pad[i] > 100 ? 100 : (int)pad[i]);
  }
}

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
  tunables(env, cfg, weight, maskBlur, jPadding, detScore, lmkScore, pixelBoost,
           largestOnly, faceEnhance, enhanceBlend);

  // PUSHED, not passed. There are four paths into init -- the preview, runSwap, the
  // self-test and the API -- and a per-call-site argument is a list you can be absent
  // from: the fifth one added later would compile, run, and quietly reload the tier this
  // device has already proved it cannot execute. Same reasoning the content gate is
  // written down with, for the same reason.
  cfg.skipVariants = g_skipTiers;

  bool ok = g_pipe->init(jstr(env, jLib), jstr(env, jSkel), jstr(env, jModels), swapper, cfg);
  // Read BEFORE the reset, and on the success path too. A tier can be rejected and the
  // next one work -- that device still wants the rejection remembered, or it pays the
  // failed load again on every launch for the life of the install.
  g_rejectedTier = g_pipe->rejectedVariant();
  if (!ok) {
    g_err = g_pipe->error();
    g_pipe.reset();
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

/**
 * Change the per-frame tunables WITHOUT reloading anything.
 *
 * Returns false only when nothing is loaded, which is not an error -- the caller then does
 * a normal init and these values go in through it.
 *
 * ⚠ The SWAPPER is not a parameter here, deliberately. It selects a different model file
 * and different input geometry, so it is the one option that still costs a reload; leaving
 * it out means this entry point cannot be used to ask for one by accident.
 */
JNIEXPORT jboolean JNICALL
Java_com_facefusion_mobile_NativePipe_setOptionsEx(JNIEnv* env, jclass,
                                                   jfloat weight, jfloat maskBlur,
                                                   jintArray jPadding, jfloat detScore,
                                                   jfloat lmkScore, jint pixelBoost,
                                                   jboolean largestOnly,
                                                   jboolean faceEnhance,
                                                   jfloat enhanceBlend) {
  if (!g_pipe) return JNI_FALSE;
  ffpipe::Config cfg;
  tunables(env, cfg, weight, maskBlur, jPadding, detScore, lmkScore, pixelBoost,
           largestOnly, faceEnhance, enhanceBlend);
  g_pipe->updateConfig(cfg);
  return JNI_TRUE;
}

// The tier that loaded and would not execute, or "". Survives the failed init that
// produced it, which is the only reason it is a global here rather than read off g_pipe.
JNIEXPORT jstring JNICALL
Java_com_facefusion_mobile_NativePipe_rejectedTier(JNIEnv* env, jclass) {
  return env->NewStringUTF(g_rejectedTier.c_str());
}

// Comma-separated, because a JNI array of strings costs three more calls and this list is
// at most three short tokens long. Applied to every subsequent init; "" clears it.
JNIEXPORT void JNICALL
Java_com_facefusion_mobile_NativePipe_setSkipTiers(JNIEnv* env, jclass, jstring jTiers) {
  g_skipTiers.clear();
  std::string s = jstr(env, jTiers);
  size_t start = 0;
  while (start < s.size()) {
    size_t comma = s.find(',', start);
    size_t end = comma == std::string::npos ? s.size() : comma;
    std::string one = s.substr(start, end - start);
    if (!one.empty()) g_skipTiers.push_back(one);
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
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
  // Ask the SEAM first, because on a non-Qualcomm part the answer is not an arch tier at
  // all -- it is "ncnn", one variant, and the downloader must fetch a completely different
  // model set. Going straight to ffqnn here is what would hand a Mali phone a chain of
  // Hexagon context binaries it can never load: ffqnn::init fails, `d` stays unmeasured,
  // and tierChain's fallback confidently answers "v68".
  ffnn::InitSpec spec;
  spec.libDir = lib;
  spec.skelDir = jstr(env, jSkel);
  spec.modelDir = lib;   // unused by init
  if (ffnn::init(ffnn::Backend::Auto, spec) && ffnn::active() == ffnn::Backend::Ncnn) {
    std::string out;
    for (const std::string& v : ffnn::variantChain(ffnn::Backend::Ncnn)) {
      if (!out.empty()) out += ",";
      out += v;
    }
    return env->NewStringUTF(out.c_str());
  }
  // QNN, including the case where the backend did not come up at all. Deliberately NOT
  // routed through the seam: ffqnn::tierChain has its own fallback for an unmeasured
  // device, and that fallback is the right answer here -- a transient QNN init failure on
  // a Hexagon part must still produce a loadable chain, not an empty one.
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
  // The RUNTIME first, and OUTSIDE the ok=0 early returns below. `ok` describes the HTP
  // probe, and on a part with no Hexagon that probe is *supposed* to fail -- reporting only
  // "ok=0" there would leave the settings panel unable to say what the device is actually
  // running, which is the one thing a non-Qualcomm user needs it to say.
  std::string pre;
  {
    ffnn::InitSpec spec;
    spec.libDir = lib;
    spec.skelDir = jstr(env, jSkel);
    spec.modelDir = lib;
    if (ffnn::init(ffnn::Backend::Auto, spec)) {
      const bool ncnn = ffnn::active() == ffnn::Backend::Ncnn;
      pre = std::string(";backend=") + (ncnn ? "ncnn" : "qnn");
      if (ncnn) pre += ffnn::deviceInfo(ffnn::Backend::Ncnn).gpu ? ";gpu=1" : ";gpu=0";
    } else {
      pre = ";backend=none";
    }
  }
  if (!ffqnn::init(lib + "/libQnnHtp.so", lib + "/libQnnSystem.so", jstr(env, jSkel))) {
    g_err = ffqnn::lastError();
    return env->NewStringUTF(("ok=0" + pre).c_str());
  }
  ffqnn::DeviceInfo d = ffqnn::deviceInfo();
  if (!d.ok) {
    g_err = ffqnn::lastError();
    return env->NewStringUTF(("ok=0" + pre).c_str());
  }
  std::string s = "ok=1" + pre;
  s += ";arch=" + std::to_string(d.arch);
  s += ";vtcm=" + std::to_string((unsigned long long)d.vtcmMb);
  s += ";soc=" + std::to_string((unsigned long)d.socModel);
  s += ";signedPd=" + std::to_string(d.signedPd ? 1 : 0);
  s += ";dlbc=" + std::to_string(d.dlbc ? 1 : 0);
  s += ";tier=" + ffqnn::pickTier(d);
  return env->NewStringUTF(s.c_str());
}

// Was the ncnn backend LINKED into this build?
//
// Not "is there a GPU" and not "which backend is active": whether the code exists at all.
// FF_NCNN is off unless work/android/ncnn/ was staged, so a build can ship with no second
// runtime -- and a settings control that offers to switch to a backend that is not in the
// binary is a control that silently does nothing.
JNIEXPORT jboolean JNICALL
Java_com_facefusion_mobile_NativePipe_hasNcnnBackend(JNIEnv*, jclass) {
#ifdef FFNN_HAVE_NCNN
  return JNI_TRUE;
#else
  return JNI_FALSE;
#endif
}

// Force a backend for the rest of the process, or "" to go back to Auto.
//
// The non-Qualcomm path is otherwise UNTESTABLE on any bench that has a Hexagon: Auto tries
// QNN first and QNN wins, so ncnn could only ever be exercised on a phone this project does
// not own. `ffnn::init(Auto)` already honours FFBACKEND for the headless CLI, where the
// environment is the shell's; an Android app has no environment to set from outside, so it
// sets its own. One mechanism, and the CLI's is the one already documented.
//
// ⚠ The CALLER must release the pipeline BEFORE calling this. Handles are tagged with the
// backend that opened them (ffnn.cpp), so a stale handle is freed correctly either way --
// but a pipeline half-built on the other runtime is still not a thing to keep.
JNIEXPORT void JNICALL
Java_com_facefusion_mobile_NativePipe_setForcedBackend(JNIEnv* env, jclass, jstring jName) {
  std::string name = jstr(env, jName);
  if (name.empty()) unsetenv("FFBACKEND");
  else setenv("FFBACKEND", name.c_str(), 1);
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

// Whether wav2lip_<tier>.bin was present at init. Same contract as hasEnhancer: it
// decides whether the switch is OFFERED, and false before init hides a control rather
// than showing one that cannot work.
JNIEXPORT jboolean JNICALL
Java_com_facefusion_mobile_NativePipe_hasLipSyncer(JNIEnv*, jclass) {
  return (g_pipe && g_pipe->hasLipSyncer()) ? JNI_TRUE : JNI_FALSE;
}

/**
 * Hand the clip's decoded PCM over once, before the frame loop.
 *
 * `fps` is the OUTPUT frame rate, not the source's: window k belongs to output frame k,
 * and a rate-reduced run writes fewer frames than it decodes.
 */
JNIEXPORT jboolean JNICALL
Java_com_facefusion_mobile_NativePipe_setAudio(JNIEnv* env, jclass, jshortArray jPcm,
                                               jint channels, jint sampleRate, jdouble fps) {
  if (!g_pipe) { g_err = "pipeline not initialised"; return JNI_FALSE; }
  const jsize n = env->GetArrayLength(jPcm);
  if (n <= 0 || channels <= 0) { g_err = "no audio samples"; return JNI_FALSE; }
  std::vector<int16_t> pcm((size_t)n);
  env->GetShortArrayRegion(jPcm, 0, n, (jshort*)pcm.data());
  const size_t frames = (size_t)n / (size_t)channels;
  if (!g_pipe->setAudio(pcm.data(), frames, channels, sampleRate, fps)) {
    g_err = g_pipe->error().empty() ? "could not prepare the audio" : g_pipe->error();
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

// How many mel windows setAudio produced, for a caller that wants to report coverage.
JNIEXPORT jint JNICALL
Java_com_facefusion_mobile_NativePipe_melWindowTotal(JNIEnv*, jclass) {
  return g_pipe ? (jint)g_pipe->melWindowTotal() : 0;
}

/**
 * Swap and then LIP SYNC one BGR frame, in place. Returns the face count, or -1.
 *
 * Separate from processFrame only because of the index: the lip syncer is the first stage
 * here whose input depends on WHICH frame this is. A negative index, no lip syncer on the
 * device, or audio that was never set all fall back to a plain swap, so a caller may use
 * this unconditionally.
 */
JNIEXPORT jint JNICALL
Java_com_facefusion_mobile_NativePipe_processFrameAt(JNIEnv* env, jclass, jbyteArray jBgr,
                                                     jint w, jint h, jint frameIndex) {
  if (!g_pipe) { g_err = "pipeline not initialised"; return -1; }
  ffcv::Image img(w, h, 3);
  env->GetByteArrayRegion(jBgr, 0, (jsize)img.data.size(), (jbyte*)img.data.data());
  auto faces = g_pipe->analyse(img);
  if (!faces.empty()) {
    if (!g_pipe->swapAll(img, faces)) { g_err = g_pipe->error(); return -1; }
    if (frameIndex >= 0 && g_pipe->hasLipSyncer() && g_pipe->melWindowTotal() > 0) {
      if (!g_pipe->syncLip(img, faces, g_pipe->melWindow(frameIndex))) {
        g_err = g_pipe->error();
        return -1;
      }
    }
    env->SetByteArrayRegion(jBgr, 0, (jsize)img.data.size(), (const jbyte*)img.data.data());
  }
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
