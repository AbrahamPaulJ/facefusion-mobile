#include "ffpipe.h"

#include "ffaudio.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>

#include "ffnn.h"

// FFDEBUG=1 prints what each stage actually produced.  Cheap, and the alternative is
// guessing at a tensor layout from the host side.
#include <cstdio>
#include <cstdlib>
static bool ffdebug() { static bool v = getenv("FFDEBUG") != nullptr; return v; }

// FFNOMASKCACHE=1 forces the box-mask memo to miss on every frame, so ONE binary can
// measure cached against uncached by alternating runs inside a single thermal session --
// which is the only way an A/B on this device means anything (running one build first
// every time biases the other by ~20%, larger than most effects worth measuring).
static bool ffNoMaskCache() { static bool v = getenv("FFNOMASKCACHE") != nullptr; return v; }

namespace ffpipe {
namespace {

double nowMs() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// fan_68_5 is a 0.9 MB 5->68 landmark MLP.  run_reference keeps it on the CPU because it
// is negligible; here it runs on the NPU alongside the rest simply because it is already
// a graph and that avoids porting its weights separately.
struct Nets {
  ffnn::Handle det = nullptr, fan = nullptr, fan685 = nullptr, arc = nullptr, swap = nullptr;
  ffnn::Handle lip = nullptr;   // the lip syncer, optional exactly like the enhancer
  // WHICH lip syncer. The two are not variants of one graph -- they take different crops
  // at different sizes and a different number of inputs -- so syncLip branches on this
  // rather than on a size alone.
  bool lipIsEdtalk = false;
  ffnn::Handle nsfw = nullptr;
  // Optional, like fan685: absent simply means the enhancer cannot be offered.
  ffnn::Handle enh = nullptr;
};

// One clip's audio, reduced to what the lip syncer eats: a mel window per OUTPUT frame.
// Held here rather than handed across JNI per frame, for the same reason the source
// embedding is -- it is set once and read every frame.
struct AudioState {
  std::vector<ffaudio::MelWindow> windows;
  // create_empty_audio_frame through prepare_audio_frame: zeros become a uniform -4,
  // which is what the model reads as silence.
  std::vector<float> silence;
};

// content_analyser.py:create_static_model_set -- nsfw_2 is 384x384, mean 0, std 1.
constexpr int kNsfwSize = 384;

// face_enhancer/core.py -- gpen_bfr_256 is 256x256 on the `arcface_128` template, the
// same template and size hyperswap_1a_256 declares.  NOT cfg.swapSize: inswapper is 128,
// and the enhancer graph does not change with the swapper.
constexpr int kEnhSize = 256;

/**
 * Run one loaded graph on synthetic input, purely to find out whether it runs.
 *
 * Shapes come from the MODEL, never from a table here: the app already knows every input
 * size in the code that does the real work, and a second copy in the prober would be wrong
 * the first time a conversion changed one -- silently, because a probe that feeds the
 * wrong size and fails looks exactly like a chip that cannot execute.
 *
 * 0.5 rather than 0: arcface L2-normalises, and a zero vector normalises to 0/0. Probing
 * with an input that can produce a legitimate NaN would make the prober's own verdict
 * indistinguishable from the fault it is looking for.
 *
 * ⚠ Only `execute` returning false is a failure. A non-finite output is REPORTED and
 * tolerated, because this runs on every device at every init and a probe that is stricter
 * than the pipeline could reject a tier that works -- turning a diagnostic into an outage
 * on hardware that was fine. The gate is the one exception, and ffpipe checks it
 * separately, because there a wrong number is the whole failure mode.
 */
bool probeExecutes(ffnn::Handle h, std::string* why) {
  std::vector<std::string> names = ffnn::inputNames(h);
  std::vector<std::vector<int>> shapes = ffnn::inputShapes(h);
  if (names.empty() || names.size() != shapes.size()) {
    *why = "no input metadata";
    return false;
  }
  std::vector<std::vector<float>> bufs(names.size());
  std::vector<const float*> ptrs;
  ptrs.reserve(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    size_t n = 1;
    for (int d : shapes[i]) n *= (size_t)(d > 0 ? d : 1);
    bufs[i].assign(n, 0.5f);
    ptrs.push_back(bufs[i].data());
  }
  std::vector<std::vector<float>> outs;
  if (!ffnn::execute(h, names, ptrs, outs)) {
    *why = ffnn::lastError();
    return false;
  }
  if (outs.empty() || outs[0].empty()) {
    *why = "executed but produced no output";
    return false;
  }
  for (const std::vector<float>& o : outs)
    for (float f : o)
      if (!std::isfinite(f)) { *why = "non-finite output"; return true; }
  return true;
}

}  // namespace

struct Pipeline::Impl {
  Nets n;
  AudioState audio;
  Config cfg;
  float srcEmbedding[512]{};
  float srcEmbeddingNorm[512]{};
  bool haveSource = false;
  std::vector<float> emap;   // inswapper only: the 512x512 initializer

  // A box mask depends only on (size, cfg.maskBlur, cfg.maskPadding) -- constant across
  // every frame and every face of a run -- but createBoxMask's gaussianBlur at the default
  // 0.3 blur is a 79-tap separable pass in double precision, so rebuilding it per frame
  // cost 43.93 ms on the lip syncer's 512 canvas and 10.4 M MACs on the swapper's 256 crop.
  //
  // Memoised on the PARAMETERS rather than merely computed once, so a live maskBlur,
  // maskPadding or pixelBoost change through updateConfig still takes effect on the very
  // next frame. `size` is in the key because the swap path's crop is swapSize*pixelBoost
  // and pixel boost is one of the sliders.
  struct BoxMaskCache {
    ffcv::MatF m;
    int size = -1;
    float blur = -1.f;
    int padding[4] = {-1, -1, -1, -1};

    const ffcv::MatF& get(int size_, float blur_, const int padding_[4]) {
      if (ffNoMaskCache() || size != size_ || blur != blur_ ||
          std::memcmp(padding, padding_, sizeof(padding)) != 0) {
        m = ffcv::createBoxMask(size_, size_, blur_, padding_);
        size = size_;
        blur = blur_;
        std::memcpy(padding, padding_, sizeof(padding));
      }
      return m;
    }
  };
  BoxMaskCache lipBoxMask;    // at LS, for syncLip
  BoxMaskCache swapBoxMask;   // at swapSize*pixelBoost, for swapAll
};

Pipeline::Pipeline() = default;

Pipeline::~Pipeline() {
  if (p_) {
    for (auto h : {p_->n.det, p_->n.fan, p_->n.fan685, p_->n.arc, p_->n.swap, p_->n.lip, p_->n.nsfw,
                   p_->n.enh})
      if (h) ffnn::release(h);
  }
}

bool Pipeline::init(const std::string& libDir, const std::string& skelDir,
                    const std::string& modelDir, const std::string& swapperName,
                    const Config& cfg) {
  p_.reset(new Impl());
  p_->cfg = cfg;

  // Which RUNTIME, before which model. Auto because the HTP cannot be probed before QNN is
  // started -- asking first and choosing after reports no-NPU on every device.
  ffnn::InitSpec spec;
  spec.libDir = libDir;
  spec.skelDir = skelDir;
  spec.modelDir = modelDir;
  if (!ffnn::init(ffnn::Backend::Auto, spec)) {
    err_ = std::string("backend init: ") + ffnn::lastError();
    return false;
  }
  // Which arch tier this chip needs, measured rather than assumed. Everything before
  // 2026-08-24 hard-coded _v79 here, which is exactly one phone; the chain falls back to
  // v68 whenever the probe cannot measure, so an unrecognised device still loads.
  //
  // Then resolve that against WHAT IS ACTUALLY ON DISK.  The best tier for a chip is not
  // always one we have published: a v81 part asks for v81 the moment the app knows the
  // arch exists, which is necessarily before the binaries are hosted.  Taking the first
  // tier whose detector is present turns that from a hard init failure into "keeps running
  // the tier it ran yesterday, upgrades itself the day the files land".
  //
  // The detector is the probe because it is mandatory and the smallest -- 3.8 MB. A tier
  // is never half-present: the downloader writes a `.part` and renames only after the
  // SHA256 matches, so yoloface existing means the rest of that tier does too.
  // Which arch tier this chip needs, measured rather than assumed, resolved against WHAT
  // IS ACTUALLY ON DISK. The best tier for a chip is not always one we have published: a
  // v81 part asks for v81 the moment the app knows the arch exists, which is necessarily
  // before the binaries are hosted.
  //
  // The detector is the probe because it is mandatory and the smallest -- 3.8 MB. A tier is
  // never half-present: the downloader writes a `.part` and renames only after the SHA256
  // matches, so yoloface existing means the rest of that tier does too.
  const std::vector<std::string> chain = ffnn::variantChain(ffnn::active());
  auto skipped = [&](const std::string& t) {
    return std::find(cfg.skipVariants.begin(), cfg.skipVariants.end(), t) !=
           cfg.skipVariants.end();
  };
  std::vector<std::string> candidates;
  for (const std::string& t : chain)
    if (ffnn::variantPresent(t) && !skipped(t)) candidates.push_back(t);
  // ⚠ The skip list loses to having nothing to run. It is a record of what failed HERE,
  // and a device that has rejected every tier it holds is better served by trying one and
  // reporting honestly than by refusing to start with "no tier" -- which would read, to
  // the user, as the app breaking itself over a note it wrote about their phone.
  if (candidates.empty())
    for (const std::string& t : chain)
      if (ffnn::variantPresent(t)) candidates.push_back(t);
  if (candidates.empty() && !chain.empty()) candidates.push_back(chain.front());
  if (candidates.empty()) { err_ = "no tier: the HTP probe returned nothing"; return false; }

  auto dropNets = [&]() {
    for (ffnn::Handle* h : {&p_->n.det, &p_->n.fan, &p_->n.fan685, &p_->n.arc, &p_->n.lip,
                            &p_->n.swap, &p_->n.nsfw, &p_->n.enh}) {
      if (*h) ffnn::release(*h);
      *h = nullptr;
    }
    nsfwQuantised_ = false;
  };

  // One tier's worth of loading, plus the proof that it actually RUNS.
  auto tryTier = [&](const std::string& t) -> bool {
    tier_ = t;
    ffnn::useVariant(t);
    // PLACEMENT is a property of the MODEL, measured 2026-08-30 and not a preference:
    // the content gate and the enhancer must not run on a GPU. The gate because ncnn's
    // Vulkan moves its decision statistic toward ALLOWING -- the one direction a gate must
    // not err -- and the enhancer because it fails outright on every GPU backend tried.
    // On QNN this is a no-op, since everything runs on the HTP there either way.
    auto open = [&](const char* name,
                    ffnn::Placement place = ffnn::Placement::Default) -> ffnn::Handle {
      ffnn::Handle h = ffnn::open(name, place);
      if (!h) err_ = std::string("open ") + name + " (" + tier_ + "): " + ffnn::lastError();
      return h;
    };
    p_->n.det = open("yoloface");             if (!p_->n.det) return false;
    p_->n.fan = open("fan2d");                if (!p_->n.fan) return false;
    p_->n.arc = open("arcface");              if (!p_->n.arc) return false;
    p_->n.swap = open(swapperName.c_str());   if (!p_->n.swap) return false;
    p_->n.fan685 = open("fan685");   // optional; absent -> geometric fallback

    // The face enhancer. Optional in the strongest sense: a separate download, most
    // installs will not have it, and a missing file must never be an init failure.
    p_->n.enh = open("gpen", ffnn::Placement::Cpu);

    // The lip syncer. Same rule as the enhancer: a separate download, absent on every
    // install that has not asked for it, and never an init failure when missing.
    //
    // edtalk FIRST, and wav2lip only if it is absent. Both are supported on purpose: an
    // install that already has wav2lip keeps working and is not silently downgraded to
    // "no lip syncer" by an app update, and the two can be compared on one device. edtalk
    // wins where both are present because it is the reason the other one is still here --
    // it draws the mouth at 256 where wav2lip draws it at 96, and the measured box is
    // 274x285, so wav2lip upscales it ~3x and edtalk does not.
    p_->n.lip = open("edtalk");
    p_->n.lipIsEdtalk = (p_->n.lip != nullptr);
    if (!p_->n.lip) p_->n.lip = open("wav2lip");

    // The content gate is MANDATORY, because it blocks. A gate that silently does not run
    // is worse than no gate: it reports "checked" to every caller above it.
    //
    // fp32 is the build that tracks the host, but it only finalizes on v79 -- below that
    // the GELU cannot be created in float, and v81 refuses it too (docs/traps.md #10), so
    // every other tier carries the quantised one.
    p_->n.nsfw = open("nsfw", ffnn::Placement::Cpu);
    if (!p_->n.nsfw) {
      // nsfwq2, not nsfwq: the quantised gate is calibrated for the input range this file
      // feeds it, and that range changed. See work/qnn/convert.sh.
      p_->n.nsfw = open("nsfwq2", ffnn::Placement::Cpu);
      nsfwQuantised_ = p_->n.nsfw != nullptr;
    }
    if (!p_->n.nsfw) {
      err_ = "no content gate: neither nsfw_" + tier_ + ".bin nor nsfwq2_" + tier_ +
             ".bin is in " + modelDir + " -- the gate blocks, so it cannot be skipped";
      return false;
    }

    // ⚠ LOADING IS NOT RUNNING. A context can load and then fail to execute -- reported
    // from an 8 Elite Gen 5 on 0.2.1, where every model loaded and the first execution
    // returned NaN, which surfaced to the user as "the content check could not run" with
    // no way back. Tier selection was by file presence alone, so there was no fallback
    // from a tier the chip would not run.
    //
    // So prove it, by running EVERY model once on synthetic input -- about 50 ms for the
    // whole set, once per init.
    //
    // ⚠ It used to prove only the gate, and that was not enough to act on. The gate is
    // mandatory and executes first, so it is where a broken tier surfaces -- but bf633ac
    // had already written down that this makes it the CANARY and not the culprit, and two
    // releases later nobody could still say whether a v81 part fails at the gate
    // specifically or at everything. Those have different answers: one is a 6.6 MB gate
    // from a lower tier, the other is a 310 MB tier download. The probe that cannot tell
    // them apart is the reason the question stayed open, so it now names every model.
    struct Probe { std::string name; ffnn::Handle h; };
    const Probe probes[] = {
        {"yoloface", p_->n.det},   {"fan2d", p_->n.fan},  {"arcface", p_->n.arc},
        {swapperName, p_->n.swap}, {"fan685", p_->n.fan685},
        {p_->n.lipIsEdtalk ? "edtalk" : "wav2lip", p_->n.lip},
        {"gpen", p_->n.enh},       {"gate", p_->n.nsfw},
    };
    std::string report;
    bool ran = true;
    for (const Probe& pr : probes) {
      // fan685 and gpen are optional and absent on most installs. Not loaded is not a
      // failure, and listing them as one would make every ordinary device look broken.
      if (!pr.h) continue;
      if (!report.empty()) report += ", ";
      report += pr.name;
      std::string why;
      if (probeExecutes(pr.h, &why)) {
        // "ok (non-finite output)" is a real state and it has to reach the report. It is
        // not fatal here, but it is exactly what an 8 Elite Gen 5 was described as doing
        // in 0.2.1, so a device that does it must not look identical to one that does not.
        report += why.empty() ? " ok" : " ok (" + why + ")";
      } else {
        report += " FAILED (" + why + ")";
        ran = false;
      }
    }
    if (!ran) {
      rejected_ = tier_;
      err_ = "tier " + tier_ + " loaded but does not execute: " + report;
      return false;
    }

    // The gate a second time, through its REAL path. The sweep above proves the context
    // executes; this proves the thing the gate is actually asked for -- a finite decision
    // statistic out of checkContent's own preprocessing -- and it is the one model where
    // a plausible-but-wrong answer blocks or admits real content.
    ffcv::Image probe(64, 64, 3);
    std::fill(probe.data.begin(), probe.data.end(), (uint8_t)128);
    ContentVerdict v = checkContent(probe);
    if (!v.ok || !std::isfinite(v.score)) {
      rejected_ = tier_;
      err_ = "tier " + tier_ + " loaded but does not execute: " + report +
             ", gate verdict " + (v.ok ? std::string("not finite") : err_);
      return false;
    }
    return true;
  };

  std::string firstErr;
  for (size_t i = 0; i < candidates.size(); ++i) {
    dropNets();
    if (tryTier(candidates[i])) {
      if (ffdebug() && candidates[i] != chain.front())
        fprintf(stderr, "[dbg] tier %s unusable, fell back to %s\n",
                chain.front().c_str(), tier_.c_str());
      return true;
    }
    if (i == 0) firstErr = err_;
  }
  dropNets();
  err_ = firstErr.empty() ? err_ : firstErr;
  return false;
}

bool Pipeline::hasEnhancer() const { return p_ && p_->n.enh != nullptr; }
bool Pipeline::hasLipSyncer() const { return p_ && p_->n.lip != nullptr; }

bool Pipeline::setAudio(const int16_t* pcm, size_t frames, int channels, int inRate,
                        double fps) {
  if (!p_) return false;
  p_->audio.windows.clear();
  if (!pcm || frames == 0 || channels <= 0 || inRate <= 0 || fps <= 0.0) return false;

  // prepare_voice, in upstream's order: resample first, THEN mix and normalise. The mix
  // is linear and so is the resample, so the two commute, but the PEAK the normalise
  // divides by is measured after resampling and that does not commute -- doing it the
  // other way scales the whole signal differently.
  std::vector<float> wide((size_t)frames * channels);
  for (size_t i = 0; i < wide.size(); ++i) wide[i] = (float)pcm[i];

  std::vector<float> mono(frames);
  if (channels == 1) {
    mono.assign(wide.begin(), wide.end());
  } else {
    for (size_t i = 0; i < frames; ++i) {
      double acc = 0.0;
      for (int c = 0; c < channels; ++c) acc += wide[i * channels + c];
      mono[i] = (float)(acc / channels);
    }
  }
  std::vector<float> at16k = ffaudio::resampleToVoiceRate(mono.data(), mono.size(), inRate);
  if (at16k.empty()) return false;
  std::vector<float> prepared = ffaudio::prepareAudio(at16k.data(), at16k.size(), 1);

  ffaudio::Spectrogram spec = ffaudio::createSpectrogram(prepared);
  if (spec.columns > ffaudio::melColumnLimit()) {
    // Upstream's index array is int16 and wraps negative past here. Refusing is the only
    // honest option: reproducing the wrap would silently sync the wrong 6.8 minutes.
    err_ = "audio too long for the lip syncer (over 6.8 minutes)";
    return false;
  }
  p_->audio.windows = ffaudio::extractWindows(spec, fps);
  // Built here rather than lazily in melWindow(), which is const: a const method quietly
  // filling a cache is legal and is still a worse thing to read.
  p_->audio.silence.assign((size_t)ffaudio::kMelFilterTotal * ffaudio::kAudioStepSize,
                           -4.0f);
  return !p_->audio.windows.empty();
}

const float* Pipeline::melWindow(int frameIndex) const {
  if (!p_ || p_->audio.windows.empty()) return nullptr;
  if (frameIndex >= 0 && (size_t)frameIndex < p_->audio.windows.size()) {
    return p_->audio.windows[(size_t)frameIndex].data.data();
  }
  return p_->audio.silence.data();
}

int Pipeline::melWindowTotal() const {
  return p_ ? (int)p_->audio.windows.size() : 0;
}

void Pipeline::updateConfig(const Config& c) {
  if (!p_) return;
  Config& live = p_->cfg;
  // Named one by one rather than assigned wholesale. A `p_->cfg = c` would also overwrite
  // the swapper-derived geometry with whatever the caller happened to be carrying, and the
  // failure that produces is not a crash -- it is a swap that runs and looks wrong.
  live.detectorScore     = c.detectorScore;
  live.landmarkerScore   = c.landmarkerScore;
  live.nmsThreshold      = c.nmsThreshold;
  live.maskBlur          = c.maskBlur;
  for (int i = 0; i < 4; ++i) live.maskPadding[i] = c.maskPadding[i];
  live.swapperWeight     = c.swapperWeight;
  live.pixelBoost        = c.pixelBoost;
  live.swapLargestOnly   = c.swapLargestOnly;
  live.faceEnhance       = c.faceEnhance;
  live.faceEnhancerBlend = c.faceEnhancerBlend;
  live.lipSyncWeight     = c.lipSyncWeight;
}

// ------------------------------------------------------------ content gate

ContentVerdict Pipeline::checkContent(const ffcv::Image& frame) {
  ContentVerdict v;
  if (!p_ || !p_->n.nsfw) { err_ = "content gate not loaded"; return v; }

  // vision.py:fit_contain_frame -- scale to FIT, then pad CENTRED.
  //
  // ⚠ This is NOT the detector's letterbox.  analyse() scales and pads into the top-left
  // corner; padding centred instead moves every pixel and with it the score.  Reusing the
  // detector's input path here would be silently wrong.
  const int S = kNsfwSize;
  double scale = std::min((double)S / frame.h, (double)S / frame.w);
  int nw = (int)(frame.w * scale), nh = (int)(frame.h * scale);
  int x0 = std::max(0, (S - nw) / 2), y0 = std::max(0, (S - nh) / 2);
  ffcv::Image temp = ffcv::resizeLinear(frame, nw, nh);

  // content_analyser.py:prepare_detect_frame -- BGR -> RGB, /255, then mean 0.5 / std 0.5.
  //
  // ⚠ mean 0.5 and std 0.5, i.e. [0,1] -> [-1,1].  This read `/ 255.0f` alone until
  // 2026-08-30 and handed the model [0,1], which is not the range facefusion 3.8.2 feeds
  // it.  nsfw_reference.py carried the identical mistake, so every device-vs-host
  // comparison agreed and none of them was evidence about upstream.  The decision
  // statistic moves by -1.15 on average against a 0.25 threshold, so this was not a
  // rounding matter -- the gate was judging on a distribution the model never saw.
  //
  // The zero-padded border is left at the pad value the model expects for "no pixel",
  // which after this normalisation is -1, not 0 -- so the buffer is initialised to -1.f.
  std::vector<float> in((size_t)3 * S * S, -1.f);
  for (int y = 0; y < nh; ++y) {
    const uint8_t* row = temp.row(y);
    for (int x = 0; x < nw; ++x)
      for (int c = 0; c < 3; ++c)
        in[(size_t)c * S * S + (size_t)(y + y0) * S + (x + x0)] =
            row[x * 3 + (2 - c)] / 127.5f - 1.0f;   // (2 - c): the [:, :, ::-1] flip
  }

  std::vector<std::vector<float>> out;
  if (!ffnn::execute(p_->n.nsfw, {"input"}, {in.data()}, out)) {
    err_ = std::string("content gate: ") + ffnn::lastError();
    return v;
  }
  if (out.empty() || out[0].size() < 2) {
    err_ = "content gate: expected 2 logits, got " +
           std::to_string(out.empty() ? 0 : out[0].size());
    return v;
  }
  v.ok = true;
  v.score = out[0][0] - out[0][1];
  v.blocked = v.score > p_->cfg.nsfwThreshold;
  return v;
}

// ---------------------------------------------------------------- detection

std::vector<Face> Pipeline::analyse(const ffcv::Image& frame) {
  std::vector<Face> faces;
  const Config& cfg = p_->cfg;
  const int S = cfg.detectorSize;

  double t0 = nowMs();
  // restrict_frame + zero-pad to SxS  (vision.py:222, face_detector.py:445)
  double scale = 1.0;
  if (frame.h > S || frame.w > S)
    scale = std::min((double)S / frame.h, (double)S / frame.w);
  int tw = (int)(frame.w * scale), th = (int)(frame.h * scale);
  if (scale >= 1.0) { tw = frame.w; th = frame.h; }
  double ratioW = (double)frame.w / tw, ratioH = (double)frame.h / th;

  std::vector<float> in((size_t)3 * S * S, 0.f);   // CHW, /255, zero-padded
  if (getenv("FFDETPREPLEGACY")) {
    // The two-step version, kept for the A/B only. See resizeToCHW's header note.
    ffcv::Image temp = (scale < 1.0) ? ffcv::resizeLinear(frame, tw, th) : frame;
    for (int y = 0; y < temp.h; ++y) {
      const uint8_t* row = temp.row(y);
      for (int x = 0; x < temp.w; ++x)
        for (int c = 0; c < 3; ++c)
          in[(size_t)c * S * S + (size_t)y * S + x] = row[x * 3 + c] / 255.0f;
    }
  } else {
    ffcv::resizeToCHW(frame, tw, th, S, in.data());
  }
  msGeom += nowMs() - t0;
  msGeomDetPrep += nowMs() - t0;

  t0 = nowMs();
  std::vector<std::vector<float>> out;
  if (!ffnn::execute(p_->n.det, {"input"}, {in.data()}, out)) {
    err_ = std::string("detector: ") + ffnn::lastError();
    return faces;
  }
  msDetect += nowMs() - t0;

  if (ffdebug()) {
    auto sh = ffnn::outputShapes(p_->n.det);
    fprintf(stderr, "[dbg] detector out elems=%zu shape=", out[0].size());
    for (int v : sh[0]) fprintf(stderr, "%d,", v);
    float mx = -1e9f, mn = 1e9f;
    for (float v : out[0]) { if (v > mx) mx = v; if (v < mn) mn = v; }
    fprintf(stderr, "  range [%.4f, %.4f]\n", mn, mx);
    // score channel under both plausible orderings of a rank-3 [1,20,8400] tensor
    float bestC = -1e9f, bestI = -1e9f;
    for (int i = 0; i < 8400; ++i) {
      bestC = std::max(bestC, out[0][(size_t)4 * 8400 + i]);   // [20,8400]
      bestI = std::max(bestI, out[0][(size_t)i * 20 + 4]);     // [8400,20]
    }
    fprintf(stderr, "[dbg] best score as [20,8400]=%.4f   as [8400,20]=%.4f\n", bestC, bestI);
    fprintf(stderr, "[dbg] frame %dx%d temp %dx%d ratio %.3f/%.3f\n",
            frame.w, frame.h, tw, th, ratioW, ratioH);
  }

  // [1,20,8400] -> per-anchor (cx,cy,w,h, score, 5x(x,y,vis))
  t0 = nowMs();
  const int A = 8400;
  const float* d = out[0].data();
  std::vector<std::array<float, 4>> boxes;
  std::vector<float> scores;
  std::vector<std::array<float, 10>> lms;
  for (int i = 0; i < A; ++i) {
    float sc = d[4 * A + i];
    if (sc <= cfg.detectorScore) continue;
    float cx = d[0 * A + i], cy = d[1 * A + i], w = d[2 * A + i], h = d[3 * A + i];
    boxes.push_back({(float)((cx - w / 2) * ratioW), (float)((cy - h / 2) * ratioH),
                     (float)((cx + w / 2) * ratioW), (float)((cy + h / 2) * ratioH)});
    scores.push_back(sc);
    std::array<float, 10> l{};
    for (int k = 0; k < 5; ++k) {
      l[2 * k] = (float)(d[(5 + 3 * k + 0) * A + i] * ratioW);
      l[2 * k + 1] = (float)(d[(5 + 3 * k + 1) * A + i] * ratioH);
    }
    lms.push_back(l);
  }
  auto keep = ffcv::nmsBoxes(boxes, scores, cfg.detectorScore, cfg.nmsThreshold);
  msGeom += nowMs() - t0;

  for (int idx : keep) {
    Face f{};
    std::memcpy(f.landmark5, lms[idx].data(), sizeof(f.landmark5));
    for (int i = 0; i < 4; ++i) f.box[i] = boxes[idx][i];
    f.detScore = scores[idx];

    // ---- 5 -> 68, then the snapped face angle
    t0 = nowMs();
    float lm68_5[136];
    if (p_->n.fan685) {
      std::vector<std::vector<float>> o5;
      if (ffnn::execute(p_->n.fan685, {"input"}, {f.landmark5}, o5) && o5[0].size() >= 136)
        std::memcpy(lm68_5, o5[0].data(), sizeof(lm68_5));
      else
        std::memset(lm68_5, 0, sizeof(lm68_5));
    } else {
      // Without fan_68_5 only the angle is needed, and estimate_face_angle reads points
      // 0 and 16 -- the outer jaw corners, which track the eye axis. The eye centres give
      // the same snapped result for every non-rotated face.
      std::memset(lm68_5, 0, sizeof(lm68_5));
      lm68_5[0] = f.landmark5[0]; lm68_5[1] = f.landmark5[1];
      lm68_5[32] = f.landmark5[2]; lm68_5[33] = f.landmark5[3];
    }
    int angle = ffcv::estimateFaceAngle(lm68_5);
    msGeom += nowMs() - t0;

    // ---- landmarker: warp to 256, run 2dfan4, soft-argmax back
    t0 = nowMs();
    const int MS = 256;
    float bw = f.box[2] - f.box[0], bh = f.box[3] - f.box[1];
    float sc2 = 195.0f / std::max(1.0f, std::max(bw, bh));
    float tx = (MS - (f.box[0] + f.box[2]) * sc2) * 0.5f;
    float ty = (MS - (f.box[1] + f.box[3]) * sc2) * 0.5f;
    ffcv::Affine aff;
    aff(0, 0) = sc2; aff(0, 1) = 0; aff(0, 2) = tx;
    aff(1, 0) = 0; aff(1, 1) = sc2; aff(1, 2) = ty;
    ffcv::Image crop = ffcv::warpAffine(frame, aff, MS, MS, ffcv::BORDER_CONSTANT);

    ffcv::Affine rot = ffcv::getRotationMatrix2D(MS / 2.0, MS / 2.0, angle, 1.0);
    int rw = MS, rh = MS;
    if (angle % 180 != 0) { rw = MS; rh = MS; }   // square crop: rotation size is unchanged
    if (angle != 0) crop = ffcv::warpAffine(crop, rot, rw, rh, ffcv::BORDER_CONSTANT);
    msGeomWarp += nowMs() - t0;

    double tt = nowMs();
    std::vector<float> fin((size_t)3 * MS * MS);
    for (int y = 0; y < MS; ++y) {
      const uint8_t* row = crop.row(y);
      for (int x = 0; x < MS; ++x)
        for (int c = 0; c < 3; ++c)
          fin[(size_t)c * MS * MS + (size_t)y * MS + x] = row[x * 3 + c] / 255.0f;
    }
    msGeomTensor += nowMs() - tt;
    msGeom += nowMs() - t0;

    t0 = nowMs();
    std::vector<std::vector<float>> hm;
    if (!ffnn::execute(p_->n.fan, {"input"}, {fin.data()}, hm)) {
      err_ = std::string("landmarker: ") + ffnn::lastError();
      return faces;
    }
    msLandmark += nowMs() - t0;

    t0 = nowMs();
    float xy[136], peak[68];
    ffcv::decodeHeatmaps(hm[0].data(), 68, 64, 64, xy, peak);
    for (int k = 0; k < 68; ++k) { xy[2 * k] = xy[2 * k] / 64.f * MS; xy[2 * k + 1] = xy[2 * k + 1] / 64.f * MS; }
    if (angle != 0) {
      ffcv::Affine ir = ffcv::invertAffine(rot);
      float tmp[136]; ffcv::transformPoints(xy, 68, ir, tmp); std::memcpy(xy, tmp, sizeof(xy));
    }
    ffcv::Affine ia = ffcv::invertAffine(aff);
    ffcv::transformPoints(xy, 68, ia, f.landmark68);

    double meanPeak = 0;
    for (int k = 0; k < 68; ++k) meanPeak += peak[k];
    meanPeak /= 68.0;
    float score68 = (float)std::min(1.0, std::max(0.0, meanPeak / 0.9));  // numpy.interp
    if (score68 > cfg.landmarkerScore)
      ffcv::toLandmark5(f.landmark68, f.landmark5_68);
    else
      std::memcpy(f.landmark5_68, f.landmark5, sizeof(f.landmark5_68));

    // ---- recogniser: warp to 112 on arcface_112_v2, BGR->RGB, /127.5-1
    const int RS = 112;
    float tmpl[10];
    const float* T = ffcv::warpTemplate(0);
    for (int i = 0; i < 10; ++i) tmpl[i] = T[i] * RS;
    ffcv::Affine am = ffcv::umeyama(f.landmark5_68, tmpl, 5);
    double tw2 = nowMs();
    ffcv::Image rc = ffcv::warpAffine(frame, am, RS, RS, ffcv::BORDER_REPLICATE);
    msGeomWarp += nowMs() - tw2;
    tw2 = nowMs();
    std::vector<float> rin((size_t)3 * RS * RS);
    for (int y = 0; y < RS; ++y) {
      const uint8_t* row = rc.row(y);
      for (int x = 0; x < RS; ++x)
        for (int c = 0; c < 3; ++c)     // BGR -> RGB is the [:, :, ::-1]
          rin[(size_t)(2 - c) * RS * RS + (size_t)y * RS + x] = row[x * 3 + c] / 127.5f - 1.0f;
    }
    msGeomTensor += nowMs() - tw2;
    msGeom += nowMs() - t0;

    t0 = nowMs();
    std::vector<std::vector<float>> emb;
    if (!ffnn::execute(p_->n.arc, {"input"}, {rin.data()}, emb)) {
      err_ = std::string("recogniser: ") + ffnn::lastError();
      return faces;
    }
    msRecognise += nowMs() - t0;

    std::memcpy(f.embedding, emb[0].data(), sizeof(f.embedding));
    double nrm = 0;
    for (int i = 0; i < 512; ++i) nrm += (double)f.embedding[i] * f.embedding[i];
    nrm = std::sqrt(nrm);
    for (int i = 0; i < 512; ++i) f.embeddingNorm[i] = (float)(f.embedding[i] / nrm);

    faces.push_back(f);
    ++facesDone;
  }
  ++framesDone;
  return faces;
}

bool Pipeline::setSource(const ffcv::Image& img) {
  err_.clear();
  auto faces = analyse(img);
  if (faces.empty()) {
    // Do NOT overwrite a real failure from analyse() -- a graph that failed to execute
    // and a frame with genuinely no face both arrive here as an empty vector, and
    // reporting the second when it was the first sends the next hour in the wrong place.
    if (err_.empty()) err_ = "no face detected in source image";
    return false;
  }
  const Face* best = &faces[0];
  float bestArea = 0;
  for (const auto& f : faces) {
    float a = (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]);
    if (a > bestArea) { bestArea = a; best = &f; }
  }
  std::memcpy(p_->srcEmbedding, best->embedding, sizeof(p_->srcEmbedding));
  std::memcpy(p_->srcEmbeddingNorm, best->embeddingNorm, sizeof(p_->srcEmbeddingNorm));
  p_->haveSource = true;
  // the source frame's own analysis is bookkeeping, not output
  framesDone = 0; facesDone = 0;
  msDetect = msLandmark = msRecognise = msGeom = msEnhance = 0;
  return true;
}

void Pipeline::resetStats() {
  msDetect = msLandmark = msRecognise = msSwap = msGeom = msEnhance = msLipSync = 0;
  msLipCrop = msLipMask = msLipPrep = msLipPaste = 0;
  msGeomDetPrep = msGeomWarp = msGeomTensor = msGeomMask = msGeomPaste = 0;
  framesDone = facesDone = 0;
}

// ---------------------------------------------------- lip syncer (wav2lip_gan_96)
//
// lip_syncer/core.py:sync_lip, the wav2lip branch, in order. The one thing to keep in
// mind reading it: NOTHING here reuses the swapper's crop. The swapper works on
// arcface_128 at 256; this works on ffhq_512 at 512 and then on a 96x96 mouth box taken
// from the 68 landmarks INSIDE that crop. Handing it the swapper's crop would run a
// numerically perfect graph over the wrong pixels, which is trap #9.
bool Pipeline::syncLip(ffcv::Image& frame, const std::vector<Face>& faces,
                       const float* melWindow) {
  if (!p_ || !p_->n.lip || !melWindow) return true;   // absent is not a failure
  const int LS = 512;
  const bool ed = p_->n.lipIsEdtalk;
  const Config& cfg = p_->cfg;

  for (const Face& f : faces) {
    double t0 = nowMs();
    // warp_face_by_face_landmark_5(frame, landmark_set['5/68'], 'ffhq_512', (512, 512))
    float tmpl[10];
    const float* T = ffcv::warpTemplate(2);          // ffhq_512
    for (int i = 0; i < 10; ++i) tmpl[i] = T[i] * LS;
    ffcv::Affine am = ffcv::umeyama(f.landmark5_68, tmpl, 5);

    if (ed) {
      // edtalk (lip_syncer/core.py, the edtalk branch of sync_lip) is NOT the same shape
      // as wav2lip and was wrongly ported as if it were -- checked against upstream after
      // the user reported discoloration and warping, not assumed. It is a full-face
      // generator, not a mouth-box inpainter:
      //   * the model sees the WHOLE 512 crop resized to 256 (cv2.resize, not a
      //     landmark-derived mouth box warped up 256/~280 -- there IS no mouth box);
      //   * it wants RGB. Every other graph in this file does too and gets the (2-c)
      //     index swap on the way in and out (swapAll, the enhancer); this branch was the
      //     one place that swap was missing, feeding a model trained on RGB three BGR
      //     channels it reads as R and B swapped -- the discoloration was exactly that;
      //   * the blend mask is create_box_mask (the swapper's own mask, `cfg.maskBlur` /
      //     `cfg.maskPadding` -- a near-whole-crop feathered rectangle), not the
      //     lower-face hull. The lower-face mask was wav2lip's inpaint region leaking into
      //     a model that was never restricted to one -- the visible seam around a smaller
      //     patch than the model actually redrew was the "warping".
      double tc = nowMs();
      ffcv::Image crop = ffcv::warpAffine(frame, am, LS, LS, ffcv::BORDER_REPLICATE);
      msLipCrop += nowMs() - tc;

      tc = nowMs();
      const ffcv::MatF& mask = p_->lipBoxMask.get(LS, cfg.maskBlur, cfg.maskPadding);
      msLipMask += nowMs() - tc;
      msGeom += nowMs() - t0;

      t0 = nowMs();
      const int MS = 256;
      // cv2.resize(crop, (256,256), INTER_AREA): measured bit-identical to bilinear at
      // this exact 2x ratio (0 LSB over a blurred random 512x512), so resizeLinear is
      // exact here -- ffcv has no separate area filter and does not need one.
      ffcv::Image area = ffcv::resizeLinear(crop, MS, MS);
      std::vector<float> in((size_t)3 * MS * MS);
      for (int y = 0; y < MS; ++y) {
        const uint8_t* row = area.row(y);
        for (int x = 0; x < MS; ++x)
          for (int c = 0; c < 3; ++c)
            in[(size_t)(2 - c) * MS * MS + (size_t)y * MS + x] = row[x * 3 + c] / 255.0f;
      }
      msGeom += nowMs() - t0;
      msLipPrep += nowMs() - t0;

      t0 = nowMs();
      std::vector<std::vector<float>> out;
      // The lip-direction scale, `--lip-syncer-weight` -- read directly off `cfg`, not
      // fixed. Upstream's default is 0.5, not 1.0: driving it at a hardcoded 1.0 was this
      // port inventing a value, not matching upstream's own default.
      float lipWeight = cfg.lipSyncWeight;
      if (!ffnn::execute(p_->n.lip, {"source", "target", "weight"},
                         {melWindow, in.data(), &lipWeight}, out) ||
          out.empty() || out[0].size() < (size_t)3 * MS * MS) {
        err_ = std::string("lip syncer: ") + ffnn::lastError();
        return false;
      }
      msLipSync += nowMs() - t0;

      t0 = nowMs();
      ffcv::Image synced(MS, MS, 3);
      for (int y = 0; y < MS; ++y) {
        uint8_t* row = synced.row(y);
        for (int x = 0; x < MS; ++x)
          for (int c = 0; c < 3; ++c) {
            float v = out[0][(size_t)c * MS * MS + (size_t)y * MS + x];
            v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
            row[x * 3 + (2 - c)] = (uint8_t)(v * 255.0f);   // RGB -> BGR
          }
      }
      // cv2.resize(.., (512,512), INTER_CUBIC): the one approximation left in this branch.
      // Measured against bilinear at this ratio: 3 LSB max / 0.4 LSB mean over a blurred
      // random image -- not exact, but sub-1% of range and a long way under the two bugs
      // this branch exists to fix. A byte-exact bicubic is future work if that residual
      // ever shows up in a deploy-SNR number instead of just this comment.
      ffcv::Image back = ffcv::resizeLinear(synced, LS, LS);
      ffcv::MatF backF(LS, LS, 3);
      for (int y = 0; y < LS; ++y) {
        const uint8_t* srow = back.row(y);
        float* drow = backF.row(y);
        for (int i = 0, n = LS * 3; i < n; ++i) drow[i] = srow[i];
      }
      ffcv::pasteBack(frame, backF, mask, am);
      msGeom += nowMs() - t0;
      msLipPaste += nowMs() - t0;
      continue;
    }

    // wav2lip below. MS=96 is the size the 68-landmark mouth box is warped to, and it IS
    // the resolution the mouth gets drawn at -- the box measures ~274x285 inside the 512
    // crop, so this upscales the mouth about 3x on the way back. wav2lip stays BGR
    // in and out (upstream's prepare_crop_frame never flips it for this model).
    const int MS = 96;

    // cv2.transform(landmark_68, affine_matrix) -- the 68 points INTO crop space. Every
    // step below reads these, not the frame-space ones. Taken BEFORE the crop, because
    // the crop is now built only where these say it will be read.
    float lm[136];
    ffcv::transformPoints(f.landmark68, 68, am, lm);

    float box[4];
    ffcv::createBoundingBox(lm, box);

    // Only a rectangle of the 512 crop is ever read: the box comes from these same 68
    // points, and the lower-face hull is built from a SUBSET of them, so hull ⊆ box. The
    // margin is two blur radii, because createAreaMask blurs at sigma 5 and the mask can
    // therefore be non-zero up to ~20 px outside the hull.
    //
    // Warping the whole 512x512 measured 6.66 ms on the host and this is the same picture
    // wherever anything looks at it. Outside the rectangle both the crop and the pasted
    // result are multiplied by a mask that is zero.
    const int kBlurMargin = 2 * ((int)std::lround(5.0 * 4.0 * 2.0 + 1.0) / 2);
    const int rx0 = (int)box[0] - kBlurMargin, ry0 = (int)box[1] - kBlurMargin;
    const int rx1 = (int)box[2] + kBlurMargin + 1, ry1 = (int)box[3] + kBlurMargin + 1;
    double tc = nowMs();
    ffcv::Image crop = ffcv::warpAffineRoi(frame, am, LS, LS, ffcv::BORDER_REPLICATE,
                                           rx0, ry0, rx1, ry1);
    msLipCrop += nowMs() - tc;

    tc = nowMs();
    ffcv::MatF areaMask = ffcv::createAreaMask(LS, LS, lm, ffcv::AREA_LOWER_FACE);
    ffcv::Affine areaM;
    ffcv::Image area = ffcv::warpFaceByBoundingBox(crop, box, MS, &areaM);
    msLipMask += nowMs() - tc;
    msGeom += nowMs() - t0;

    // prepare_crop_frame: the masked copy CONCATENATED with the reference on the channel
    // axis, masked first. Upstream zeroes rows 48.. of the HWC frame, i.e. the BOTTOM
    // half -- the mouth is what the model is asked to INVENT, so it must not be shown it.
    t0 = nowMs();
    std::vector<float> in((size_t)6 * MS * MS);
    for (int y = 0; y < MS; ++y) {
      const uint8_t* row = area.row(y);
      const bool masked = y >= MS / 2;
      for (int x = 0; x < MS; ++x) {
        for (int c = 0; c < 3; ++c) {
          const float v = row[x * 3 + c] / 255.0f;
          in[(size_t)c * MS * MS + (size_t)y * MS + x] = masked ? 0.0f : v;
          in[(size_t)(c + 3) * MS * MS + (size_t)y * MS + x] = v;
        }
      }
    }
    msGeom += nowMs() - t0;
    msLipPrep += nowMs() - t0;

    t0 = nowMs();
    std::vector<std::vector<float>> out;
    // prepare_audio_frame (wav2lip branch): scale the ALREADY-COMPUTED mel window by
    // weight*2.0, not the target crop and not the raw audio. `melWindow` is shared, owned
    // storage read again for other faces/frames, so this copies rather than scaling it in
    // place. Upstream's default 0.5 makes this a *1.0 no-op, which is why the missing knob
    // was invisible until it was looked for.
    std::vector<float> scaledMel(melWindow, melWindow + 80 * 16);
    const float melScale = cfg.lipSyncWeight * 2.0f;
    for (float& v : scaledMel) v *= melScale;
    if (!ffnn::execute(p_->n.lip, {"source", "target"}, {scaledMel.data(), in.data()}, out) ||
        out.empty() || out[0].size() < (size_t)3 * MS * MS) {
      err_ = std::string("lip syncer: ") + ffnn::lastError();
      return false;
    }
    msLipSync += nowMs() - t0;

    // normalize_crop_frame: clip(0, 1) * 255, back to interleaved BGR.
    t0 = nowMs();
    ffcv::Image synced(MS, MS, 3);
    for (int y = 0; y < MS; ++y) {
      uint8_t* row = synced.row(y);
      for (int x = 0; x < MS; ++x) {
        for (int c = 0; c < 3; ++c) {
          float v = out[0][(size_t)c * MS * MS + (size_t)y * MS + x];
          v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
          row[x * 3 + c] = (uint8_t)(v * 255.0f);
        }
      }
    }

    // Back into the 512 crop through the inverse box warp, BORDER_REPLICATE, then paste
    // the crop into the frame through the lower-face mask.
    ffcv::Image back = ffcv::warpAffineRoi(synced, ffcv::invertAffine(areaM), LS, LS,
                                           ffcv::BORDER_REPLICATE, rx0, ry0, rx1, ry1);
    // Only the rectangle, for the same reason the warp above fills only the rectangle:
    // `back` is zero outside it and MatF value-initialises, so the two agree pixel for
    // pixel and pasteBack multiplies the difference by a mask that is zero there anyway.
    // The full-canvas version converted 786432 pixels to reach a mouth about 300 across,
    // and this block MEASURED 10.71 ms/frame, 49% of the lip syncer's geometry.
    ffcv::MatF backF(LS, LS, 3);
    const int cy0 = std::max(ry0, 0), cy1 = std::min(ry1, LS);
    const int cx0 = std::max(rx0, 0), cx1 = std::min(rx1, LS);
    for (int y = cy0; y < cy1; ++y) {
      const uint8_t* srow = back.row(y) + (size_t)cx0 * 3;
      float* drow = backF.row(y) + (size_t)cx0 * 3;
      for (int i = 0, n = (cx1 - cx0) * 3; i < n; ++i) drow[i] = srow[i];
    }
    ffcv::pasteBackRoi(frame, backF, areaMask, am, cx0, cy0, cx1, cy1);
    msGeom += nowMs() - t0;
    msLipPaste += nowMs() - t0;
  }
  return true;
}

bool Pipeline::swapAll(ffcv::Image& frame, const std::vector<Face>& faces) {
  if (!p_->haveSource) { err_ = "setSource not called"; return false; }
  const Config& cfg = p_->cfg;
  const int SS = cfg.swapSize;                       // what the GRAPH takes: always 256
  const int PB = cfg.pixelBoost < 1 ? 1 : cfg.pixelBoost;
  const int CS = SS * PB;                            // what the FACE is warped at

  float tmpl[10];
  const float* T = ffcv::warpTemplate(1);            // arcface_128, both swappers
  for (int i = 0; i < 10; ++i) tmpl[i] = T[i] * CS;

  // The mask is built at the crop size, not the model size: with pixel boost the crop is
  // larger than the graph and paste_back works on the crop. Cached, because CS, maskBlur
  // and maskPadding are identical on every frame of a run and this is a 79-tap blur.
  double tm = nowMs();
  const ffcv::MatF& mask = p_->swapBoxMask.get(CS, cfg.maskBlur, cfg.maskPadding);
  msGeom += nowMs() - tm;
  msGeomMask += nowMs() - tm;

  // face_selector_mode `one`: the largest detected face by box area.
  const Face* only = nullptr;
  if (cfg.swapLargestOnly) {
    float best = -1.f;
    for (const Face& f : faces) {
      float a = (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]);
      if (a > best) { best = a; only = &f; }
    }
  }

  for (const Face& f : faces) {
    if (only && &f != only) continue;

    double t0 = nowMs();
    ffcv::Affine am = ffcv::umeyama(f.landmark5_68, tmpl, 5);
    double tw3 = nowMs();
    ffcv::Image crop = ffcv::warpAffine(frame, am, CS, CS, ffcv::BORDER_REPLICATE);
    msGeomWarp += nowMs() - tw3;

    // balance_source_embedding (face_swapper/core.py:715). Identity-space, so it is
    // computed once per face and shared by every pixel-boost sub-image.
    float w = 0.35f - 0.70f * cfg.swapperWeight;     // numpy.interp(w, [0,1], [0.35,-0.35])
    double tn = 0;
    for (int i = 0; i < 512; ++i) tn += (double)f.embedding[i] * f.embedding[i];
    tn = std::sqrt(tn);
    std::vector<float> src(512);
    for (int i = 0; i < 512; ++i)
      src[i] = p_->srcEmbeddingNorm[i] * (1.f - w) + (float)(f.embedding[i] / tn) * w;
    msGeom += nowMs() - t0;

    ffcv::MatF outCrop(CS, CS, 3);
    std::vector<float> tin((size_t)3 * SS * SS);

    // implode_pixel_boost / explode_pixel_boost (processors/pixel_boost.py), done as
    // strided indexing rather than the reshape+transpose pair: sub-image k = (ty, tx)
    // holds crop[y*PB + ty, x*PB + tx], and the output goes straight back to the same
    // positions. PB == 1 collapses to the original single pass with stride 1.
    for (int k = 0; k < PB * PB; ++k) {
      const int ty = k / PB, tx = k % PB;

      t0 = nowMs();
      for (int y = 0; y < SS; ++y) {
        const uint8_t* row = crop.row(y * PB + ty);
        for (int x = 0; x < SS; ++x)
          for (int c = 0; c < 3; ++c)
            tin[(size_t)(2 - c) * SS * SS + (size_t)y * SS + x] =
                ((row[(x * PB + tx) * 3 + c] / 255.0f) - cfg.swapMean) / cfg.swapStd;
      }
      msGeom += nowMs() - t0;
      msGeomTensor += nowMs() - t0;

      t0 = nowMs();
      std::vector<std::vector<float>> so;
      if (!ffnn::execute(p_->n.swap, {"target", "source"}, {tin.data(), src.data()}, so)) {
        err_ = std::string("swapper: ") + ffnn::lastError();
        return false;
      }
      msSwap += nowMs() - t0;

      t0 = nowMs();
      const float* o = so[0].data();
      for (int y = 0; y < SS; ++y) {
        float* orow = outCrop.row(y * PB + ty);
        for (int x = 0; x < SS; ++x)
          for (int c = 0; c < 3; ++c) {
            float v = o[(size_t)c * SS * SS + (size_t)y * SS + x];
            if (cfg.swapDenorm) v = v * cfg.swapStd + cfg.swapMean;
            v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
            orow[(x * PB + tx) * 3 + (2 - c)] = v * 255.f;   // RGB -> BGR
          }
      }
      msGeom += nowMs() - t0;
      msGeomTensor += nowMs() - t0;
    }

    // The enhancer used to run HERE, fused into the swap crop. It no longer does -- see
    // `enhance()`, always called as its own pass now, after this and after `syncLip`
    // when the caller has one. Measured: fusing it here meant edtalk's whole-face
    // regeneration overwrote nearly everything it had just sharpened.

    t0 = nowMs();
    ffcv::pasteBack(frame, outCrop, mask, am);
    msGeom += nowMs() - t0;
    msGeomPaste += nowMs() - t0;
  }
  return true;
}

bool Pipeline::enhance(ffcv::Image& frame, const std::vector<Face>& faces) {
  if (!p_ || !p_->n.enh) return true;   // absent is not a failure, matches syncLip
  const Config& cfg = p_->cfg;
  if (!cfg.faceEnhance) return true;
  const int SS = cfg.swapSize;
  const int PB = cfg.pixelBoost < 1 ? 1 : cfg.pixelBoost;
  const int CS = SS * PB;
  if (CS % kEnhSize != 0) return true;

  float tmpl[10];
  const float* T = ffcv::warpTemplate(1);            // arcface_128, same as swapAll
  for (int i = 0; i < 10; ++i) tmpl[i] = T[i] * CS;

  ffcv::MatF mask = ffcv::createBoxMask(CS, CS, cfg.maskBlur, cfg.maskPadding);

  const Face* only = nullptr;
  if (cfg.swapLargestOnly) {
    float best = -1.f;
    for (const Face& f : faces) {
      float a = (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]);
      if (a > best) { best = a; only = &f; }
    }
  }

  const int ES = kEnhSize, EPB = CS / ES;
  for (const Face& f : faces) {
    if (only && &f != only) continue;

    double t0 = nowMs();
    ffcv::Affine am = ffcv::umeyama(f.landmark5_68, tmpl, 5);
    // Re-warp whatever is in `frame` NOW -- this IS upstream's actual face_enhancer pass
    // (see swapAll's comment on the fused version being the approximation of this), just
    // called at a different POINT in the pipeline than the old fused call was.
    ffcv::Image crop = ffcv::warpAffine(frame, am, CS, CS, ffcv::BORDER_REPLICATE);
    msGeom += nowMs() - t0;

    ffcv::MatF enhCrop(CS, CS, 3);
    std::vector<float> ein((size_t)3 * ES * ES);
    for (int k = 0; k < EPB * EPB; ++k) {
      const int ty = k / EPB, tx = k % EPB;

      t0 = nowMs();
      for (int y = 0; y < ES; ++y) {
        const uint8_t* row = crop.row(y * EPB + ty);
        for (int x = 0; x < ES; ++x)
          for (int c = 0; c < 3; ++c)
            ein[(size_t)(2 - c) * ES * ES + (size_t)y * ES + x] =
                ((row[(x * EPB + tx) * 3 + c] / 255.0f) - 0.5f) / 0.5f;
      }
      msGeom += nowMs() - t0;

      t0 = nowMs();
      std::vector<std::vector<float>> eo;
      if (!ffnn::execute(p_->n.enh, {"input"}, {ein.data()}, eo)) {
        err_ = std::string("enhancer: ") + ffnn::lastError();
        return false;
      }
      msEnhance += nowMs() - t0;

      t0 = nowMs();
      const float* e = eo[0].data();
      for (int y = 0; y < ES; ++y) {
        float* erow = enhCrop.row(y * EPB + ty);
        for (int x = 0; x < ES; ++x)
          for (int c = 0; c < 3; ++c) {
            float v = e[(size_t)c * ES * ES + (size_t)y * ES + x];
            v = v < -1.f ? -1.f : (v > 1.f ? 1.f : v);
            erow[(x * EPB + tx) * 3 + (2 - c)] = (v + 1.f) * 0.5f * 255.f;
          }
      }
      msGeom += nowMs() - t0;
    }

    // blend_paste_frame, against `crop` -- the frame this pass started from -- since
    // there is no separate "un-enhanced swap output" once syncLip has already run.
    t0 = nowMs();
    const float b = cfg.faceEnhancerBlend < 0.f ? 0.f
                  : (cfg.faceEnhancerBlend > 1.f ? 1.f : cfg.faceEnhancerBlend);
    ffcv::MatF outF(CS, CS, 3);
    for (int y = 0; y < CS; ++y) {
      const uint8_t* srow = crop.row(y);
      const float* erow = enhCrop.row(y);
      float* orow = outF.row(y);
      for (int i = 0, n = CS * 3; i < n; ++i)
        orow[i] = (float)srow[i] * (1.f - b) + erow[i] * b;
    }
    msGeom += nowMs() - t0;

    t0 = nowMs();
    ffcv::pasteBack(frame, outF, mask, am);
    msGeom += nowMs() - t0;
  }
  return true;
}

}  // namespace ffpipe
