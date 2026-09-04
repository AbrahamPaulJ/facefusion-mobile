// The FaceFusion default swap chain, on device.
//
// Mirrors work/pipeline/run_reference.py step for step -- that file is the golden
// reference every number in docs/ is measured against, and it cites the upstream source
// line for each stage.  Keep the two in sync.
//
// Per frame:
//   restrict + letterbox -> [yoloface] -> decode 8400 anchors -> NMS
//   per face: 5->68 (fan_68_5) -> angle -> warp 256 -> [2dfan4] -> soft-argmax
//             -> warp 112 -> [arcface] -> warp 256 -> [swapper] -> box mask -> paste back

#pragma once
#include <memory>
#include <string>
#include <vector>

#include "ffcv.h"

namespace ffpipe {

struct Face {
  float landmark5[10];       // from the detector
  float landmark5_68[10];    // refined by the landmarker, used for every alignment
  float landmark68[136];
  float box[4];
  float detScore = 0.f;
  float embedding[512];
  float embeddingNorm[512];
};

struct Config {
  // facefusion/program.py defaults for the default swap path
  int detectorSize = 640;
  float detectorScore = 0.5f;
  float landmarkerScore = 0.5f;
  float nmsThreshold = 0.4f;
  float maskBlur = 0.3f;
  int maskPadding[4] = {0, 0, 0, 0};   // top, right, bottom, left -- percent, 0..100

  // --face-swapper-weight.  Not a blend of images: it blends the two IDENTITY embeddings
  // before the generator sees them (face_swapper/core.py:715), so it changes who the face
  // is rather than how strongly a finished swap is composited.
  //
  //   0.5 (default) -> w = 0     -> the source embedding, unmodified
  //   1.0           -> w = -0.35 -> source scaled 1.35 AND the target identity subtracted
  //   0.0           -> w = +0.35 -> 35% of the target's own identity blended back in
  //
  // The upper half of the range is what "make the source stronger" means here.
  float swapperWeight = 0.5f;

  // --face-swapper-pixel-boost, as a per-axis factor: 1 = 256, 2 = 512, 3 = 768, 4 = 1024.
  //
  // The graph stays 256x256 -- this is NOT a bigger model. The face is warped at
  // 256*pixelBoost, split into pixelBoost^2 POLYPHASE sub-images (every Nth pixel, not
  // tiles), each run through the same context, and interleaved back. So it costs
  // pixelBoost^2 swapper invocations and needs no reconversion, which is the one quality
  // knob a fixed-shape QNN graph still allows.
  int pixelBoost = 1;

  // --face-selector-mode, reduced to the two that need no reference-face UI.
  // false = `many`, every detected face; true = `one`, the largest by box area.
  bool swapLargestOnly = false;
  // hyperswap: 256, mean/std 0.5, denormalise.  inswapper: 128, mean 0/std 1, no denorm.
  int swapSize = 256;
  float swapMean = 0.5f, swapStd = 0.5f;
  bool swapDenorm = true;
  bool swapperIsHyperswap = true;

  // --face-enhancer, gpen_bfr_256.  OFF by default and deliberately so: it is another
  // 8.57 GMAC per face on top of the swapper's 31.93, and its context binary is a separate
  // download that most installs will not have.
  bool faceEnhance = false;

  // --face-enhancer-blend.  Upstream is 0-100; this is the same number as a fraction.
  // 1 = fully enhanced, 0 = the swapper's output untouched.  80 is upstream's default.
  float faceEnhancerBlend = 0.8f;

  // --lip-syncer-weight, 0-1, upstream default 0.5. ONE knob, applied differently per
  // model (lip_syncer/core.py): edtalk reads it directly as its third input, the
  // lip-direction scale; wav2lip scales the mel window by `weight * 2.0` right before the
  // model sees it. See syncLip in ffpipe.cpp for where each is applied.
  float lipSyncWeight = 0.5f;

  // content_analyser.py:detect_with_nsfw_2 -- `logit[0] - logit[1] > 0.25` flags a frame.
  float nsfwThreshold = 0.25f;

  // Tiers this DEVICE has already proved it cannot run, so init does not spend a load on
  // them again. Set by the caller from what a previous init reported through
  // `rejectedVariant()`; empty on a device that has never rejected one.
  //
  // ⚠ Advisory, never absolute. If skipping leaves no candidate at all, init ignores this
  // list entirely and tries anyway: a wrong entry here must cost a slow start, never the
  // whole app. The caller is also expected to forget these on an app UPDATE -- the next
  // build may be exactly the one that fixes the tier.
  std::vector<std::string> skipVariants;
};

// The content gate's answer for ONE frame.
//
// ⚠ `score` is upstream's decision statistic, `logit[0] - logit[1]`, and is comparable
// straight across host and device -- which is the only reason the quantisation bias below
// is measurable at all.
struct ContentVerdict {
  bool ok = false;        // the graph ran; false means `error()` says why
  bool blocked = false;   // score > Config::nsfwThreshold
  float score = 0.f;
};

class Pipeline {
 public:
  // Both are defined out of line: with a pimpl, the implicitly generated ones would need
  // Impl complete at every call site that merely constructs a Pipeline.
  Pipeline();
  ~Pipeline();
  // modelDir holds the `<name>_<tier>.bin` context binaries, where the tier is measured
  // off the HTP at init (v68 / v73 / v79); libDir/skelDir the QNN runtime.
  bool init(const std::string& libDir, const std::string& skelDir,
            const std::string& modelDir, const std::string& swapperName,
            const Config& cfg);

  /**
   * Change the per-frame tunables on a pipeline that is ALREADY LOADED.
   *
   * ⚠ Every field this copies is read inside `analyse`/`swapAll`/`checkContent`, once per
   * frame — none of them is consumed at init. So a weight, a mask blur, a detector
   * threshold, a pixel boost, and the face enhancer's on/off are all just numbers the next
   * frame will read, and changing one never needed a model reloaded. `gpen` in particular
   * is opened whether or not `faceEnhance` is set, because the flag decides whether the
   * STAGE RUNS, not whether the model exists.
   *
   * What it deliberately does NOT copy is everything derived from WHICH SWAPPER is loaded
   * — swapSize, swapMean, swapStd, swapDenorm, swapperIsHyperswap — and `skipVariants`,
   * which only tier selection reads. Those describe the loaded graph, so accepting them
   * here would let a caller silently tell a 256-pixel hyperswap context that it is a
   * 128-pixel inswapper. Changing the swapper is a reload, and that is the only thing that
   * still is one.
   */
  void updateConfig(const Config& c);

  // The tier init() chose. Empty until init() has run.
  const std::string& tier() const { return tier_; }

  /**
   * A tier that LOADED and then would not execute, or empty.
   *
   * The distinction from an ordinary init failure is the whole point: a tier whose files
   * are missing is a download problem and says nothing about the silicon, while a tier
   * that loads every context and then fails to run one is a property of THIS chip that
   * will be just as true next launch. Only the second is worth remembering, and only the
   * second belongs in `Config::skipVariants`.
   *
   * Set even when init ultimately SUCCEEDS on a later tier, so the caller can record the
   * rejection without having to fail first.
   */
  const std::string& rejectedVariant() const { return rejected_; }

  /**
   * Upstream's content gate on ONE frame (content_analyser.py:detect_with_nsfw_2).
   *
   * This port gates on `nsfw_2` alone; upstream votes 2-of-3 across three models totalling
   * 461 MB.  A deliberate divergence -- see docs/roadmap.md 2.
   *
   * The caller applies the policy.  For a still that is one call; for a video it is one
   * sampled frame per second, refused above a 10% flagged rate.  Sampling is upstream's,
   * and it is what makes a 5 ms graph cost ~56 ms for a whole clip instead of 1.5 s.
   */
  ContentVerdict checkContent(const ffcv::Image& frame);

  // True when the gate is quantised, i.e. this tier had no fp32 context.  The W8A16 build
  // shifts `score` by +0.087 mean / +0.153 max TOWARD blocking, against a 0.25 threshold,
  // so a caller that cares about false refusals has to know.  No compensation is applied
  // here: the correction is an unmeasured constant and does not belong in the runner.
  bool contentGateIsQuantised() const { return nsfwQuantised_; }

  // Whether gpen_<tier>.bin was found at init.  The UI offers the enhancer switch
  // only when this is true -- the same rule inswapper follows, and for the same
  // reason: offering a model that is not on the device turns a missing file into a
  // failed run at the worst possible moment.
  bool hasEnhancer() const;

  // Every face in one frame, fully analysed (detector + landmarker + recogniser).
  std::vector<Face> analyse(const ffcv::Image& frame);

  // The source identity: the largest face of the source image, embedding only.
  bool setSource(const ffcv::Image& sourceImage);

  /**
   * Take the clip's audio once, and hold one 80x16 mel window per video frame.
   *
   * `pcm` is interleaved as the decoder produced it; the resample to 16 kHz, the mix to
   * mono, the normalise and the pre-emphasis all happen HERE rather than in Kotlin, so
   * there is exactly one implementation of upstream's arithmetic and it is the one the
   * host test measures.
   *
   * `fps` is the rate frames will be PRESENTED at, which for a rate-reduced run is the
   * output rate and not the source's: window k belongs to output frame k.
   */
  bool setAudio(const int16_t* pcm, size_t frames, int channels, int inRate, double fps);

  /**
   * The window for one output frame, or SILENCE past the end of the audio.
   *
   * Silence is upstream's behaviour, not a guard: `create_empty_audio_frame` gives zeros,
   * which `prepare_audio_frame` turns into a uniform -4, and the model reads that as "no
   * speech" and closes the mouth. Returning null and skipping the frame instead would
   * leave the mouth wherever the swapper put it, which is worse and looks like a stutter.
   *
   * Null only when setAudio has never been called.
   */
  const float* melWindow(int frameIndex) const;

  // How many windows setAudio produced. Zero before it is called.
  int melWindowTotal() const;

  // Whether wav2lip_<tier>.bin was found at init, on the same rule as hasEnhancer():
  // a switch is offered only for a model that is actually on the device.
  bool hasLipSyncer() const;

  /**
   * Upstream's lip syncer on ONE frame (lip_syncer/core.py:sync_lip, wav2lip branch).
   *
   * `melWindow` is 80 x 16 row-major, ALREADY through ffaudio::extractWindows -- that is
   * upstream's prepare_audio_frame including the weight, so this does not scale it again.
   *
   * ⚠ This does NOT reuse the swap crop, and that is the whole subtlety. It warps to
   * ffhq_512 by landmark5_68, transforms the 68 points INTO that crop, and takes both the
   * lower-face mask and the mouth box from them there. Feeding it the swapper's
   * arcface_128 crop would run a numerically perfect graph on the wrong pixels.
   *
   * Runs after the swap, as its own pass, exactly as upstream orders its processors.
   */
  bool syncLip(ffcv::Image& frame, const std::vector<Face>& faces, const float* melWindow);

  // Swap every face in `frame`, in place. Never enhances -- see `enhance()`, always
  // called separately now, after `syncLip` when the caller has one.
  bool swapAll(ffcv::Image& frame, const std::vector<Face>& faces);

  /**
   * The enhancer, as its OWN pass -- ALWAYS called separately, never fused into
   * `swapAll`, so it runs LAST: after `syncLip` when the caller has one, otherwise
   * straight after `swapAll`. Re-warps `frame` (whatever is in it NOW) from the target
   * landmarks, enhances, pastes back a second time. This is what upstream's own
   * face_enhancer actually does -- a genuinely separate processor pass, not a shortcut
   * fused into the swap crop the way an earlier version of this file had it.
   *
   * Measured, not assumed: fusing enhance before lip sync meant edtalk's whole-face
   * regeneration overwrote nearly everything the enhancer had just sharpened, visibly --
   * a side-by-side comparison showed noticeably softer skin texture, eyes and eyebrows
   * with the fused order against the same frame enhanced after lip sync instead.
   *
   * A no-op returning true when `cfg.faceEnhance` is off or the model is not loaded --
   * same "absent is not a failure" convention as `syncLip`.
   */
  bool enhance(ffcv::Image& frame, const std::vector<Face>& faces);

  const std::string& error() const { return err_; }
  // Cumulative per-stage milliseconds, for the CLI's report.
  double msDetect = 0, msLandmark = 0, msRecognise = 0, msSwap = 0, msGeom = 0;
  // Separate from msSwap on purpose: the enhancer is the one stage a user can turn
  // off, so its cost has to be attributable rather than folded into the swapper's.
  double msEnhance = 0;
  // Likewise separate: the lip syncer is optional and its cost has to be attributable.
  double msLipSync = 0;
  // syncLip's GEOMETRY, split, because the first device measurement of it said the
  // stage is 94% geometry and could not say which part. These four sum to the
  // msGeom that syncLip contributes; the swapper's own geometry is not in them.
  double msLipCrop = 0;    // the 960 -> 512 warp
  double msLipMask = 0;    // createAreaMask + the 512 -> 96 box warp
  double msLipPrep = 0;    // the 6x96x96 input tensor
  double msLipPaste = 0;   // 96 -> 512 warp, the float copy, and pasteBack
  // The SWAP path's geometry, split the same way and for the same reason: msGeom is one
  // bucket over four different kinds of work, and 41% of a plain swap's compute is in it.
  // These four are a SUBSET of msGeom, not additional to it -- they say where it went.
  double msGeomDetPrep = 0;  // detector letterbox resize + the 3x640x640 CHW conversion
  double msGeomWarp = 0;     // warpAffine crops: landmarker, recogniser, swapper
  double msGeomTensor = 0;   // uint8 crop <-> float CHW tensor conversions
  double msGeomMask = 0;     // createBoxMask alone -- frame-invariant, so suspect
  double msGeomPaste = 0;    // pasteBack alone
  int framesDone = 0, facesDone = 0;
  // Zero every counter above.  A run has to call this itself: the accumulators are
  // cumulative across frames by design, and setSource clears only SOME of them --
  // msSwap and msLipSync survive it, so a second run without this reports the first
  // run's time added to its own.
  void resetStats();

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  std::string err_;
  std::string tier_;
  std::string rejected_;
  bool nsfwQuantised_ = false;
};

}  // namespace ffpipe
