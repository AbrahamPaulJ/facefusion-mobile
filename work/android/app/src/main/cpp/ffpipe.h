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

  // content_analyser.py:detect_with_nsfw_2 -- `logit[0] - logit[1] > 0.25` flags a frame.
  float nsfwThreshold = 0.25f;
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

  // The tier init() chose. Empty until init() has run.
  const std::string& tier() const { return tier_; }

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

  // Every face in one frame, fully analysed (detector + landmarker + recogniser).
  std::vector<Face> analyse(const ffcv::Image& frame);

  // The source identity: the largest face of the source image, embedding only.
  bool setSource(const ffcv::Image& sourceImage);

  // Swap every face in `frame`, in place.
  bool swapAll(ffcv::Image& frame, const std::vector<Face>& faces);

  const std::string& error() const { return err_; }
  // Cumulative per-stage milliseconds, for the CLI's report.
  double msDetect = 0, msLandmark = 0, msRecognise = 0, msSwap = 0, msGeom = 0;
  int framesDone = 0, facesDone = 0;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  std::string err_;
  std::string tier_;
  bool nsfwQuantised_ = false;
};

}  // namespace ffpipe
