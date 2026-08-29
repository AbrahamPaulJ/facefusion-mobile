#include "ffpipe.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>

#include "ffqnn.h"

// FFDEBUG=1 prints what each stage actually produced.  Cheap, and the alternative is
// guessing at a tensor layout from the host side.
#include <cstdio>
#include <cstdlib>
static bool ffdebug() { static bool v = getenv("FFDEBUG") != nullptr; return v; }

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
  ffqnn::Handle det = nullptr, fan = nullptr, fan685 = nullptr, arc = nullptr, swap = nullptr;
  ffqnn::Handle nsfw = nullptr;
};

// content_analyser.py:create_static_model_set -- nsfw_2 is 384x384, mean 0, std 1.
constexpr int kNsfwSize = 384;

}  // namespace

struct Pipeline::Impl {
  Nets n;
  Config cfg;
  float srcEmbedding[512]{};
  float srcEmbeddingNorm[512]{};
  bool haveSource = false;
  std::vector<float> emap;   // inswapper only: the 512x512 initializer
};

Pipeline::Pipeline() = default;

Pipeline::~Pipeline() {
  if (p_) {
    for (auto h : {p_->n.det, p_->n.fan, p_->n.fan685, p_->n.arc, p_->n.swap, p_->n.nsfw})
      if (h) ffqnn::release(h);
  }
}

bool Pipeline::init(const std::string& libDir, const std::string& skelDir,
                    const std::string& modelDir, const std::string& swapperName,
                    const Config& cfg) {
  p_.reset(new Impl());
  p_->cfg = cfg;

  if (!ffqnn::init(libDir + "/libQnnHtp.so", libDir + "/libQnnSystem.so", skelDir)) {
    err_ = std::string("qnn init: ") + ffqnn::lastError();
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
  const std::vector<std::string> chain = ffqnn::tierChain(ffqnn::deviceInfo());
  tier_ = chain.front();
  for (const std::string& t : chain) {
    std::string probe = modelDir + "/yoloface_" + t + ".bin";
    if (FILE* f = std::fopen(probe.c_str(), "rb")) {
      std::fclose(f);
      tier_ = t;
      break;
    }
  }
  // No logging channel exists this early -- init() predates the app's log box. It is still
  // visible in a bug report, which prints pickTier's answer next to the files on disk: a
  // fallback reads as `tier v81` beside `yoloface_v73.bin`, which is the whole story.
  if (ffdebug() && tier_ != chain.front())
    fprintf(stderr, "[dbg] tier %s not present, using %s\n",
            chain.front().c_str(), tier_.c_str());

  auto open = [&](const char* name) -> ffqnn::Handle {
    std::string path = modelDir + "/" + name + "_" + tier_ + ".bin";
    ffqnn::Handle h = ffqnn::load(path);
    if (!h) err_ = std::string("load ") + path + ": " + ffqnn::lastError();
    return h;
  };
  p_->n.det = open("yoloface");     if (!p_->n.det) return false;
  p_->n.fan = open("fan2d");        if (!p_->n.fan) return false;
  p_->n.arc = open("arcface");      if (!p_->n.arc) return false;
  p_->n.swap = open(swapperName.c_str()); if (!p_->n.swap) return false;
  p_->n.fan685 = open("fan685");    // optional; absent -> geometric fallback below

  // The content gate is MANDATORY, because it blocks.  A gate that silently does not run
  // is worse than no gate: it reports "checked" to every caller above it.  So a missing
  // context is an init failure, not a fallback.
  //
  // Two names, in preference order.  fp32 is the shipping build and the one that tracks
  // the host, but it only finalizes on v79 -- below that the GELU cannot be created in
  // float and the quantised build is the only one that exists (docs/roadmap.md 2).
  p_->n.nsfw = open("nsfw");
  if (!p_->n.nsfw) {
    p_->n.nsfw = open("nsfwq");
    nsfwQuantised_ = p_->n.nsfw != nullptr;
  }
  if (!p_->n.nsfw) {
    err_ = "no content gate: neither nsfw_" + tier_ + ".bin nor nsfwq_" + tier_ +
           ".bin is in " + modelDir + " -- the gate blocks, so it cannot be skipped";
    return false;
  }
  return true;
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

  // content_analyser.py:prepare_detect_frame -- BGR -> RGB, /255, mean 0, std 1, NCHW.
  std::vector<float> in((size_t)3 * S * S, 0.f);
  for (int y = 0; y < nh; ++y) {
    const uint8_t* row = temp.row(y);
    for (int x = 0; x < nw; ++x)
      for (int c = 0; c < 3; ++c)
        in[(size_t)c * S * S + (size_t)(y + y0) * S + (x + x0)] =
            row[x * 3 + (2 - c)] / 255.0f;   // (2 - c): the [:, :, ::-1] flip
  }

  std::vector<std::vector<float>> out;
  if (!ffqnn::execute(p_->n.nsfw, {"input"}, {in.data()}, out)) {
    err_ = std::string("content gate: ") + ffqnn::lastError();
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
  ffcv::Image temp = (scale < 1.0) ? ffcv::resizeLinear(frame, tw, th) : frame;
  double ratioW = (double)frame.w / temp.w, ratioH = (double)frame.h / temp.h;

  std::vector<float> in((size_t)3 * S * S, 0.f);   // CHW, /255, zero-padded
  for (int y = 0; y < temp.h; ++y) {
    const uint8_t* row = temp.row(y);
    for (int x = 0; x < temp.w; ++x)
      for (int c = 0; c < 3; ++c)
        in[(size_t)c * S * S + (size_t)y * S + x] = row[x * 3 + c] / 255.0f;
  }
  msGeom += nowMs() - t0;

  t0 = nowMs();
  std::vector<std::vector<float>> out;
  if (!ffqnn::execute(p_->n.det, {"input"}, {in.data()}, out)) {
    err_ = std::string("detector: ") + ffqnn::lastError();
    return faces;
  }
  msDetect += nowMs() - t0;

  if (ffdebug()) {
    auto sh = ffqnn::outputShapes(p_->n.det);
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
            frame.w, frame.h, temp.w, temp.h, ratioW, ratioH);
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
      if (ffqnn::execute(p_->n.fan685, {"input"}, {f.landmark5}, o5) && o5[0].size() >= 136)
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

    std::vector<float> fin((size_t)3 * MS * MS);
    for (int y = 0; y < MS; ++y) {
      const uint8_t* row = crop.row(y);
      for (int x = 0; x < MS; ++x)
        for (int c = 0; c < 3; ++c)
          fin[(size_t)c * MS * MS + (size_t)y * MS + x] = row[x * 3 + c] / 255.0f;
    }
    msGeom += nowMs() - t0;

    t0 = nowMs();
    std::vector<std::vector<float>> hm;
    if (!ffqnn::execute(p_->n.fan, {"input"}, {fin.data()}, hm)) {
      err_ = std::string("landmarker: ") + ffqnn::lastError();
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
    ffcv::Image rc = ffcv::warpAffine(frame, am, RS, RS, ffcv::BORDER_REPLICATE);
    std::vector<float> rin((size_t)3 * RS * RS);
    for (int y = 0; y < RS; ++y) {
      const uint8_t* row = rc.row(y);
      for (int x = 0; x < RS; ++x)
        for (int c = 0; c < 3; ++c)     // BGR -> RGB is the [:, :, ::-1]
          rin[(size_t)(2 - c) * RS * RS + (size_t)y * RS + x] = row[x * 3 + c] / 127.5f - 1.0f;
    }
    msGeom += nowMs() - t0;

    t0 = nowMs();
    std::vector<std::vector<float>> emb;
    if (!ffqnn::execute(p_->n.arc, {"input"}, {rin.data()}, emb)) {
      err_ = std::string("recogniser: ") + ffqnn::lastError();
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
  msDetect = msLandmark = msRecognise = msGeom = 0;
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
  // larger than the graph and paste_back works on the crop.
  ffcv::MatF mask = ffcv::createBoxMask(CS, CS, cfg.maskBlur, cfg.maskPadding);

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
    ffcv::Image crop = ffcv::warpAffine(frame, am, CS, CS, ffcv::BORDER_REPLICATE);

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

      t0 = nowMs();
      std::vector<std::vector<float>> so;
      if (!ffqnn::execute(p_->n.swap, {"target", "source"}, {tin.data(), src.data()}, so)) {
        err_ = std::string("swapper: ") + ffqnn::lastError();
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
    }

    t0 = nowMs();
    ffcv::pasteBack(frame, outCrop, mask, am);
    msGeom += nowMs() - t0;
  }
  return true;
}

}  // namespace ffpipe
