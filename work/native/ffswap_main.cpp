// Headless on-device driver for the FaceFusion swap chain.
//
// Runs from /data/local/tmp over adb with the phone's screen locked -- nothing here
// touches the display.  This is the harness the APK is validated against: it exercises
// exactly the same ffqnn + ffcv + ffpipe code the app will link.
//
// I/O is deliberately raw BGR, not JPEG/PNG: an image decoder is a dependency that adds
// nothing to what is being tested, and the host side already has cv2 to convert with.
//
//   ffswap --lib DIR --skel DIR --models DIR --swapper hyperswap
//          --source src.bgr --sw W --sh H
//          --target tgt.bgr --tw W --th H --out out.bgr
//          [--frames N]        # target file holds N frames back to back
//          [--weight 0.5]      # face_swapper_weight; >0.5 amplifies the source identity
//          [--blur 0.3] [--pad 0]        # face_mask_blur / face_mask_padding
//          [--boost 1]         # pixel boost: 1=256, 2=512 ... costs boost^2 runs per face
//          [--enhance]         # gpen_bfr_256 on the swapped crop; needs gpen_<tier>.bin
//          [--enhance-blend 0.8]
//          [--largest]         # swap only the largest face
//
//   ffswap --lib DIR --skel DIR --probe [--canary DIR]
//          # measure the HTP and exit: arch, VTCM, soc_model, the fp16 verdict and the
//          # context tier those imply.  Loads no model, needs no --models.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

#include "../android/app/src/main/cpp/ffpipe.h"
#include "../android/app/src/main/cpp/ffnn.h"
#include "../android/app/src/main/cpp/ffqnn.h"

static bool readAll(const char* path, std::vector<uint8_t>& out) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  out.resize((size_t)n);
  size_t got = fread(out.data(), 1, (size_t)n, f);
  fclose(f);
  return got == (size_t)n;
}

int main(int argc, char** argv) {
  std::string libDir, skelDir, modelDir, swapper = "hyperswap";
  std::string srcPath, tgtPath, outPath, canaryDir;
  int sw = 0, sh = 0, tw = 0, th = 0, frames = 1;
  bool probe = false;
  // The runtime knobs the app exposes, so a setting can be A/B'd headlessly before it is
  // trusted in the UI. Defaults match ffpipe::Config, i.e. FaceFusion's own.
  float weight = 0.5f, maskBlur = 0.3f;
  int pad = 0, boost = 1;
  bool largestOnly = false;
  bool enhance = false;
  float enhanceBlend = 0.8f;

  // The loop used to stop at argc-1 so a value-taking flag in last position could not read
  // past the end.  That also silently ignored any flag that appeared last -- fine while
  // every flag took a value, wrong the moment --probe did not.  Guard the fetch instead.
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto v = [&]() -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
    };
    auto num = [&]() -> int { return (i + 1 < argc) ? atoi(argv[++i]) : 0; };
    if (a == "--probe") probe = true;
    else if (a == "--lib") libDir = v();
    else if (a == "--skel") skelDir = v();
    else if (a == "--models") modelDir = v();
    else if (a == "--canary") canaryDir = v();
    else if (a == "--swapper") swapper = v();
    else if (a == "--source") srcPath = v();
    else if (a == "--target") tgtPath = v();
    else if (a == "--out") outPath = v();
    else if (a == "--sw") sw = num();
    else if (a == "--sh") sh = num();
    else if (a == "--tw") tw = num();
    else if (a == "--th") th = num();
    else if (a == "--frames") frames = num();
    else if (a == "--weight") weight = (float)atof(v().c_str());
    else if (a == "--blur") maskBlur = (float)atof(v().c_str());
    else if (a == "--pad") pad = num();
    else if (a == "--boost") boost = num();
    else if (a == "--largest") largestOnly = true;
    else if (a == "--enhance") enhance = true;
    else if (a == "--enhance-blend") enhanceBlend = (float)atof(v().c_str());
  }
  if (libDir.empty() || (!probe && (modelDir.empty() || srcPath.empty() || tgtPath.empty()))) {
    fprintf(stderr, "missing arguments; see the header of ffswap_main.cpp\n");
    return 2;
  }
  if (skelDir.empty()) skelDir = libDir;

  // --probe: measure the HTP and say which build this chip needs. Deliberately before
  // every --models check -- the whole point is to answer this on a phone with nothing
  // downloaded yet.
  if (probe) {
    if (!ffqnn::init(libDir + "/libQnnHtp.so", libDir + "/libQnnSystem.so", skelDir)) {
      fprintf(stderr, "qnn init: %s\n", ffqnn::lastError());
      return 1;
    }
    ffqnn::DeviceInfo d = ffqnn::deviceInfo();
    if (!d.ok) {
      printf("device    : UNMEASURED (%s)\n", ffqnn::lastError());
    } else {
      printf("device    : arch v%d, vtcm %zu MB, soc_model %u, signed_pd %d, dlbc %d\n",
             d.arch, d.vtcmMb, d.socModel, (int)d.signedPd, (int)d.dlbc);
    }
    if (canaryDir.empty() && !modelDir.empty()) canaryDir = modelDir + "/canary";
    if (!canaryDir.empty()) {
      switch (ffqnn::fp16Canary(canaryDir)) {
        case ffqnn::Fp16::Yes:
          printf("fp16      : YES -- 2.49 builds load here\n"); break;
        case ffqnn::Fp16::No:
          printf("fp16      : NO -- this chip needs the 2.28 compatibility build\n"); break;
        case ffqnn::Fp16::Unknown:
          // The control failed, so this says nothing about the chip. Never print it as a
          // negative: one broken unpack would otherwise condemn every working device.
          printf("fp16      : UNKNOWN -- control canary failed (%s)\n", ffqnn::lastError());
          break;
      }
    } else {
      printf("fp16      : not probed (pass --canary DIR)\n");
    }
    printf("tier      : %s\n", ffqnn::pickTier(d).c_str());

    // Every branch of pickTier except this phone's is unreachable here, and the chips it
    // decides for are exactly the ones we cannot borrow. So assert the mapping against a
    // table of synthetic devices instead: it cannot prove a v68 part loads the v68 build,
    // but it does prove the rule is the rule we wrote down, and it fails loudly if
    // someone edits pickTier without meaning to.
    struct Case { const char* what; ffqnn::DeviceInfo d; const char* want; };
    const Case cases[] = {
      {"SD 888        v68 / 2 MB", {true, 0, 0,  68, 2, false, false}, "v68"},
      {"8 Gen 1       v69 / 8 MB", {true, 0, 0,  69, 8, false, false}, "v68"},
      {"8 Gen 2       v73 / 8 MB", {true, 0, 0,  73, 8, false, false}, "v73"},
      {"8 Gen 3       v75 / 8 MB", {true, 0, 0,  75, 8, false, false}, "v73"},
      {"8 Elite       v79 / 8 MB", {true, 0, 69, 79, 8, false, false}, "v79"},
      {"v79, other SoC          ", {true, 0, 70, 79, 8, false, false}, "v73"},
      // Was "v73" until the v81 tier landed -- that fall-through is what sent every
      // S26 Ultra to a two-generation-old context.
      {"8 Elite Gen 5 v81 / 8 MB", {true, 0, 99, 81, 8, false, false}, "v81"},
      {"beyond v81    v85 / 8 MB", {true, 0, 99, 85, 8, false, false}, "v81"},
      {"v79 but only 4 MB VTCM  ", {true, 0, 69, 79, 4, false, false}, "v68"},
      {"probe failed            ", {},                                 "v68"},
    };
    int bad = 0;
    printf("\ntier table\n");
    for (const Case& c : cases) {
      std::string got = ffqnn::pickTier(c.d);
      bool ok = got == c.want;
      if (!ok) ++bad;
      printf("  %s -> %-4s %s\n", c.what, got.c_str(), ok ? "" : "MISMATCH");
    }

    // The chain, separately, because its subtle property is not what it STARTS with but
    // what it must never CONTAIN: v79 is pinned to soc_model 69, so offering it to a v81
    // part is not a fallback, it is a guaranteed rejection. ffpipe walks this list against
    // the filesystem, so a wrong entry here fails on a user's phone and nowhere else.
    struct ChainCase { const char* what; ffqnn::DeviceInfo d; const char* want; };
    const ChainCase chains[] = {
      {"8 Elite Gen 5 v81", {true, 0, 99, 81, 8, false, false}, "v81,v73,v68"},
      {"8 Elite       v79", {true, 0, 69, 79, 8, false, false}, "v79,v73,v68"},
      {"8 Gen 2       v73", {true, 0,  0, 73, 8, false, false}, "v73,v68"},
      {"SD 888        v68", {true, 0,  0, 68, 2, false, false}, "v68"},
      {"probe failed     ", {},                                 "v68"},
    };
    printf("\ntier chains (best first; ffpipe takes the first one present on disk)\n");
    for (const ChainCase& c : chains) {
      std::string got;
      for (const std::string& t : ffqnn::tierChain(c.d)) {
        if (!got.empty()) got += ",";
        got += t;
      }
      bool ok = got == c.want;
      if (!ok) ++bad;
      printf("  %s -> %-12s %s\n", c.what, got.c_str(), ok ? "" : "MISMATCH");
    }
    for (const ChainCase& c : chains) {
      for (const std::string& t : ffqnn::tierChain(c.d)) {
        if (c.d.arch >= 81 && t == "v79") {
          printf("  FATAL: v79 is soc-pinned and must never appear in a v81 chain\n");
          ++bad;
        }
      }
    }

    if (bad) printf("  %d MISMATCHES\n", bad);
    return (d.ok && !bad) ? 0 : 1;
  }

  ffpipe::Config cfg;
  cfg.swapperWeight = weight;
  cfg.maskBlur = maskBlur;
  for (int i = 0; i < 4; ++i) cfg.maskPadding[i] = pad;
  cfg.pixelBoost = boost;
  cfg.swapLargestOnly = largestOnly;
  // Requesting it is not having it: Pipeline skips the stage when gpen_<tier>.bin is
  // absent, and the banner prints what actually loaded rather than what was asked.
  cfg.faceEnhance = enhance;
  cfg.faceEnhancerBlend = enhanceBlend;
  if (swapper == "inswapper") {
    cfg.swapSize = 128; cfg.swapMean = 0.f; cfg.swapStd = 1.f;
    cfg.swapDenorm = false; cfg.swapperIsHyperswap = false;
  }

  // --list: print each binary's own tensor names and shapes.  Tensor names are matched
  // by NAME against the context binary's metadata, and the converter does not always keep
  // the ONNX name, so this is the first thing to check when a graph refuses its input.
  if (getenv("FFLIST")) {
    if (!ffqnn::init(libDir + "/libQnnHtp.so", libDir + "/libQnnSystem.so", skelDir)) {
      fprintf(stderr, "qnn init: %s\n", ffqnn::lastError());
      return 1;
    }
    const std::string tier = ffqnn::pickTier(ffqnn::deviceInfo());
    const char* names[] = {"yoloface", "fan2d", "arcface", "hyperswap", "inswapper", "fan685"};
    for (const char* nm : names) {
      std::string path = modelDir + "/" + nm + "_" + tier + ".bin";
      ffqnn::Handle h = ffqnn::load(path);
      if (!h) { printf("%-10s -- %s\n", nm, ffqnn::lastError()); continue; }
      auto in = ffqnn::inputNames(h), on = ffqnn::outputNames(h);
      auto is = ffqnn::inputShapes(h), os = ffqnn::outputShapes(h);
      printf("%-10s\n", nm);
      for (size_t i = 0; i < in.size(); ++i) {
        printf("   in  %-28s [", in[i].c_str());
        for (int d : is[i]) printf("%d,", d);
        printf("]\n");
      }
      for (size_t i = 0; i < on.size(); ++i) {
        printf("   out %-28s [", on[i].c_str());
        for (int d : os[i]) printf("%d,", d);
        printf("]\n");
      }
      ffqnn::release(h);
    }
    return 0;
  }

  ffpipe::Pipeline pipe;
  double t0 = (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now().time_since_epoch()).count();
  if (!pipe.init(libDir, skelDir, modelDir, swapper, cfg)) {
    fprintf(stderr, "init failed: %s\n", pipe.error().c_str());
    return 1;
  }
  double tInit = (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now().time_since_epoch()).count() - t0;
  printf("models loaded in %.0f ms (tier %s)\n", tInit, pipe.tier().c_str());
  printf("weight %.2f  blur %.2f  pad %d  boost %dx (%d px, %d run/face)%s\n",
         weight, maskBlur, pad, boost, 256 * boost, boost * boost,
         largestOnly ? "  largest face only" : "");
  // Asked-for is not loaded. Saying which avoids reading an unchanged image as
  // "the enhancer did nothing" when in fact it never ran.
  if (enhance)
    printf("enhancer: %s\n", pipe.hasEnhancer()
           ? "gpen_bfr_256"
           : "REQUESTED BUT NOT LOADED -- no gpen binary for this tier, stage skipped");

  std::vector<uint8_t> srcRaw, tgtRaw;
  if (!readAll(srcPath.c_str(), srcRaw)) { fprintf(stderr, "cannot read %s\n", srcPath.c_str()); return 1; }
  if (!readAll(tgtPath.c_str(), tgtRaw)) { fprintf(stderr, "cannot read %s\n", tgtPath.c_str()); return 1; }

  ffcv::Image src(sw, sh, 3);
  if (srcRaw.size() != src.data.size()) {
    fprintf(stderr, "source is %zu bytes, expected %zu for %dx%d\n",
            srcRaw.size(), src.data.size(), sw, sh);
    return 1;
  }
  std::memcpy(src.data.data(), srcRaw.data(), srcRaw.size());

  // The content gate, before anything else touches the pixels -- upstream checks the still
  // and the video before it processes either, and this port BLOCKS as upstream does.
  //
  // Printed with the score, not just the verdict: on SFW footage every frame lands ~-0.9
  // against a +0.25 threshold, and only the number shows how much margin the quantised
  // build's +0.087 bias is eating (docs/roadmap.md 2).
  printf("content gate: %s build\n", pipe.contentGateIsQuantised() ? "W8A16 (BIASED +0.087 "
         "toward blocking -- no fp32 context for this tier)" : "fp32");
  {
    ffpipe::ContentVerdict cv = pipe.checkContent(src);
    if (!cv.ok) { fprintf(stderr, "content gate: %s\n", pipe.error().c_str()); return 1; }
    printf("  source  score %+.4f  %s\n", cv.score, cv.blocked ? "BLOCKED" : "allow");
    if (cv.blocked) { fprintf(stderr, "refused: source image\n"); return 3; }
  }

  if (!pipe.setSource(src)) { fprintf(stderr, "source: %s\n", pipe.error().c_str()); return 1; }
  printf("source identity ready\n");

  size_t frameBytes = (size_t)tw * th * 3;
  if (tgtRaw.size() < frameBytes * (size_t)frames) {
    fprintf(stderr, "target holds %zu bytes, need %zu for %d frames of %dx%d\n",
            tgtRaw.size(), frameBytes * frames, frames, tw, th);
    return 1;
  }

  // content_analyser.py:analyse_video -- ONE frame per second, refuse above a 10% rate.
  // Sampling rather than per-frame is what makes this affordable: 11 calls for a 10 s clip
  // instead of 300.  With no container here, assume the project's 30 fps target.
  {
    const int kFps = 30, kRatePct = 10;
    int sampled = 0, flagged = 0;
    float worst = -1e9f;
    for (int i = 0; i < frames; i += kFps) {
      ffcv::Image frame(tw, th, 3);
      std::memcpy(frame.data.data(), tgtRaw.data() + frameBytes * i, frameBytes);
      ffpipe::ContentVerdict cv = pipe.checkContent(frame);
      if (!cv.ok) { fprintf(stderr, "content gate: %s\n", pipe.error().c_str()); return 1; }
      ++sampled;
      if (cv.blocked) ++flagged;
      if (cv.score > worst) worst = cv.score;
    }
    double rate = 100.0 * flagged / (sampled ? sampled : 1);
    printf("  target  %d sampled, %d flagged (%.1f%%), worst score %+.4f  %s\n",
           sampled, flagged, rate, worst, rate > kRatePct ? "BLOCKED" : "allow");
    if (rate > kRatePct) { fprintf(stderr, "refused: target video\n"); return 3; }
  }

  FILE* fo = outPath.empty() ? nullptr : fopen(outPath.c_str(), "wb");
  double tStart = (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now().time_since_epoch()).count();
  int totalFaces = 0;
  for (int i = 0; i < frames; ++i) {
    ffcv::Image frame(tw, th, 3);
    std::memcpy(frame.data.data(), tgtRaw.data() + frameBytes * i, frameBytes);
    auto faces = pipe.analyse(frame);
    totalFaces += (int)faces.size();
    if (!faces.empty() && !pipe.swapAll(frame, faces)) {
      fprintf(stderr, "swap: %s\n", pipe.error().c_str());
      return 1;
    }
    if (fo) fwrite(frame.data.data(), 1, frameBytes, fo);
    if (frames > 1 && (i + 1) % 10 == 0) { printf("  %d/%d\n", i + 1, frames); fflush(stdout); }
  }
  double tEnd = (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
  if (fo) fclose(fo);

  double wall = tEnd - tStart;
  printf("\n%d frames, %d faces, %.0f ms wall (%.1f ms/frame)\n",
         frames, totalFaces, wall, wall / std::max(1, frames));
  printf("  %-12s %8.1f ms\n", "detector", pipe.msDetect);
  printf("  %-12s %8.1f ms\n", "landmarker", pipe.msLandmark);
  printf("  %-12s %8.1f ms\n", "recogniser", pipe.msRecognise);
  printf("  %-12s %8.1f ms\n", "swapper", pipe.msSwap);
  if (pipe.msEnhance > 0)
    printf("  %-12s %8.1f ms\n", "enhancer", pipe.msEnhance);
  printf("  %-12s %8.1f ms  <- CPU geometry\n", "geometry", pipe.msGeom);
  // Itemised, because "geometry" is four unrelated pieces of work and optimising it
  // blind is how 0.4.26 removed a 786432-pixel conversion and moved nothing. Per FRAME,
  // unlike the totals above, since that is the number a target is set against.
  {
    const double n = std::max(1, frames);
    printf("    %-10s %8.2f ms/frame  detector letterbox + CHW\n",
           "detprep", pipe.msGeomDetPrep / n);
    printf("    %-10s %8.2f ms/frame  warpAffine to the crop\n",
           "warp", pipe.msGeomWarp / n);
    printf("    %-10s %8.2f ms/frame  crop -> model tensor\n",
           "tensor", pipe.msGeomTensor / n);
    printf("    %-10s %8.2f ms/frame  createBoxMask (cached)\n",
           "mask", pipe.msGeomMask / n);
    printf("    %-10s %8.2f ms/frame  pasteBack\n",
           "paste", pipe.msGeomPaste / n);
  }
  double npu = pipe.msDetect + pipe.msLandmark + pipe.msRecognise + pipe.msSwap +
               pipe.msEnhance;
  // "NPU" was hardcoded and printed over an ncnn run as `NPU 996.6 ms (94%)`, which is
  // exactly the kind of confident wrong label that costs an afternoon later.
  const char* unit = ffnn::active() == ffnn::Backend::Qnn ? "NPU" : "ncnn";
  printf("  %s %.1f ms (%.0f%%), CPU %.1f ms (%.0f%%)\n",
         unit, npu, 100 * npu / wall, pipe.msGeom, 100 * pipe.msGeom / wall);
  return 0;
}
