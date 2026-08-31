// The neural-runtime seam.  One interface, two backends behind it.
//
// `ffpipe.cpp` is the geometry and the swap chain; it should not know whether a graph runs
// on the Hexagon NPU through QNN or on the CPU/GPU through ncnn.  Until this header existed
// it called `ffqnn::` directly in 24 places and resolved context filenames itself, which
// meant every non-Qualcomm ambition had to start by rewriting the pipeline.
//
// EIGHT functions, and the shape of `open` is the important one.
//
// ---------------------------------------------------------------------------
// Why open() takes a LOGICAL NAME
// ---------------------------------------------------------------------------
// The two backends do not name files alike, and not merely cosmetically:
//
//     QNN    yoloface_v79.bin              one file, arch tier in the name
//     ncnn   yoloface_8n_b1.ncnn.param     TWO files, no tier, and the graph's own name
//            yoloface_8n_b1.ncnn.bin
//
// A caller that builds paths cannot be backend-neutral, so it does not build them: it asks
// for "yoloface" and the backend resolves its own. That also moves the arch-tier logic out
// of ffpipe entirely -- tiers are a QNN concept and mean nothing to ncnn.
//
// ---------------------------------------------------------------------------
// Why PLACEMENT is per model
// ---------------------------------------------------------------------------
// Measured 2026-08-30, and this is not a preference:
//
//   * The CONTENT GATE must not run on the GPU. ncnn's Vulkan moves its decision statistic
//     by -0.106 mean / -0.175 max, AWAY from flagging, against a 0.25 threshold -- it errs
//     toward allowing, which is the one direction a gate must not err. On MNN's GPU paths it
//     is also the slowest model measured. It runs ~11 times per video, so CPU costs nothing.
//   * The ENHANCER cannot run on the GPU at all. gpen fails on ncnn Vulkan (10.86 dB, wrong
//     even with every fp16 path disabled), MNN OpenCL (segfault) and MNN Vulkan (segfault).
//     Three independent implementations, one StyleGAN generator.
//
// So a single global "use the GPU" switch would be wrong for two of six graphs. Placement
// belongs to the model, not to the session.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ffnn {

enum class Backend {
  Qnn,    // Hexagon NPU via QAIRT context binaries
  Ncnn,   // CPU (NEON) or GPU (Vulkan), for parts with no Qualcomm NPU
  Auto,   // try Qnn, fall back to Ncnn -- see init()
};

// Where a MODEL should run. Meaningful only for backends that have a choice: QNN always
// means the HTP, so everything but Default is ignored there rather than failing -- a caller
// asking for Cpu on the NPU path wants "the safe placement", and on QNN that IS the HTP.
enum class Placement {
  Default,  // the backend's preferred unit
  Cpu,      // pin to CPU -- the gate and the enhancer, for the reasons above
  Gpu,      // prefer the GPU
};

struct InitSpec {
  std::string libDir;     // QNN: where libQnnHtp.so lives.  ncnn: unused.
  std::string skelDir;    // QNN: the DSP skels, exported before the backend is dlopen'd.
  std::string modelDir;   // both: where the model files are.
};

// Choose and start a backend. One per process, as ffqnn already required.
//
// ⚠ Prefer `Backend::Auto`. Whether this part has a usable HTP cannot be answered WITHOUT
// starting QNN -- `QnnDevice_getPlatformInfo` needs the backend dlopen'd and the skels on
// ADSP_LIBRARY_PATH -- so any "ask first, then choose" API is a lie that reports no-NPU on
// every device. Auto tries Qnn and falls back to Ncnn, which is the only order that can
// actually tell them apart. `active()` reports which one won.
bool init(Backend b, const InitSpec& spec);
Backend active();

using Handle = void*;

// Open a model BY LOGICAL NAME -- "yoloface", "fan2d", "arcface", "hyperswap",
// "inswapper", "gpen", "nsfw". The backend resolves the filename, including the arch tier
// where that concept exists. Returns nullptr and sets lastError() when it cannot.
Handle open(const std::string& logicalName, Placement p = Placement::Default);
void release(Handle h);

// Float in, float out. Inputs are matched BY NAME against the graph's own metadata, so a
// converter that reordered the inputs cannot be fed the wrong tensor -- the bug this
// signature exists to make impossible.
bool execute(Handle h, const std::vector<std::string>& names,
             const std::vector<const float*>& data,
             std::vector<std::vector<float>>& outs);

std::vector<std::vector<int>> outputShapes(Handle h);

// What this graph WANTS, asked of the graph rather than assumed by the caller.
//
// Exists for the tier probe: proving a context executes means feeding it something, and a
// prober that hardcoded "yoloface is 640x640x3" would be a second copy of every shape in
// the app, wrong the moment a conversion changes one. Asking the model is the only form
// that cannot drift.
std::vector<std::string> inputNames(Handle h);
std::vector<std::vector<int>> inputShapes(Handle h);

const char* lastError();

// ---------------------------------------------------------------------------
// What this device can run, asked before any model is opened.
// ---------------------------------------------------------------------------

struct DeviceInfo {
  bool ok = false;          // false means the probe FAILED, not that the part is old
  Backend backend = Backend::Qnn;
  int arch = 0;             // QNN: the bare Hexagon V-number.  ncnn: 0.
  size_t vtcmMb = 0;        // QNN only
  bool gpu = false;         // ncnn: a usable Vulkan device is present
  std::string name;         // human-readable, for the settings panel and bug reports
};

DeviceInfo deviceInfo(Backend b);

// The variants this device can load, best first. QNN returns the arch tier chain
// ("v81","v73","v68"); ncnn returns a single entry, because it has no tiers -- one set of
// files runs everywhere. Callers must not assume the strings mean an architecture.
std::vector<std::string> variantChain(Backend b);

// Is this variant's model set actually on disk?
//
// The caller must not answer this itself: "is v73 present" is `yoloface_v73.bin` on QNN and
// a `.ncnn.param`/`.ncnn.bin` PAIR on ncnn, and a pipeline that knows that is not backend
// neutral. Asking the backend keeps the one thing the seam exists to hide -- filenames --
// on the far side of it.
bool variantPresent(const std::string& v);

// Commit to a variant. Selection stays with the CALLER because only the caller can tell
// whether a variant is good: that takes executing a model, and ffpipe is what can do it.
// The seam supplies the candidates and resolves the filenames; it does not choose.
void useVariant(const std::string& v);
const std::string& variant();

}  // namespace ffnn
