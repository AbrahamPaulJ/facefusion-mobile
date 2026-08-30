// The QNN side of the ffnn seam.
//
// Thin on purpose: ffqnn.cpp is the runner that has been measured, shipped and debugged for
// months, and this file must not become a second place where QNN behaviour lives. What it
// owns is exactly what the seam moved out of ffpipe -- resolving a LOGICAL name to a
// context binary, including the arch tier, and remembering which tier answered.
#include "ffnn.h"

#include <cstdio>
#include <string>

#include "ffqnn.h"

namespace ffnn {
namespace {
InitSpec g_spec;
std::string g_tier;
std::vector<std::string> g_chain;
std::string g_err;
}  // namespace

bool qnnInit(const InitSpec& spec) {
  g_spec = spec;
  if (!ffqnn::init(spec.libDir + "/libQnnHtp.so", spec.libDir + "/libQnnSystem.so",
                   spec.skelDir)) {
    g_err = std::string("qnn init: ") + ffqnn::lastError();
    return false;
  }
  g_chain = ffqnn::tierChain(ffqnn::deviceInfo());
  g_tier = g_chain.empty() ? std::string() : g_chain.front();
  return !g_tier.empty();
}

// The tier a caller has settled on. `ffpipe` walks the chain itself -- it is the only place
// that knows a tier is good, because only it can execute the gate to find out -- so the
// seam takes the answer rather than guessing at it.
void qnnUseTier(const std::string& t) { g_tier = t; }
const std::string& qnnTier() { return g_tier; }
const std::vector<std::string>& qnnChain() { return g_chain; }

// The detector is the probe: mandatory, and the smallest file in a tier at 3.8 MB. A tier
// is never half-present -- the downloader writes a `.part` and renames only after the
// SHA256 matches -- so yoloface existing means the rest of that tier does too.
bool qnnVariantPresent(const std::string& v) {
  std::string probe = g_spec.modelDir + "/yoloface_" + v + ".bin";
  if (FILE* f = std::fopen(probe.c_str(), "rb")) { std::fclose(f); return true; }
  return false;
}

Handle qnnOpen(const std::string& logicalName) {
  std::string path = g_spec.modelDir + "/" + logicalName + "_" + g_tier + ".bin";
  Handle h = ffqnn::load(path);
  if (!h) g_err = std::string("load ") + path + ": " + ffqnn::lastError();
  return h;
}

const char* qnnLastError() { return g_err.empty() ? ffqnn::lastError() : g_err.c_str(); }

DeviceInfo qnnDeviceInfo() {
  ffqnn::DeviceInfo d = ffqnn::deviceInfo();
  DeviceInfo out;
  out.ok = d.ok;
  out.backend = Backend::Qnn;
  out.arch = d.arch;
  out.vtcmMb = d.vtcmMb;
  out.gpu = false;   // the HTP is not a GPU; a caller asking about Vulkan means ncnn
  char buf[64];
  std::snprintf(buf, sizeof(buf), "Hexagon v%d, %zu MB VTCM", d.arch, d.vtcmMb);
  out.name = d.ok ? buf : "HTP probe failed";
  return out;
}

}  // namespace ffnn
