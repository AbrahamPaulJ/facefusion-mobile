// C++ API over the in-process QNN runner.
//
// The runner core in ffqnn.cpp is Neodragon's `ndqnn.cpp` unchanged -- see its header for
// why an app must dlopen the backend rather than exec a helper (trap #33), and why the
// quantisation direction (real = (q + offset) * scale, offset NEGATIVE) has to be handled
// here rather than in Kotlin.
//
// This header exists so the same runner serves two callers: the on-device CLI, which is
// how everything gets tested headlessly over adb, and the APK's JNI layer.

#pragma once
#include <string>
#include <vector>

namespace ffqnn {

// Load the backend once per process.  skelDir must contain libQnnHtpV79Skel.so; it is
// exported as ADSP_LIBRARY_PATH *before* the backend is dlopen'd, because fastrpc reads
// that variable when it is first loaded.
bool init(const std::string& backendLib, const std::string& systemLib,
          const std::string& skelDir);

// A context binary, kept resident.  Loading is the expensive part (76 ms for hyperswap's
// 196 MB), so models are loaded once and reused across frames.
using Handle = void*;

Handle load(const std::string& binPath);
void release(Handle h);

std::vector<std::string> inputNames(Handle h);
std::vector<std::string> outputNames(Handle h);
std::vector<std::vector<int>> inputShapes(Handle h);
std::vector<std::vector<int>> outputShapes(Handle h);

// Execute with float I/O.  Inputs are matched BY NAME against the binary's own metadata,
// so a graph whose inputs the converter reordered cannot be fed the wrong tensor.
// `outs` is resized to the graph's outputs, each dequantised to float.
bool execute(Handle h, const std::vector<std::string>& names,
             const std::vector<const float*>& data,
             std::vector<std::vector<float>>& outs);

const char* lastError();

// ---------------------------------------------------------------------------
// Device support.  Which context binary does THIS chip need?
//
// Three independent gates decide whether a context loads, and the device rejects the
// first one it lacks (docs/roadmap.md 1.3):
//   arch  -- a v79 context does not run on a v73 chip
//   VTCM  -- our v69/v73/v79 configs pin vtcm_mb 8; a v68 part has 2
//   fp16  -- every QAIRT 2.49 build declares it; only a 2.28 build does not
//
// The first two can be MEASURED with no model at all.  The third cannot be read from
// anywhere -- not the platform info, not the context binary -- so it has to be provoked.
// ---------------------------------------------------------------------------

struct DeviceInfo {
  bool ok = false;         // false means the probe failed, NOT that the chip is old
  uint32_t deviceId = 0;
  uint32_t socModel = 0;   // 69 == SM8750, the S25 Ultra
  int arch = 0;            // the bare V-number: 68 / 69 / 73 / 75 / 79 / 81
  size_t vtcmMb = 0;
  bool signedPd = false, dlbc = false;
};

// Reads the HTP through QnnDevice_getPlatformInfo.  Requires init(); loads no model.
DeviceInfo deviceInfo();

// The arch suffix of the context binaries this chip should load -- "v68" ... "v79", i.e.
// the `<name>_<tier>.bin` that ffpipe opens.  Falls back to the most permissive tier
// whenever the probe could not measure, so an unknown chip behaves like an old one.
std::string pickTier(const DeviceInfo& d);

enum class Fp16 { Unknown, Yes, No };

// PROVOKES the fp16 gate with two ~50 KB context binaries: `canary_249.bin` declares the
// fp16 requirement, `canary_228.bin` is the control that does not.  Both are built v68 /
// 2 MB so arch and VTCM cannot be what fails.
//
// Yes     -- the 2.49 canary loaded; the shipping binaries are safe here.
// No      -- the control loaded and the 2.49 canary did not; this chip needs 2.28 builds.
// Unknown -- the CONTROL failed, so the probe is broken and the chip said nothing.
//            Never report Unknown as "no fp16".
Fp16 fp16Canary(const std::string& canaryDir);

}  // namespace ffqnn
