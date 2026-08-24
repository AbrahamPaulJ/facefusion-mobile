// In-process QNN runner.
//
// WHY THIS EXISTS, and why the obvious approach does not work:
//
// The desktop harness drives every graph by exec'ing `qnn-net-run` over adb, and the first
// version of this app shipped that same binary inside the APK as `libqnnnetrun.so`. It runs
// -- it prints its build banner and reads its inputs -- and then dies in backend init with
//
//     QnnDsp <E> loadRemoteSymbols failed with err 4000
//     QnnDsp <E> Failed to load skel, error: 4000
//     Device Creation failure
//
// That is not a missing library and not a wrong ADSP_LIBRARY_PATH. Bisected on device, the
// byte-identical binary (same md5) with identical LD_LIBRARY_PATH, identical skel directory
// and identical CWD succeeds from /data/local/tmp and fails from the APK:
//
//     /data/local/tmp/nd/qnn-net-run          u:object_r:shell_data_file:s0   works
//     <apk>/lib/arm64/libqnnnetrun.so         u:object_r:apk_data_file:s0     fails
//
// The only variable is where the executable lives, so a process exec'd out of the APK is
// denied the Hexagon fastrpc device. Exec'ing a helper is therefore a dead end for an app
// no matter how the environment is arranged; the backend has to be dlopen'd INTO the app
// process, which already holds the DSP permission it needs.
//
// The rewrite pays for itself twice over: the context binary stays resident between calls
// instead of being re-read every inference. Loading mmdit_s2f is 1.58 GB off flash, and the
// autoregressive video loop visits all three MMDiT stages every unit.
//
// Quantization is handled here rather than in Kotlin because it has to be. qnn-net-run
// reads fp32 raws and converts them using each tensor's own encoding from the binary; doing
// it in-process means replicating that, and the direction is easy to get backwards -- QNN's
// convention is real = (quantized + offset) * scale with a NEGATIVE offset. Getting it wrong
// yields plausible-looking output rather than an error (trap #8 in the same family).

#include <android/log.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "QNN/QnnBackend.h"
#include "QNN/QnnCommon.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnDevice.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnInterface.h"
#include "QNN/QnnLog.h"
#include "QNN/QnnTensor.h"
#include "QNN/QnnTypes.h"
#include "QNN/HTP/QnnHtpDevice.h"
#include "QNN/HTP/QnnHtpPerfInfrastructure.h"
#include "QNN/System/QnnSystemContext.h"
#include "QNN/System/QnnSystemInterface.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "ffqnn", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "ffqnn", __VA_ARGS__)

namespace {

std::string g_err;  // last error, surfaced to Kotlin as an exception message

bool fail(const std::string& m) {
  g_err = m;
  LOGE("%s", m.c_str());
  return false;
}

// ---------------------------------------------------------------------------
// Tensor field access. Qnn_Tensor_t is a versioned union and the binaries this
// app loads are written by 2.49, but reading through the version tag rather than
// assuming v2 costs nothing and survives an SDK bump.
// ---------------------------------------------------------------------------

#define TF(t, field) \
  ((t).version == QNN_TENSOR_VERSION_1 ? (t).v1.field : (t).v2.field)

const char* tName(const Qnn_Tensor_t& t) { return TF(t, name); }
uint32_t tRank(const Qnn_Tensor_t& t) { return TF(t, rank); }
uint32_t* tDims(const Qnn_Tensor_t& t) { return TF(t, dimensions); }
Qnn_DataType_t tType(const Qnn_Tensor_t& t) { return TF(t, dataType); }
Qnn_QuantizeParams_t tQuant(const Qnn_Tensor_t& t) { return TF(t, quantizeParams); }

void tSetBuf(Qnn_Tensor_t& t, void* data, uint32_t bytes) {
  if (t.version == QNN_TENSOR_VERSION_1) {
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    t.v1.clientBuf = {data, bytes};
  } else {
    t.v2.memType = QNN_TENSORMEMTYPE_RAW;
    t.v2.clientBuf = {data, bytes};
  }
}

size_t elemSize(Qnn_DataType_t d) {
  switch (d) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:      return 1;
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16: return 2;
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32: return 4;
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
    case QNN_DATATYPE_FLOAT_64:    return 8;
    default:                        return 0;
  }
}

size_t numElems(const Qnn_Tensor_t& t) {
  size_t n = 1;
  uint32_t r = tRank(t);
  uint32_t* d = tDims(t);
  for (uint32_t i = 0; i < r; ++i) n *= d[i];
  return n;
}

// fp16 <-> fp32, needed because some graphs ship FLOAT_16 I/O.
float h2f(uint16_t h) {
  uint32_t s = (h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1F;
  uint32_t m = h & 0x3FF;
  uint32_t out;
  if (e == 0) {
    if (m == 0) { out = s; }
    else {                                  // subnormal -> normalise
      e = 127 - 15 + 1;
      while (!(m & 0x400)) { m <<= 1; --e; }
      m &= 0x3FF;
      out = s | (e << 23) | (m << 13);
    }
  } else if (e == 31) {
    out = s | 0x7F800000u | (m << 13);
  } else {
    out = s | ((e - 15 + 127) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &out, 4);
  return f;
}

uint16_t f2h(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t s = (x >> 16) & 0x8000u;
  int32_t e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t m = x & 0x7FFFFF;
  if (e <= 0) return (uint16_t)s;                        // flush subnormals to zero
  if (e >= 31) return (uint16_t)(s | 0x7C00u);           // saturate to inf
  return (uint16_t)(s | (e << 10) | (m >> 13));
}

// Per-tensor encoding. Per-CHANNEL (axis) encodings appear on weights, never on
// the graph I/O these models expose, so an axis encoding here means something is
// wrong and is reported rather than silently mis-scaled.
bool encodingOf(const Qnn_Tensor_t& t, double* scale, double* offset) {
  Qnn_QuantizeParams_t q = tQuant(t);
  if (q.encodingDefinition != QNN_DEFINITION_DEFINED) { *scale = 1.0; *offset = 0.0; return true; }
  if (q.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    *scale = q.scaleOffsetEncoding.scale;
    *offset = q.scaleOffsetEncoding.offset;
    return true;
  }
  if (q.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET) {
    *scale = q.bwScaleOffsetEncoding.scale;
    *offset = q.bwScaleOffsetEncoding.offset;
    return true;
  }
  return fail(std::string("tensor ") + tName(t) + ": unsupported quantization encoding " +
              std::to_string((int)q.quantizationEncoding));
}

// real -> device bytes.  QNN: real = (q + offset) * scale, offset negative.
bool writeTensor(const Qnn_Tensor_t& t, void* dst, const float* src, size_t n) {
  double sc, off;
  if (!encodingOf(t, &sc, &off)) return false;
  switch (tType(t)) {
    case QNN_DATATYPE_FLOAT_32:
      std::memcpy(dst, src, n * 4);
      return true;
    case QNN_DATATYPE_FLOAT_16: {
      auto* d = (uint16_t*)dst;
      for (size_t i = 0; i < n; ++i) d[i] = f2h(src[i]);
      return true;
    }
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32: {
      auto* d = (int32_t*)dst;
      for (size_t i = 0; i < n; ++i) d[i] = (int32_t)llrintf(src[i]);
      return true;
    }
    case QNN_DATATYPE_UFIXED_POINT_16: {
      auto* d = (uint16_t*)dst;
      for (size_t i = 0; i < n; ++i) {
        double q = std::nearbyint(src[i] / sc) - off;
        d[i] = (uint16_t)(q < 0 ? 0 : (q > 65535 ? 65535 : q));
      }
      return true;
    }
    case QNN_DATATYPE_UFIXED_POINT_8: {
      auto* d = (uint8_t*)dst;
      for (size_t i = 0; i < n; ++i) {
        double q = std::nearbyint(src[i] / sc) - off;
        d[i] = (uint8_t)(q < 0 ? 0 : (q > 255 ? 255 : q));
      }
      return true;
    }
    case QNN_DATATYPE_SFIXED_POINT_16: {
      auto* d = (int16_t*)dst;
      for (size_t i = 0; i < n; ++i) {
        double q = std::nearbyint(src[i] / sc) - off;
        d[i] = (int16_t)(q < -32768 ? -32768 : (q > 32767 ? 32767 : q));
      }
      return true;
    }
    case QNN_DATATYPE_SFIXED_POINT_8: {
      auto* d = (int8_t*)dst;
      for (size_t i = 0; i < n; ++i) {
        double q = std::nearbyint(src[i] / sc) - off;
        d[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
      }
      return true;
    }
    default:
      return fail(std::string("tensor ") + tName(t) + ": unsupported input dtype " +
                  std::to_string((int)tType(t)));
  }
}

// device bytes -> real
bool readTensor(const Qnn_Tensor_t& t, const void* src, float* dst, size_t n) {
  double sc, off;
  if (!encodingOf(t, &sc, &off)) return false;
  switch (tType(t)) {
    case QNN_DATATYPE_FLOAT_32:
      std::memcpy(dst, src, n * 4);
      return true;
    case QNN_DATATYPE_FLOAT_16: {
      auto* s = (const uint16_t*)src;
      for (size_t i = 0; i < n; ++i) dst[i] = h2f(s[i]);
      return true;
    }
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32: {
      auto* s = (const int32_t*)src;
      for (size_t i = 0; i < n; ++i) dst[i] = (float)s[i];
      return true;
    }
    case QNN_DATATYPE_UFIXED_POINT_16: {
      auto* s = (const uint16_t*)src;
      for (size_t i = 0; i < n; ++i) dst[i] = (float)((s[i] + off) * sc);
      return true;
    }
    case QNN_DATATYPE_UFIXED_POINT_8: {
      auto* s = (const uint8_t*)src;
      for (size_t i = 0; i < n; ++i) dst[i] = (float)((s[i] + off) * sc);
      return true;
    }
    case QNN_DATATYPE_SFIXED_POINT_16: {
      auto* s = (const int16_t*)src;
      for (size_t i = 0; i < n; ++i) dst[i] = (float)((s[i] + off) * sc);
      return true;
    }
    case QNN_DATATYPE_SFIXED_POINT_8: {
      auto* s = (const int8_t*)src;
      for (size_t i = 0; i < n; ++i) dst[i] = (float)((s[i] + off) * sc);
      return true;
    }
    default:
      return fail(std::string("tensor ") + tName(t) + ": unsupported output dtype " +
                  std::to_string((int)tType(t)));
  }
}

// ---------------------------------------------------------------------------
// Backend, held once for the process.
// ---------------------------------------------------------------------------

struct Backend {
  void* libBackend = nullptr;
  void* libSystem = nullptr;
  QNN_INTERFACE_VER_TYPE qnn{};
  QNN_SYSTEM_INTERFACE_VER_TYPE sys{};
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  Qnn_LogHandle_t log = nullptr;
  uint32_t powerId = 0;
  bool ready = false;
};

Backend g_be;
std::mutex g_mu;

// Forward QNN's own diagnostics to logcat. Without this the DSP's explanation of a
// failed deviceCreate ("Failed to load skel", "loadRemoteSymbols failed") is discarded
// and all that survives is an unhelpful error code.
void qnnLog(const char* fmt, QnnLog_Level_t level, uint64_t, va_list argp) {
  char buf[1024];
  vsnprintf(buf, sizeof(buf), fmt, argp);
  int prio = level == QNN_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR
           : level == QNN_LOG_LEVEL_WARN  ? ANDROID_LOG_WARN
                                          : ANDROID_LOG_INFO;
  __android_log_print(prio, "ffqnn.qnn", "%s", buf);
}

bool initBackend(const std::string& backendPath, const std::string& systemPath,
                 const std::string& skelDir) {
  if (g_be.ready) return true;

  // The Hexagon skel is found through ADSP_LIBRARY_PATH, which the fastrpc layer reads
  // when it is first loaded -- so this must happen BEFORE dlopen'ing the backend. The
  // exec'd version of this app got it from ProcessBuilder's environment; in-process
  // there is nothing to inherit it from. The vendor rfsa directories are kept on the
  // path so the platform's own stubs still resolve.
  std::string adsp = skelDir + ";/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp";
  setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1);
  LOGI("ADSP_LIBRARY_PATH=%s", adsp.c_str());

  g_be.libBackend = dlopen(backendPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!g_be.libBackend) return fail(std::string("dlopen backend: ") + dlerror());
  g_be.libSystem = dlopen(systemPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!g_be.libSystem) return fail(std::string("dlopen system: ") + dlerror());

  auto getProviders = (Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*))
      dlsym(g_be.libBackend, "QnnInterface_getProviders");
  if (!getProviders) return fail("QnnInterface_getProviders not found");

  const QnnInterface_t** providers = nullptr;
  uint32_t n = 0;
  if (getProviders(&providers, &n) != QNN_SUCCESS || n == 0)
    return fail("QnnInterface_getProviders returned none");
  g_be.qnn = providers[0]->QNN_INTERFACE_VER_NAME;

  auto getSysProviders = (Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*))
      dlsym(g_be.libSystem, "QnnSystemInterface_getProviders");
  if (!getSysProviders) return fail("QnnSystemInterface_getProviders not found");
  const QnnSystemInterface_t** sysProviders = nullptr;
  uint32_t sn = 0;
  if (getSysProviders(&sysProviders, &sn) != QNN_SUCCESS || sn == 0)
    return fail("QnnSystemInterface_getProviders returned none");
  g_be.sys = sysProviders[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

  if (g_be.qnn.logCreate) g_be.qnn.logCreate(qnnLog, QNN_LOG_LEVEL_WARN, &g_be.log);

  if (g_be.qnn.backendCreate(g_be.log, nullptr, &g_be.backend) != QNN_SUCCESS)
    return fail("backendCreate failed");

  // This is the call that failed with err 4000 when qnn-net-run was exec'd out of
  // the APK. In-process it succeeds, which is the whole point of this file.
  if (g_be.qnn.deviceCreate &&
      g_be.qnn.deviceCreate(g_be.log, nullptr, &g_be.device) != QNN_SUCCESS)
    return fail("deviceCreate failed -- the DSP is not reachable from this process");

  // Burst clocks. Without this the HTP ramps lazily and every measured latency in
  // docs/ (which were all taken with --perf_profile burst) is unreproducible here.
  QnnDevice_Infrastructure_t infra{};
  if (g_be.qnn.deviceGetInfrastructure &&
      g_be.qnn.deviceGetInfrastructure(&infra) == QNN_SUCCESS) {
    auto* htpInfra = (QnnHtpDevice_Infrastructure_t*)infra;
    if (htpInfra && htpInfra->infraType == QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
      auto& perf = htpInfra->perfInfra;
      if (perf.createPowerConfigId &&
          perf.createPowerConfigId(0, 0, &g_be.powerId) == QNN_SUCCESS) {
        QnnHtpPerfInfrastructure_PowerConfig_t dcvs{};
        dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        dcvs.dcvsV3Config.contextId = g_be.powerId;
        dcvs.dcvsV3Config.setDcvsEnable = 1;
        dcvs.dcvsV3Config.dcvsEnable = 0;              // pin, do not let DCVS drop us
        dcvs.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        dcvs.dcvsV3Config.setSleepLatency = 1;
        dcvs.dcvsV3Config.sleepLatency = 40;
        dcvs.dcvsV3Config.setBusParams = 1;
        dcvs.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.setCoreParams = 1;
        dcvs.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&dcvs, nullptr};
        if (perf.setPowerConfig) perf.setPowerConfig(g_be.powerId, cfgs);
      }
    }
  }

  g_be.ready = true;
  LOGI("QNN backend ready");
  return true;
}

// ---------------------------------------------------------------------------
// A loaded context binary, kept resident.
// ---------------------------------------------------------------------------

struct Model {
  Qnn_ContextHandle_t context = nullptr;
  Qnn_GraphHandle_t graph = nullptr;
  std::vector<Qnn_Tensor_t> inputs, outputs;      // shallow copies of binary metadata
  std::vector<std::vector<uint8_t>> inBuf, outBuf;
  QnnSystemContext_Handle_t sysCtx = nullptr;
};

bool bindTensors(Model* m, const QnnSystemContext_GraphInfo_t& gi) {
  const char* gname = nullptr;
  uint32_t nIn = 0, nOut = 0;
  Qnn_Tensor_t *ins = nullptr, *outs = nullptr;

  switch (gi.version) {
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:
      gname = gi.graphInfoV1.graphName; nIn = gi.graphInfoV1.numGraphInputs;
      ins = gi.graphInfoV1.graphInputs; nOut = gi.graphInfoV1.numGraphOutputs;
      outs = gi.graphInfoV1.graphOutputs; break;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:
      gname = gi.graphInfoV2.graphName; nIn = gi.graphInfoV2.numGraphInputs;
      ins = gi.graphInfoV2.graphInputs; nOut = gi.graphInfoV2.numGraphOutputs;
      outs = gi.graphInfoV2.graphOutputs; break;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:
      gname = gi.graphInfoV3.graphName; nIn = gi.graphInfoV3.numGraphInputs;
      ins = gi.graphInfoV3.graphInputs; nOut = gi.graphInfoV3.numGraphOutputs;
      outs = gi.graphInfoV3.graphOutputs; break;
    default:
      return fail("unsupported graph info version");
  }

  if (g_be.qnn.graphRetrieve(m->context, gname, &m->graph) != QNN_SUCCESS)
    return fail(std::string("graphRetrieve failed for ") + (gname ? gname : "?"));

  m->inputs.assign(ins, ins + nIn);
  m->outputs.assign(outs, outs + nOut);
  m->inBuf.resize(nIn);
  m->outBuf.resize(nOut);

  for (uint32_t i = 0; i < nIn; ++i) {
    size_t bytes = numElems(m->inputs[i]) * elemSize(tType(m->inputs[i]));
    if (!bytes) return fail(std::string("input ") + tName(m->inputs[i]) + ": zero size");
    m->inBuf[i].resize(bytes);
    tSetBuf(m->inputs[i], m->inBuf[i].data(), (uint32_t)bytes);
  }
  for (uint32_t i = 0; i < nOut; ++i) {
    size_t bytes = numElems(m->outputs[i]) * elemSize(tType(m->outputs[i]));
    if (!bytes) return fail(std::string("output ") + tName(m->outputs[i]) + ": zero size");
    m->outBuf[i].resize(bytes);
    tSetBuf(m->outputs[i], m->outBuf[i].data(), (uint32_t)bytes);
  }
  return true;
}

}  // namespace
// ---------------------------------------------------------------------------
// C++ API (see ffqnn.h).  Everything above this line is Neodragon's ndqnn.cpp core,
// unchanged apart from the log tag; the JNI layer it used to end with lives in ffjni.cpp
// so the same runner also serves the headless CLI.
// ---------------------------------------------------------------------------

#include "ffqnn.h"

namespace ffqnn {

const char* lastError() { return g_err.c_str(); }

bool init(const std::string& backendLib, const std::string& systemLib,
          const std::string& skelDir) {
  std::lock_guard<std::mutex> lk(g_mu);
  return initBackend(backendLib, systemLib, skelDir);
}

Handle load(const std::string& binPath) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!g_be.ready) { fail("ffqnn::init not called"); return nullptr; }

  // mmap, never a heap copy: a heap copy doubles peak footprint for the duration of the
  // load, which on an 11 GB phone drew lmkd and killed the activity mid-run (trap #35).
  int fd = open(binPath.c_str(), O_RDONLY);
  if (fd < 0) { fail("open " + binPath); return nullptr; }
  struct stat st{};
  if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); fail("stat " + binPath); return nullptr; }
  void* map = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) { fail("mmap " + binPath); return nullptr; }
  madvise(map, (size_t)st.st_size, MADV_SEQUENTIAL);

  auto* m = new Model();
  const QnnSystemContext_BinaryInfo_t* info = nullptr;
  Qnn_ContextBinarySize_t infoSize = 0;

  // The system context OWNS the tensor metadata -- every `name` and `dimensions` in the
  // Qnn_Tensor_t structs copied below is a POINTER into it.  So it must outlive the model:
  // freeing it here left every name dangling and execute() reported "no input named input"
  // for a graph whose input really is called `input`.  It is released in ffqnn::release.
  if (g_be.sys.systemContextCreate(&m->sysCtx) != QNN_SUCCESS) {
    munmap(map, (size_t)st.st_size); delete m; fail("systemContextCreate"); return nullptr;
  }
  if (g_be.sys.systemContextGetBinaryInfo(m->sysCtx, map, (uint64_t)st.st_size, &info,
                                          &infoSize) != QNN_SUCCESS || !info) {
    munmap(map, (size_t)st.st_size); delete m;
    fail("systemContextGetBinaryInfo"); return nullptr;
  }
  if (g_be.qnn.contextCreateFromBinary(g_be.backend, g_be.device, nullptr, map,
                                       (uint64_t)st.st_size, &m->context, nullptr) != QNN_SUCCESS) {
    munmap(map, (size_t)st.st_size); delete m;
    fail("contextCreateFromBinary " + binPath); return nullptr;
  }

  const QnnSystemContext_GraphInfo_t* graphs = nullptr;
  uint32_t nGraphs = 0;
  if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    graphs = info->contextBinaryInfoV1.graphs; nGraphs = info->contextBinaryInfoV1.numGraphs;
  } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    graphs = info->contextBinaryInfoV2.graphs; nGraphs = info->contextBinaryInfoV2.numGraphs;
  } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    graphs = info->contextBinaryInfoV3.graphs; nGraphs = info->contextBinaryInfoV3.numGraphs;
  }
  if (nGraphs != 1) {
    // Every binary this project converts holds exactly one graph; more than one means
    // the wrong file was pushed.
    munmap(map, (size_t)st.st_size); delete m;
    fail(binPath + ": expected 1 graph, found " + std::to_string(nGraphs));
    return nullptr;
  }
  bool ok = bindTensors(m, graphs[0]);

  // The context has deserialised into its own storage, so the MAPPING can go -- but the
  // system context cannot (see above).
  munmap(map, (size_t)st.st_size);
  if (!ok) { delete m; return nullptr; }
  LOGI("loaded %s (%.2f MB, %zu in, %zu out)", binPath.c_str(), st.st_size / 1048576.0,
       m->inputs.size(), m->outputs.size());
  return (Handle)m;
}

void release(Handle h) {
  if (!h) return;
  std::lock_guard<std::mutex> lk(g_mu);
  auto* m = (Model*)h;
  if (m->context) g_be.qnn.contextFree(m->context, nullptr);
  if (m->sysCtx) g_be.sys.systemContextFree(m->sysCtx);
  delete m;
}

static std::vector<std::string> namesOf(const std::vector<Qnn_Tensor_t>& v) {
  std::vector<std::string> out;
  out.reserve(v.size());
  for (const auto& t : v) out.emplace_back(tName(t));
  return out;
}

static std::vector<std::vector<int>> shapesOf(const std::vector<Qnn_Tensor_t>& v) {
  std::vector<std::vector<int>> out;
  for (const auto& t : v) {
    std::vector<int> d;
    for (uint32_t i = 0; i < tRank(t); ++i) d.push_back((int)tDims(t)[i]);
    out.push_back(std::move(d));
  }
  return out;
}

std::vector<std::string> inputNames(Handle h) { return namesOf(((Model*)h)->inputs); }
std::vector<std::string> outputNames(Handle h) { return namesOf(((Model*)h)->outputs); }
std::vector<std::vector<int>> inputShapes(Handle h) { return shapesOf(((Model*)h)->inputs); }
std::vector<std::vector<int>> outputShapes(Handle h) { return shapesOf(((Model*)h)->outputs); }

bool execute(Handle h, const std::vector<std::string>& names,
             const std::vector<const float*>& data,
             std::vector<std::vector<float>>& outs) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto* m = (Model*)h;
  if (!m || !m->graph) return fail("execute: model not loaded");
  if (names.size() != data.size()) return fail("execute: names/data length mismatch");

  for (size_t k = 0; k < names.size(); ++k) {
    int found = -1;
    for (size_t i = 0; i < m->inputs.size(); ++i)
      if (names[k] == tName(m->inputs[i])) { found = (int)i; break; }
    if (found < 0) return fail("execute: no input named " + names[k]);
    size_t n = numElems(m->inputs[found]);
    if (!writeTensor(m->inputs[found], m->inBuf[found].data(), data[k], n)) return false;
  }

  Qnn_ErrorHandle_t e = g_be.qnn.graphExecute(
      m->graph, m->inputs.data(), (uint32_t)m->inputs.size(),
      m->outputs.data(), (uint32_t)m->outputs.size(), nullptr, nullptr);
  if (e != QNN_SUCCESS) return fail("graphExecute failed: " + std::to_string((long long)e));

  outs.resize(m->outputs.size());
  for (size_t i = 0; i < m->outputs.size(); ++i) {
    size_t n = numElems(m->outputs[i]);
    outs[i].resize(n);
    if (!readTensor(m->outputs[i], m->outBuf[i].data(), outs[i].data(), n)) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Device support: measure two gates, provoke the third.
//
// Ported from ../LocalDream (DeviceProbe.hpp / Fp16Canary.hpp), which established both
// techniques.  The reasoning is theirs; only the plumbing is rewritten, because this app
// dlopens the backend itself (trap #33) instead of exec'ing a helper, so there is no
// process to run and no JSON to parse -- the answers come back as structs.
// ---------------------------------------------------------------------------

DeviceInfo deviceInfo() {
  std::lock_guard<std::mutex> lk(g_mu);
  DeviceInfo d;
  if (!g_be.ready) { fail("ffqnn::init not called"); return d; }
  if (!g_be.qnn.deviceGetPlatformInfo) {
    fail("backend exposes no deviceGetPlatformInfo");
    return d;
  }

  const QnnDevice_PlatformInfo_t* info = nullptr;
  if (g_be.qnn.deviceGetPlatformInfo(g_be.log, &info) != QNN_SUCCESS || !info) {
    fail("deviceGetPlatformInfo failed");
    return d;
  }
  if (info->version != QNN_DEVICE_PLATFORM_INFO_VERSION_1) {
    fail("unexpected platformInfo version");
    if (g_be.qnn.deviceFreePlatformInfo) g_be.qnn.deviceFreePlatformInfo(g_be.log, info);
    return d;
  }

  // Walk every hardware device rather than reading [0]. Every phone we have seen reports
  // one, but the API is a list and assuming otherwise is an assumption we cannot check.
  const QnnDevice_PlatformInfoV1_t& v1 = info->v1;
  for (uint32_t i = 0; i < v1.numHwDevices && !d.ok; ++i) {
    const QnnDevice_HardwareDeviceInfo_t& hw = v1.hwDevices[i];
    if (hw.version != QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1) continue;
    const auto* ext =
        (const QnnHtpDevice_DeviceInfoExtension_t*)hw.v1.deviceInfoExtension;
    if (!ext) continue;
    // An off-chip device carries no onChipDevice member; reading it would be reading a
    // different member of the union.
    if (ext->devType != QNN_HTP_DEVICE_TYPE_ON_CHIP) continue;

    const QnnHtpDevice_OnChipDeviceInfoExtension_t& chip = ext->onChipDevice;
    d.ok = true;
    d.deviceId = hw.v1.deviceId;
    d.socModel = chip.socModel;
    d.arch = (int)chip.arch;
    d.vtcmMb = chip.vtcmSize;
    d.signedPd = chip.signedPdSupport;
    d.dlbc = chip.dlbcSupport;
  }
  if (g_be.qnn.deviceFreePlatformInfo) g_be.qnn.deviceFreePlatformInfo(g_be.log, info);

  if (!d.ok) fail("platform info reported no on-chip HTP");
  else LOGI("HTP: arch v%d, vtcm %zu MB, soc_model %u, signed_pd %d, dlbc %d",
            d.arch, d.vtcmMb, d.socModel, (int)d.signedPd, (int)d.dlbc);
  return d;
}

std::string pickTier(const DeviceInfo& d) {
  // An unmeasured chip gets the most permissive build. v68 is the floor the SDK sets and
  // a v68 context runs FORWARD onto every later HTP, so being wrong here costs speed,
  // never a load failure -- and the alternative (guessing high) costs the whole app.
  if (!d.ok) return "v68";

  // VTCM first: the v69/v73/v79 configs all pin vtcm_mb 8, so a part with less rejects
  // them regardless of how new its arch is. Only the v68 build asks for 2.
  if (d.vtcmMb < 8) return "v68";

  // soc_model is baked into the v79 config (69 == SM8750) and pins that context to one
  // SoC, so a *newer* v79-or-above part must not be handed it -- it would be rejected for
  // the SoC, not the arch. v73 is the right forward-compatible answer there.
  if (d.arch >= 79 && d.socModel == 69) return "v79";
  if (d.arch >= 73) return "v73";

  // v69 deliberately falls through to v68. The v69 hyperswap context spills 70.0 MB --
  // MORE than the 2 MB v68 build's 41.8 MB, and more than v73's 31.6 MB. That is
  // unexplained (HANDOFF, session 3), so until someone measures a v69 part the v68 build
  // is both proven-forward-compatible and the smaller spill. Flip this line once v69 has
  // a number against it.
  return "v68";
}

// One context binary, loaded and immediately discarded. Stops at contextCreateFromBinary
// on purpose: ffqnn::load() goes on to bind tensor metadata, and a failure THERE would be
// indistinguishable from the rejection this is trying to measure.
static bool canaryLoads(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;
  struct stat st{};
  if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return false; }
  void* map = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) return false;

  Qnn_ContextHandle_t ctx = nullptr;
  bool ok = g_be.qnn.contextCreateFromBinary(g_be.backend, g_be.device, nullptr, map,
                                             (uint64_t)st.st_size, &ctx,
                                             nullptr) == QNN_SUCCESS;
  if (ok && ctx) g_be.qnn.contextFree(ctx, nullptr);
  munmap(map, (size_t)st.st_size);
  return ok;
}

Fp16 fp16Canary(const std::string& canaryDir) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!g_be.ready) { fail("ffqnn::init not called"); return Fp16::Unknown; }

  // A MISSING canary is not a rejected one. The ported version folded both into "did not
  // load", which makes an absent canary_249.bin -- an unpack that never finished, a push
  // that missed -- read as "this chip has no fp16", the one verdict that costs every user
  // the slower build. Check existence separately so only a real load failure can vote.
  const std::string p228 = canaryDir + "/canary_228.bin";
  const std::string p249 = canaryDir + "/canary_249.bin";
  for (const std::string& p : {p228, p249}) {
    struct stat st{};
    if (stat(p.c_str(), &st) != 0 || st.st_size <= 0) {
      fail("fp16 canary missing: " + p);
      return Fp16::Unknown;
    }
  }

  // The CONTROL goes first, and it is not optional. If the 2.28 canary cannot load then
  // nothing after it is interpretable -- no HTP, a backend that never came up -- and
  // reporting "no fp16" there would push every user of every working chip onto the
  // slower compatibility build.
  const bool c228 = canaryLoads(p228);
  const bool c249 = canaryLoads(p249);
  LOGI("fp16 canary: 2.28 control %s, 2.49 %s", c228 ? "loaded" : "FAILED",
       c249 ? "loaded" : "rejected");

  if (!c228) {
    fail("fp16 canary control failed -- verdict withheld");
    return Fp16::Unknown;
  }
  return c249 ? Fp16::Yes : Fp16::No;
}

}  // namespace ffqnn
