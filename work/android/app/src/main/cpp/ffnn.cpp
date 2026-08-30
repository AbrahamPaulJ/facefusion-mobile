// The ffnn dispatcher: one entry point per seam function, routed to the active backend.
//
// The routing is a switch and nothing more. Every backend-specific decision lives in that
// backend's file, so adding ncnn means adding ffnn_ncnn.cpp and one case here -- not
// touching ffpipe, which is the whole point of the seam.
#include "ffnn.h"

#include "ffqnn.h"

namespace ffnn {

// Implemented in ffnn_qnn.cpp.
bool qnnInit(const InitSpec&);
Handle qnnOpen(const std::string&);
const char* qnnLastError();
DeviceInfo qnnDeviceInfo();
const std::vector<std::string>& qnnChain();
bool qnnVariantPresent(const std::string&);
void qnnUseTier(const std::string&);
const std::string& qnnTier();

#ifdef FFNN_HAVE_NCNN
// Implemented in ffnn_ncnn.cpp.
bool ncnnInit(const InitSpec&);
Handle ncnnOpen(const std::string&, Placement);
void ncnnRelease(Handle);
bool ncnnExecute(Handle, const std::vector<std::string>&, const std::vector<const float*>&,
                 std::vector<std::vector<float>>&);
std::vector<std::vector<int>> ncnnOutputShapes(Handle);
const char* ncnnLastError();
bool ncnnVariantPresent(const std::string&);
DeviceInfo ncnnDeviceInfo();
#endif

namespace {
Backend g_active = Backend::Qnn;
bool g_ready = false;
}  // namespace

bool init(Backend b, const InitSpec& spec) {
  // Auto is the honest form of the question. Whether the HTP is usable cannot be known
  // before QNN is started, so "probe, then pick" cannot work -- an earlier draft of this
  // file had a preferredBackend() that probed first and reported no-NPU on every device,
  // including the one it was running on.
  if (b == Backend::Auto) {
    if (init(Backend::Qnn, spec)) return true;
    return init(Backend::Ncnn, spec);
  }
  g_active = b;
  switch (b) {
    case Backend::Qnn:
      g_ready = qnnInit(spec);
      return g_ready;
    case Backend::Auto:
      return false;   // handled above; here only to keep the switch exhaustive
    case Backend::Ncnn:
#ifdef FFNN_HAVE_NCNN
      g_ready = ncnnInit(spec);
#else
      // Built without ncnn. Answering "no" is honest and must never fall through to QNN: a
      // caller asking for a CPU backend on a part with no HTP would otherwise be handed a
      // working-looking pipeline that cannot run.
      g_ready = false;
#endif
      return g_ready;
  }
  return false;
}

Backend active() { return g_active; }

Handle open(const std::string& logicalName, Placement p) {
  // Placement is accepted and ignored on QNN rather than rejected: on this backend every
  // graph runs on the HTP, so "pin to CPU" is already satisfied in the only sense that
  // matters -- there is no less-safe unit to be pinned away from.
  if (!g_ready) return nullptr;
  switch (g_active) {
    case Backend::Qnn: return qnnOpen(logicalName);
#ifdef FFNN_HAVE_NCNN
    case Backend::Ncnn: return ncnnOpen(logicalName, p);
#else
    case Backend::Ncnn: return nullptr;
#endif
    case Backend::Auto: return nullptr;
  }
  return nullptr;
}

void release(Handle h) {
  if (!h) return;
#ifdef FFNN_HAVE_NCNN
  if (g_active == Backend::Ncnn) { ncnnRelease(h); return; }
#endif
  ffqnn::release(h);
}

bool execute(Handle h, const std::vector<std::string>& names,
             const std::vector<const float*>& data,
             std::vector<std::vector<float>>& outs) {
#ifdef FFNN_HAVE_NCNN
  if (g_active == Backend::Ncnn) return ncnnExecute(h, names, data, outs);
#endif
  return ffqnn::execute(h, names, data, outs);
}

std::vector<std::vector<int>> outputShapes(Handle h) {
#ifdef FFNN_HAVE_NCNN
  if (g_active == Backend::Ncnn) return ncnnOutputShapes(h);
#endif
  return ffqnn::outputShapes(h);
}

const char* lastError() {
  switch (g_active) {
    case Backend::Qnn: return qnnLastError();
#ifdef FFNN_HAVE_NCNN
    case Backend::Ncnn: return ncnnLastError();
#else
    case Backend::Ncnn: return "ncnn backend not built";
#endif
    case Backend::Auto: return "no backend started";
  }
  return "unknown backend";
}

DeviceInfo deviceInfo(Backend b) {
  switch (b) {
    case Backend::Qnn: return qnnDeviceInfo();
#ifdef FFNN_HAVE_NCNN
    case Backend::Ncnn: return ncnnDeviceInfo();
#else
    case Backend::Ncnn: {
      DeviceInfo d;
      d.backend = Backend::Ncnn;
      d.name = "ncnn backend not built";
      return d;
    }
#endif
    case Backend::Auto: return DeviceInfo();
  }
  return DeviceInfo();
}

std::vector<std::string> variantChain(Backend b) {
  switch (b) {
    case Backend::Qnn: return qnnChain();
    case Backend::Auto: return {};
    // One set of ncnn files runs on every part, so the chain is a single entry. It is NOT
    // an architecture and callers must not read it as one.
    case Backend::Ncnn: return {"ncnn"};
  }
  return {};
}

bool variantPresent(const std::string& v) {
  switch (g_active) {
    case Backend::Qnn: return qnnVariantPresent(v);
#ifdef FFNN_HAVE_NCNN
    case Backend::Ncnn: return ncnnVariantPresent(v);
#else
    case Backend::Ncnn: return false;
#endif
    case Backend::Auto: return false;
  }
  return false;
}

void useVariant(const std::string& v) {
  if (g_active == Backend::Qnn) qnnUseTier(v);
}

const std::string& variant() {
  static const std::string none;
  return g_active == Backend::Qnn ? qnnTier() : none;
}

}  // namespace ffnn
