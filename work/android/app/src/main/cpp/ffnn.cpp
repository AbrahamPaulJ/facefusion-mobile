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
      // ffnn_ncnn.cpp is not written yet. Answering "no" is the honest result and keeps the
      // seam usable meanwhile; it must never silently fall through to QNN, because a caller
      // that asked for a CPU backend on a part with no HTP would then get a working-looking
      // pipeline that cannot run.
      g_ready = false;
      return false;
  }
  return false;
}

Backend active() { return g_active; }

Handle open(const std::string& logicalName, Placement p) {
  // Placement is accepted and ignored on QNN rather than rejected: on this backend every
  // graph runs on the HTP, so "pin to CPU" is already satisfied in the only sense that
  // matters -- there is no less-safe unit to be pinned away from.
  (void)p;
  if (!g_ready) return nullptr;
  switch (g_active) {
    case Backend::Qnn: return qnnOpen(logicalName);
    case Backend::Auto:
    case Backend::Ncnn: return nullptr;
  }
  return nullptr;
}

void release(Handle h) {
  if (h) ffqnn::release(h);
}

bool execute(Handle h, const std::vector<std::string>& names,
             const std::vector<const float*>& data,
             std::vector<std::vector<float>>& outs) {
  return ffqnn::execute(h, names, data, outs);
}

std::vector<std::vector<int>> outputShapes(Handle h) { return ffqnn::outputShapes(h); }

const char* lastError() {
  switch (g_active) {
    case Backend::Qnn: return qnnLastError();
    case Backend::Auto:
    case Backend::Ncnn: return "ncnn backend not built";
  }
  return "unknown backend";
}

DeviceInfo deviceInfo(Backend b) {
  switch (b) {
    case Backend::Qnn: return qnnDeviceInfo();
    case Backend::Auto:
    case Backend::Ncnn: {
      DeviceInfo d;
      d.backend = Backend::Ncnn;
      d.name = "ncnn backend not built";
      return d;
    }
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
    case Backend::Auto:
    case Backend::Ncnn: return false;
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
