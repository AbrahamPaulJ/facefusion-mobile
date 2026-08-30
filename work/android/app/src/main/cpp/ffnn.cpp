// The ffnn dispatcher: one entry point per seam function, routed to the active backend.
//
// The routing is a switch and nothing more. Every backend-specific decision lives in that
// backend's file, so adding ncnn means adding ffnn_ncnn.cpp and one case here -- not
// touching ffpipe, which is the whole point of the seam.
#include "ffnn.h"

#include <cstdlib>
#include <cstring>

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

// ---------------------------------------------------------------------------
// A handle remembers WHICH BACKEND OPENED IT.
// ---------------------------------------------------------------------------
// `release`, `execute` and `outputShapes` used to dispatch on `g_active`, the backend
// active RIGHT NOW. That is correct exactly as long as the backend never changes while a
// handle is alive -- and 0.4.0 adds a developer switch that changes it, so it does not
// hold any more. A QNN handle released while Ncnn is active would have been handed to
// `delete static_cast<Model*>(h)`: a type-confused free of a QNN context, which is a crash
// with no useful stack anywhere near the switch that caused it.
//
// One pointer of overhead per model -- there are seven of them for a whole pipeline -- to
// make the wrong thing unrepresentable rather than merely documented.
struct HandleRec {
  Backend backend;
  Handle inner;
};
}  // namespace

bool init(Backend b, const InitSpec& spec) {
  // Auto is the honest form of the question. Whether the HTP is usable cannot be known
  // before QNN is started, so "probe, then pick" cannot work -- an earlier draft of this
  // file had a preferredBackend() that probed first and reported no-NPU on every device,
  // including the one it was running on.
  if (b == Backend::Auto) {
    // FFBACKEND=ncnn|qnn forces one, for TESTING on a device that has both. Without it the
    // non-Qualcomm path could only ever be exercised on a phone with no Hexagon, which is
    // not the bench -- and an untestable path is an unverified one.
    if (const char* forced = getenv("FFBACKEND")) {
      if (std::strcmp(forced, "ncnn") == 0) return init(Backend::Ncnn, spec);
      if (std::strcmp(forced, "qnn") == 0) return init(Backend::Qnn, spec);
    }
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
  Handle inner = nullptr;
  switch (g_active) {
    case Backend::Qnn: inner = qnnOpen(logicalName); break;
#ifdef FFNN_HAVE_NCNN
    case Backend::Ncnn: inner = ncnnOpen(logicalName, p); break;
#else
    case Backend::Ncnn: return nullptr;
#endif
    case Backend::Auto: return nullptr;
  }
  if (!inner) return nullptr;
  return new HandleRec{g_active, inner};
}

void release(Handle h) {
  if (!h) return;
  HandleRec* r = static_cast<HandleRec*>(h);
#ifdef FFNN_HAVE_NCNN
  if (r->backend == Backend::Ncnn) ncnnRelease(r->inner);
  else ffqnn::release(r->inner);
#else
  ffqnn::release(r->inner);
#endif
  delete r;
}

bool execute(Handle h, const std::vector<std::string>& names,
             const std::vector<const float*>& data,
             std::vector<std::vector<float>>& outs) {
  if (!h) return false;
  HandleRec* r = static_cast<HandleRec*>(h);
#ifdef FFNN_HAVE_NCNN
  if (r->backend == Backend::Ncnn) return ncnnExecute(r->inner, names, data, outs);
#endif
  return ffqnn::execute(r->inner, names, data, outs);
}

std::vector<std::vector<int>> outputShapes(Handle h) {
  if (!h) return {};
  HandleRec* r = static_cast<HandleRec*>(h);
#ifdef FFNN_HAVE_NCNN
  if (r->backend == Backend::Ncnn) return ncnnOutputShapes(r->inner);
#endif
  return ffqnn::outputShapes(r->inner);
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
