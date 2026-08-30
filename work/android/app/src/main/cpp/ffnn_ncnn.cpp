// The ncnn side of the ffnn seam: CPU (NEON) or GPU (Vulkan), for parts with no Hexagon.
//
// Built only when FFNN_HAVE_NCNN is defined, so the QNN-only app and CLI still build with no
// ncnn checkout present. Without it the dispatcher's Ncnn case returns false, which is the
// honest answer rather than a silent fall-through to QNN.
#include "ffnn.h"

#ifdef FFNN_HAVE_NCNN

#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "net.h"

namespace ffnn {
namespace {

// ---------------------------------------------------------------------------
// The name mapping, which is the whole reason this belongs in the BACKEND.
// ---------------------------------------------------------------------------
// pnnx names blobs positionally -- `in0`, `in1`, `out0` -- and discards the ONNX names the
// pipeline uses. So the backend has to know that hyperswap's "source" is in0 and its
// "target" is in1, and it has to know the file stems, which are the ONNX filenames rather
// than the pipeline's logical names.
//
// ⚠ hyperswap's input ORDER is source THEN target, from the ONNX graph. Reversing it
// produces a model that converts cleanly, runs, and computes nonsense -- the failure
// work/ncnn/convert_ncnn.sh already warns about. `ins` is in ncnn BLOB order, so its index
// IS the in<N> number, and that is the only place the order is written down.
// An input's shape, because `execute` is handed a bare `const float*` with no length --
// the caller cannot tell the backend how big a tensor is, so the backend has to know.
// c == 0 means a 1-D tensor of `w` floats (hyperswap's 512-d embedding).
struct InSpec {
  const char* name;   // the ONNX name ffpipe passes
  int w, h, c;
};

struct Spec {
  const char* stem;
  std::vector<InSpec> ins;   // in in0/in1 order
  const char* out;
};

const std::map<std::string, Spec>& specs() {
  static const std::map<std::string, Spec> m = {
      {"yoloface",  {"yoloface_8n_b1",        {{"input", 640, 640, 3}}, "out0"}},
      {"fan2d",     {"2dfan4_heatmaps",       {{"input", 256, 256, 3}}, "out0"}},
      {"arcface",   {"arcface_w600k_r50_b1",  {{"input", 112, 112, 3}}, "out0"}},
      // source FIRST -- in0 is the 512-d embedding, in1 the image. See the warning above.
      {"hyperswap", {"hyperswap_1a_256_fp32",
                     {{"source", 512, 1, 0}, {"target", 256, 256, 3}}, "out0"}},
      {"gpen",      {"gpen_ncnn",             {{"input", 256, 256, 3}}, "out0"}},
      // One graph serves both gate names. ncnn has no quantised build -- "nsfwq2" exists
      // because a QNN tier below v79 cannot finalize the fp32 gate, which is a QNN fact.
      {"nsfw",      {"nsfw_2_sim",            {{"input", 384, 384, 3}}, "out0"}},
      {"nsfwq2",    {"nsfw_2_sim",            {{"input", 384, 384, 3}}, "out0"}},
      // fan685 is deliberately absent: it is not converted for ncnn, and the pipeline
      // already treats a missing landmark refiner as optional.
  };
  return m;
}

struct Model {
  ncnn::Net net;
  const Spec* spec = nullptr;
  bool gpu = false;
};

InitSpec g_spec;
std::string g_err;
bool g_vulkan = false;

}  // namespace

bool ncnnInit(const InitSpec& spec) {
  g_spec = spec;
#if NCNN_VULKAN
  // Counted once. ncnn creates the instance lazily otherwise, and a device with no usable
  // Vulkan must be discovered here rather than at the first GPU model open.
  g_vulkan = ncnn::get_gpu_count() > 0;
#endif
  return true;
}

Handle ncnnOpen(const std::string& logicalName, Placement p) {
  auto it = specs().find(logicalName);
  if (it == specs().end()) {
    g_err = "ncnn: no model named " + logicalName;
    return nullptr;
  }
  std::unique_ptr<Model> m(new Model());
  m->spec = &it->second;

  // Placement, honoured rather than noted. Cpu is not a hint: the content gate and the
  // enhancer are pinned there because the GPU gets them WRONG, not because it is slower.
  //
  // ⚠ Default means CPU. That is deliberate but it is also easy to misread: the first
  // end-to-end ncnn run measured 354 ms/frame and was reported as "the ncnn path", when
  // every model in it was on the CPU because none had asked for Gpu. Nothing was wrong;
  // the number simply did not mean what it looked like.
  //
  // FFNCNN_PLACE=cpu|gpu overrides Default for MEASUREMENT, so one run gives a whole
  // stage table for each unit. It never overrides an explicit Cpu: those two are
  // correctness decisions and must not be movable by an environment variable.
  // Default PREFERS the GPU, measured per stage over 6 frames (ms/frame):
  //
  //     detector    CPU 38.4   GPU 20.6   1.9x
  //     landmarker  CPU 167.3  GPU 74.2   2.3x
  //     recogniser  CPU 22.7   GPU 19.7   1.2x
  //     swapper     CPU 285.4  GPU 182.2  1.6x
  //
  // The GPU wins every one, so Default meaning CPU would have shipped the slow half of a
  // path that is already 19x off the NPU. The gap also GROWS with clip length: the same
  // CPU run over 3 frames measured 354 ms/frame and over 6 measured 539, which is the
  // sustained-load throttling roadmap 6 already found (+40% avg, +142% worst). The GPU
  // stays flat, so on a real 300-frame clip this is worth more than the table shows.
  bool wantGpu = (p != Placement::Cpu);
  if (p == Placement::Default) {
    // Measurement escape hatch. It never overrides an explicit Cpu -- those two are
    // correctness decisions and must not be movable by an environment variable.
    const char* force = getenv("FFNCNN_PLACE");
    if (force) wantGpu = strcmp(force, "gpu") == 0;
  }
  m->gpu = wantGpu && g_vulkan;
#if NCNN_VULKAN
  m->net.opt.use_vulkan_compute = m->gpu;
#endif
  m->net.opt.lightmode = true;
  m->net.opt.num_threads = 4;
  m->net.opt.use_packing_layout = true;
  m->net.opt.use_fp16_packed = true;
  m->net.opt.use_fp16_storage = true;
  m->net.opt.use_fp16_arithmetic = true;

  const std::string base = g_spec.modelDir + "/" + it->second.stem + ".ncnn.";
  if (m->net.load_param((base + "param").c_str()) != 0) {
    g_err = "ncnn: load_param failed for " + base + "param";
    return nullptr;
  }
  if (m->net.load_model((base + "bin").c_str()) != 0) {
    g_err = "ncnn: load_model failed for " + base + "bin";
    return nullptr;
  }
  return m.release();
}

void ncnnRelease(Handle h) { delete static_cast<Model*>(h); }

bool ncnnExecute(Handle h, const std::vector<std::string>& names,
                 const std::vector<const float*>& data,
                 std::vector<std::vector<float>>& outs) {
  Model* m = static_cast<Model*>(h);
  if (!m || names.size() != data.size()) {
    g_err = "ncnn: bad execute arguments";
    return false;
  }
  ncnn::Extractor ex = m->net.create_extractor();

  // BY NAME, never by position. The caller passes {"target","source"} and this graph wants
  // source in in0 -- feeding them in the order given would run the swapper with the
  // embedding as the image and produce a plausible-looking wrong answer.
  for (size_t i = 0; i < names.size(); ++i) {
    const auto& want = m->spec->ins;
    size_t idx = want.size();
    for (size_t k = 0; k < want.size(); ++k)
      if (names[i] == want[k].name) { idx = k; break; }
    if (idx == want.size()) {
      g_err = "ncnn: " + std::string(m->spec->stem) + " has no input named " + names[i];
      return false;
    }
    const InSpec& sh = want[idx];
    // Wrapping the caller's buffer, not copying it: ffpipe owns it for the call's duration
    // and every one of these is megabytes.
    ncnn::Mat in = sh.c == 0 ? ncnn::Mat(sh.w, (void*)data[i])
                             : ncnn::Mat(sh.w, sh.h, sh.c, (void*)data[i]);
    char blob[8];
    snprintf(blob, sizeof(blob), "in%zu", idx);
    if (ex.input(blob, in) != 0) {
      g_err = "ncnn: input " + std::string(blob) + " rejected";
      return false;
    }
  }

  ncnn::Mat out;
  if (ex.extract(m->spec->out, out) != 0) {
    g_err = "ncnn: extract " + std::string(m->spec->out) + " failed";
    return false;
  }
  outs.assign(1, std::vector<float>());
  outs[0].resize((size_t)out.w * out.h * out.c);
  // Channel by channel: ncnn pads each channel to its alignment, so a flat copy of
  // `out.data` would interleave the padding into the tensor.
  size_t n = (size_t)out.w * out.h;
  for (int q = 0; q < out.c; ++q)
    memcpy(outs[0].data() + (size_t)q * n, out.channel(q), n * sizeof(float));
  return true;
}

std::vector<std::vector<int>> ncnnOutputShapes(Handle h) {
  Model* m = static_cast<Model*>(h);
  if (!m) return {};
  return {};   // ffpipe only asks this of the detector, whose shape it already knows
}

const char* ncnnLastError() { return g_err.c_str(); }

bool ncnnVariantPresent(const std::string&) {
  // ncnn has ONE variant. Presence is whether the detector's pair is on disk -- the same
  // rule QNN uses, for the same reason: the downloader renames only after the hash matches.
  std::string probe = g_spec.modelDir + "/yoloface_8n_b1.ncnn.param";
  if (FILE* f = fopen(probe.c_str(), "rb")) { fclose(f); return true; }
  return false;
}

DeviceInfo ncnnDeviceInfo() {
  DeviceInfo d;
  d.ok = true;
  d.backend = Backend::Ncnn;
  d.gpu = g_vulkan;
  d.name = g_vulkan ? "ncnn, Vulkan available" : "ncnn, CPU only";
  return d;
}

}  // namespace ffnn

#endif  // FFNN_HAVE_NCNN
