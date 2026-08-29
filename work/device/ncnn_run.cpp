// Run one ncnn model with REAL weights and dump the output, so fp16 and fp32 can be
// compared numerically.  benchncnn cannot answer this: it runs on DataReaderFromEmpty,
// which is fine for latency and useless for accuracy.
//
//   ncnn_run <param> <bin> <mode> <out.raw> <shape>...
//     mode: fp32 | fp16 | vulkan
//     shape: WxHxC  or  N   (in graph input order -- hyperswap is source[512] then
//            target[256x256x3], and swapping them converts and runs and means nothing)
//
// The input is deterministic (a fixed-seed LCG, same bytes every run and every mode), so
// any difference in the output is the arithmetic and not the input.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include "net.h"

static void fill(ncnn::Mat& m, unsigned int seed)
{
    // Deliberately NOT rand(): this has to produce identical bytes on every backend and
    // every run, and it has to be reproducible from the host when comparing.
    unsigned int s = seed;
    float* p = m;
    for (int i = 0; i < (int)m.total(); i++) {
        s = s * 1664525u + 1013904223u;
        // [-1, 1): the range the pipeline actually feeds these graphs.
        p[i] = ((float)(s >> 8) / 8388608.0f) - 1.0f;
    }
}

int main(int argc, char** argv)
{
    if (argc < 6) {
        fprintf(stderr, "usage: ncnn_run <param> <bin> <fp32|fp16|vulkan> <out.raw> <shape>...\n");
        return 1;
    }
    const char* param = argv[1];
    const char* bin   = argv[2];
    std::string mode  = argv[3];
    const char* outp  = argv[4];

    // Scoped: ~Net must run BEFORE destroy_gpu_instance(), or teardown aborts with
    // "pool allocator destroyed too early" -- the output is already written by then,
    // so it looked like a passing run with a crash stapled to the end.
    {
    ncnn::Net net;
    net.opt.lightmode = true;
    net.opt.num_threads = 4;
    net.opt.use_packing_layout = true;

    if (mode == "fp32") {
        // Every fp16 path off: this is the reference the others are scored against.
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.opt.use_bf16_storage = false;
        net.opt.use_vulkan_compute = false;
    } else if (mode == "fp16") {
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = true;
        net.opt.use_vulkan_compute = false;
    } else if (mode == "vulkan") {
#if NCNN_VULKAN
        ncnn::create_gpu_instance();
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = true;
#else
        fprintf(stderr, "built without vulkan\n"); return 1;
#endif
    } else { fprintf(stderr, "bad mode %s\n", mode.c_str()); return 1; }

    if (net.load_param(param)) { fprintf(stderr, "load_param failed\n"); return 1; }
    if (net.load_model(bin))   { fprintf(stderr, "load_model failed\n"); return 1; }

    std::vector<ncnn::Mat> ins;
    for (int i = 5; i < argc; i++) {
        int w, h, c;
        if (sscanf(argv[i], "%dx%dx%d", &w, &h, &c) == 3) ins.push_back(ncnn::Mat(w, h, c));
        else if (sscanf(argv[i], "%d", &w) == 1)          ins.push_back(ncnn::Mat(w));
        else { fprintf(stderr, "bad shape %s\n", argv[i]); return 1; }
        // Seeded per input INDEX, so input 0 and input 1 differ but are stable across modes.
        fill(ins.back(), 1234u + 7919u * i);
    }

    ncnn::Extractor ex = net.create_extractor();
    for (size_t i = 0; i < ins.size(); i++) {
        char name[16]; snprintf(name, sizeof(name), "in%d", (int)i);
        if (ex.input(name, ins[i])) { fprintf(stderr, "input %s failed\n", name); return 1; }
    }
    ncnn::Mat out;
    if (ex.extract("out0", out)) { fprintf(stderr, "extract out0 failed\n"); return 1; }

    // Always written back as fp32 regardless of the internal precision, so the files are
    // directly comparable.
    FILE* f = fopen(outp, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", outp); return 1; }
    for (int q = 0; q < out.c; q++) {
        const float* p = out.channel(q);
        fwrite(p, sizeof(float), (size_t)out.w * out.h, f);
    }
    fclose(f);
    fprintf(stderr, "%s: out %dx%dx%d (%zu floats) -> %s\n",
            mode.c_str(), out.w, out.h, out.c, out.total(), outp);

    }
#if NCNN_VULKAN
    if (mode == "vulkan") ncnn::destroy_gpu_instance();
#endif
    return 0;
}
