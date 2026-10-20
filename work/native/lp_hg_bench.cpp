// On-device timing and output dump for ffportrait::hourglassInput (lp_gencut's CPU half).
//   lp_hg_bench <fv.raw> <source.raw> <target.raw> <out.raw> [runs]
// Writes the last run's hg_input to out.raw so it can be diffed against the host build and
// the ONNX oracle, and prints min / median wall time in ms. Pin it (taskset c0) as every
// other measurement here is pinned (docs/perf.md).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../android/app/src/main/cpp/ffportrait.h"

static bool readAll(const char* path, std::vector<float>& v) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  const size_t got = std::fread(v.data(), sizeof(float), v.size(), f);
  std::fclose(f);
  return got == v.size();
}

int main(int argc, char** argv) {
  if (argc < 5) { std::fprintf(stderr, "usage: %s fv source target out [runs]\n", argv[0]); return 2; }
  const int runs = argc > 5 ? std::atoi(argv[5]) : 20;
  using namespace ffportrait;
  std::vector<float> fv((size_t)kFeatureChannels * kVoxels), src(kKeypoints * 3), tgt(kKeypoints * 3);
  if (!readAll(argv[1], fv) || !readAll(argv[2], src) || !readAll(argv[3], tgt)) {
    std::fprintf(stderr, "short read\n");
    return 1;
  }
  std::vector<float> out((size_t)kVoxels * kHourglassChannels);
  std::vector<double> ms;
  for (int r = 0; r < runs; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    hourglassInput(fv.data(), src.data(), tgt.data(), out.data());
    ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  FILE* f = std::fopen(argv[4], "wb");
  if (!f) return 1;
  std::fwrite(out.data(), sizeof(float), out.size(), f);
  std::fclose(f);
  std::sort(ms.begin(), ms.end());
  std::printf("hourglassInput  min %.2f ms  median %.2f ms  n=%d\n", ms.front(), ms[ms.size() / 2], runs);
  return 0;
}
