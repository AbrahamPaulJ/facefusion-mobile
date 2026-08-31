#include "ffaudio.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ffaudio {
namespace {

constexpr double kPi = 3.14159265358979323846;

// The periodic Hann window scipy.signal.stft uses (sym=False), and its sum, which is the
// STFT's scale factor. Built once.
struct Window {
  std::vector<double> w;
  double sum = 0.0;
  Window() : w(kMelBinTotal) {
    for (int i = 0; i < kMelBinTotal; ++i) {
      w[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / kMelBinTotal);
      sum += w[i];
    }
  }
};

const Window& window() {
  static const Window instance;
  return instance;
}

// cos/sin for the 800-point real DFT, laid out bin-major so the inner loop walks one row.
//
// This is the NAIVE transform, 401 bins x 800 taps, and it is here because it is
// obviously correct against the oracle rather than because it is fast.
//
// MEASURED, x86 host, -O2: 53.7 ms for 4 s of audio, i.e. 13.43 ms per second, so a 60 s
// clip costs 0.81 s here and plausibly 2-3 s on a phone core. That is a one-off per
// video against a swap that costs 18.44 ms EVERY frame, so it is affordable but it is
// not free, and it is an order of magnitude worse than the guess this comment first
// carried before anyone timed it.
//
// 800 factors as 32 x 25, so a four-step Cooley-Tukey (25 radix-2 FFTs of 32, twiddle,
// 32 naive DFTs of 25) is about 25x fewer multiplies and is the obvious next move.
// test_ffaudio.py is what makes swapping it in safe: the tolerance is float32 rounding
// on a bit-exact convention, so a wrong FFT cannot pass it.
struct Twiddles {
  std::vector<double> cs, sn;
  Twiddles() : cs((size_t)kSpectrumBins * kMelBinTotal), sn((size_t)kSpectrumBins * kMelBinTotal) {
    for (int k = 0; k < kSpectrumBins; ++k) {
      for (int n = 0; n < kMelBinTotal; ++n) {
        const double angle = -2.0 * kPi * k * n / kMelBinTotal;
        cs[(size_t)k * kMelBinTotal + n] = std::cos(angle);
        sn[(size_t)k * kMelBinTotal + n] = std::sin(angle);
      }
    }
  }
};

const Twiddles& twiddles() {
  static const Twiddles instance;
  return instance;
}

double hertzToMel(double hertz) {
  return 2595.0 * std::log10(1.0 + hertz / 700.0);
}

double melToHertz(double mel) {
  return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

// scipy.signal.windows.triang(n): the symmetric triangular window, which for the mel bank
// is what fills each filter. scipy's definition splits on parity, so both branches are
// reproduced rather than approximated by a single formula.
void triangleInto(float* dst, int n) {
  if (n <= 0) return;
  if (n % 2 == 0) {
    for (int i = 0; i < n; ++i) {
      const double v = (2.0 * std::min(i + 1, n - i) - 1.0) / n;
      dst[i] = (float)v;
    }
  } else {
    for (int i = 0; i < n; ++i) {
      const double v = 2.0 * std::min(i + 1, n - i) / (n + 1.0);
      dst[i] = (float)v;
    }
  }
}

}  // namespace

std::vector<float> prepareAudio(const float* interleaved, size_t frames, int channels) {
  std::vector<float> mono(frames);
  if (channels <= 1) {
    std::memcpy(mono.data(), interleaved, frames * sizeof(float));
  } else {
    for (size_t i = 0; i < frames; ++i) {
      double acc = 0.0;
      for (int c = 0; c < channels; ++c) acc += interleaved[i * channels + c];
      mono[i] = (float)(acc / channels);
    }
  }

  // audio / max(abs(audio)) -- upstream divides by the peak, so a silent buffer would
  // divide by zero. numpy yields nan there; refusing is the honest equivalent.
  double peak = 0.0;
  for (float v : mono) peak = std::max(peak, (double)std::fabs(v));
  if (peak <= 0.0) return std::vector<float>(frames, 0.0f);

  // scipy.signal.lfilter([1, -0.97], [1]) with a zero initial condition: two taps, so
  // the whole filter state is the previous sample.
  double previous = 0.0;
  for (size_t i = 0; i < frames; ++i) {
    const double x = mono[i] / peak;
    mono[i] = (float)(x - 0.97 * previous);
    previous = x;
  }
  return mono;
}

std::vector<float> prepareAudio(const int16_t* interleaved, size_t frames, int channels) {
  std::vector<float> wide((size_t)frames * std::max(channels, 1));
  for (size_t i = 0; i < wide.size(); ++i) wide[i] = (float)interleaved[i];
  return prepareAudio(wide.data(), frames, channels);
}

namespace {

// Modified Bessel I0, for the Kaiser window. The series converges fast for the betas used
// here and this runs 2 x kSincHalf times per resample, not per sample.
double besselI0(double x) {
  double sum = 1.0, term = 1.0;
  for (int k = 1; k < 40; ++k) {
    term *= (x / (2.0 * k)) * (x / (2.0 * k));
    sum += term;
    if (term < 1e-18 * sum) break;
  }
  return sum;
}

// 32 zero crossings each side and beta 8.6 is the usual high-quality windowed-sinc
// setting. It is a choice, and test_ffaudio.py is what says whether it is good enough
// against upstream's FFT resample rather than merely good in the abstract.
constexpr int kSincZeroCrossings = 32;
constexpr double kKaiserBeta = 8.6;

}  // namespace

std::vector<float> resampleToVoiceRate(const float* mono, size_t frames, int inRate) {
  if (frames == 0 || inRate <= 0) return {};
  if (inRate == kVoiceSampleRate) return std::vector<float>(mono, mono + frames);

  // round(), matching prepare_voice's audio_resample_factor.
  const size_t out = (size_t)std::llround((double)frames * kVoiceSampleRate / inRate);
  if (out == 0) return {};
  const double step = (double)frames / (double)out;   // input samples per output sample

  // Cutoff at the lower of the two Nyquists, expressed in input-sample cycles. Upsampling
  // keeps the input's band; downsampling has to band-limit or it aliases.
  const double fc = 0.5 * std::min(1.0, 1.0 / step);
  const double halfWidth = kSincZeroCrossings / (2.0 * fc);
  const double i0beta = besselI0(kKaiserBeta);

  // The kernel is sampled into a table and interpolated, NOT evaluated per tap. The first
  // version called besselI0 (a 40-term series) and sin() inside the inner loop and cost
  // 105.8 ms per second of audio, which is 6.4 s for a 60 s clip on the host and several
  // times that on a phone. The table is built once per call and makes the inner loop two
  // multiplies and an add.
  constexpr int kTable = 1 << 16;
  std::vector<double> kernel((size_t)kTable + 2);
  for (int i = 0; i <= kTable + 1; ++i) {
    const double r = (double)i / kTable;              // |t| / halfWidth, in [0, 1]
    const double t = r * halfWidth;
    const double x = 2.0 * fc * t;
    const double sinc = (x < 1e-12) ? 1.0 : std::sin(M_PI * x) / (M_PI * x);
    const double w = (r >= 1.0) ? 0.0
                   : besselI0(kKaiserBeta * std::sqrt(1.0 - r * r)) / i0beta;
    kernel[i] = sinc * w;
  }
  auto kernelAt = [&](double t) {
    const double r = std::fabs(t) / halfWidth;
    if (r >= 1.0) return 0.0;
    const double f = r * kTable;
    const size_t i = (size_t)f;
    const double frac = f - (double)i;
    return kernel[i] * (1.0 - frac) + kernel[i + 1] * frac;
  };

  std::vector<float> dst(out);
  for (size_t m = 0; m < out; ++m) {
    const double centre = (double)m * step;
    const long long lo = (long long)std::ceil(centre - halfWidth);
    const long long hi = (long long)std::floor(centre + halfWidth);
    double acc = 0.0, norm = 0.0;
    for (long long k = lo; k <= hi; ++k) {
      // WRAPPED, not clipped. scipy.signal.resample is an FFT, so it treats the signal as
      // PERIODIC: its first output samples are influenced by the end of the clip and vice
      // versa. Zero-extending instead is arguably more physical and measured 3.3e+03 max
      // error at the edges against 1.1e+01 in the interior, 50.98 dB against 62.78 overall.
      // Reproducing upstream wins: this port's job is to be FaceFusion on a phone, not to
      // be a better resampler.
      const long long kk = ((k % (long long)frames) + (long long)frames) % (long long)frames;
      const double h = kernelAt((double)k - centre);
      acc += mono[kk] * h;
      norm += h;
    }
    dst[m] = (float)(norm != 0.0 ? acc / norm : 0.0);
  }
  return dst;
}

const std::vector<float>& melFilterBank() {
  static const std::vector<float> bank = [] {
    std::vector<float> b((size_t)kMelFilterTotal * kSpectrumBins, 0.0f);
    // linspace(mel(55), mel(7600), 82), then floor((800 + 1) * hz / 16000) as int16.
    const double lo = hertzToMel(kFrequencyMin), hi = hertzToMel(kFrequencyMax);
    std::vector<int> edges(kMelFilterTotal + 2);
    for (int i = 0; i < kMelFilterTotal + 2; ++i) {
      const double mel = lo + (hi - lo) * i / (kMelFilterTotal + 1);
      edges[i] = (int)(int16_t)std::floor((kMelBinTotal + 1) * melToHertz(mel) / kVoiceSampleRate);
    }
    for (int f = 0; f < kMelFilterTotal; ++f) {
      const int start = edges[f], end = edges[f + 1];
      if (end > start && start >= 0 && end <= kSpectrumBins) {
        triangleInto(&b[(size_t)f * kSpectrumBins + start], end - start);
      }
    }
    return b;
  }();
  return bank;
}

Spectrogram createSpectrogram(const std::vector<float>& audio) {
  const Window& win = window();
  const Twiddles& tw = twiddles();

  // boundary='zeros' then padded=True: 400 zeros each side, then the tail filled out to a
  // whole number of hops. This is what fixes the column count, so it is not cosmetic.
  const size_t half = kMelBinTotal / 2;
  std::vector<double> padded(audio.size() + 2 * half);
  for (size_t i = 0; i < audio.size(); ++i) padded[half + i] = audio[i];
  const long long usable = (long long)padded.size() - kMelBinOverlap;
  const int columns = usable <= 0 ? 0 : (int)((usable + kMelHop - 1) / kMelHop);
  padded.resize((size_t)kMelBinOverlap + (size_t)columns * kMelHop, 0.0);

  Spectrogram out;
  out.columns = columns;
  out.data.assign((size_t)kMelFilterTotal * std::max(columns, 0), 0.0f);
  if (columns <= 0) return out;

  const std::vector<float>& bank = melFilterBank();
  std::vector<double> segment(kMelBinTotal);
  std::vector<double> magnitude(kSpectrumBins);

  for (int c = 0; c < columns; ++c) {
    const double* src = padded.data() + (size_t)c * kMelHop;
    for (int i = 0; i < kMelBinTotal; ++i) segment[i] = src[i] * win.w[i];

    for (int k = 0; k < kSpectrumBins; ++k) {
      const double* cs = &tw.cs[(size_t)k * kMelBinTotal];
      const double* sn = &tw.sn[(size_t)k * kMelBinTotal];
      double re = 0.0, im = 0.0;
      for (int n = 0; n < kMelBinTotal; ++n) {
        re += segment[n] * cs[n];
        im += segment[n] * sn[n];
      }
      magnitude[k] = std::sqrt(re * re + im * im) / win.sum;
    }

    // numpy.dot(bank, abs(spectrum)) -- the bank is sparse but not worth exploiting;
    // 80 x 401 is 32 k multiply-adds against the 320 k the transform above just did.
    for (int f = 0; f < kMelFilterTotal; ++f) {
      const float* row = &bank[(size_t)f * kSpectrumBins];
      double acc = 0.0;
      for (int k = 0; k < kSpectrumBins; ++k) acc += row[k] * magnitude[k];
      out.data[(size_t)f * columns + c] = (float)acc;
    }
  }
  return out;
}

std::vector<MelWindow> extractWindows(const Spectrogram& spectrogram, double fps, float weight) {
  std::vector<MelWindow> windows;
  if (spectrogram.columns <= 0 || fps <= 0.0) return windows;

  // numpy.arange(0, columns, 80 / fps).astype(int16), then keep >= 16. The int16 cast is
  // upstream's and is reproduced; melColumnLimit() is where it stops being safe.
  //
  // The position MUST be i * step, not an accumulator. numpy.arange sizes the array first
  // and then fills it with start + i * step, so accumulating instead drifts by an ulp and
  // truncates differently at integer boundaries: measured, 8 of 92 windows landed one mel
  // column early (29 instead of 30 at i=9, because nine additions give 29.999999999999996
  // where the multiply gives 30.000000000000004). Each of those is a 12.5 ms slip in the
  // mouth, which is exactly the class of error nobody sees in a tensor norm.
  const double step = (double)kMelFilterTotal / fps;
  const long long positions = (long long)std::ceil(spectrogram.columns / step);
  for (long long i = 0; i < positions; ++i) {
    const double position = (double)i * step;
    const int index = (int)(int16_t)(long long)position;
    if (index < kAudioStepSize) continue;
    const int start = std::max(0, index - kAudioStepSize);
    if (index > spectrogram.columns) break;
    if (index - start != kAudioStepSize) continue;

    MelWindow w;
    w.data.resize((size_t)kMelFilterTotal * kAudioStepSize);
    for (int f = 0; f < kMelFilterTotal; ++f) {
      for (int t = 0; t < kAudioStepSize; ++t) {
        // lip_syncer/core.py:prepare_audio_frame, the wav2lip branch:
        //   maximum(exp(-5 ln 10), x) -> log10 * 1.6 + 3.2 -> clip(-4, 4) -> * weight * 2
        double v = spectrogram.at(f, start + t);
        v = std::max(v, std::exp(-5.0 * std::log(10.0)));
        v = std::log10(v) * 1.6 + 3.2;
        v = std::min(std::max(v, -4.0), 4.0);
        w.data[(size_t)f * kAudioStepSize + t] = (float)(v * weight * 2.0);
      }
    }
    windows.push_back(std::move(w));
  }
  return windows;
}

}  // namespace ffaudio
