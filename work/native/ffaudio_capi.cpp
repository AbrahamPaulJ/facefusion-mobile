// Flat C ABI over ffaudio, so the host test can drive it from Python with ctypes and
// compare against the numpy oracle in the same process.  Not used by the app.
#include <cstring>
#include <vector>

#include "../android/app/src/main/cpp/ffaudio.h"

using namespace ffaudio;

extern "C" {

// prepare_audio: interleaved float in, mono out. Returns the sample count written.
int ff_prepare_audio(const float* interleaved, int frames, int channels, float* dst) {
  std::vector<float> mono = prepareAudio(interleaved, (size_t)frames, channels);
  std::memcpy(dst, mono.data(), mono.size() * sizeof(float));
  return (int)mono.size();
}

// The mel filter bank, 80 x 401, so the test can diff it against numpy's directly rather
// than only seeing it through the spectrogram.
void ff_mel_filter_bank(float* dst) {
  const std::vector<float>& bank = melFilterBank();
  std::memcpy(dst, bank.data(), bank.size() * sizeof(float));
}

// Column count for a given input length, so Python can size its buffer before the call.
int ff_spectrogram_columns(int samples) {
  const long long padded = (long long)samples + kMelBinTotal;
  const long long usable = padded - kMelBinOverlap;
  return usable <= 0 ? 0 : (int)((usable + kMelHop - 1) / kMelHop);
}

// create_spectrogram: 16 kHz mono in, 80 x columns out. Returns columns.
int ff_create_spectrogram(const float* audio, int samples, float* dst) {
  std::vector<float> in(audio, audio + samples);
  Spectrogram s = createSpectrogram(in);
  std::memcpy(dst, s.data.data(), s.data.size() * sizeof(float));
  return s.columns;
}

// extract_audio_frames + prepare_audio_frame. Returns the window count; `dst` must hold
// at least count * 80 * 16 floats, which ff_window_count reports first.
int ff_extract_windows(const float* spectrogram, int columns, double fps, float weight,
                       float* dst) {
  Spectrogram s;
  s.columns = columns;
  s.data.assign(spectrogram, spectrogram + (size_t)kMelFilterTotal * columns);
  std::vector<MelWindow> windows = extractWindows(s, fps, weight);
  const size_t stride = (size_t)kMelFilterTotal * kAudioStepSize;
  for (size_t i = 0; i < windows.size(); ++i) {
    std::memcpy(dst + i * stride, windows[i].data.data(), stride * sizeof(float));
  }
  return (int)windows.size();
}

}  // extern "C"
