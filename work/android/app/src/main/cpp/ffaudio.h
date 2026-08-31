// The lip syncer's audio front end: PCM in, one 80x16 mel window per video frame out.
//
// Every constant and every step here is FaceFusion 3.8.2's, from `facefusion/audio.py` and
// `lip_syncer/core.py`.  The numpy oracle is `work/pipeline/mel_reference.py`, and
// `work/native/test_ffaudio.py` measures this against it -- run that before trusting any
// of this.  Nothing here depends on Android, so the same objects build for the host test,
// the on-device CLI and the APK.
//
// Three things that are upstream's rather than conventional, and that a "sensible"
// implementation would get wrong:
//
//   * The STFT is scipy.signal.stft's, NOT librosa's. roadmap 9 said librosa and that is
//     wrong; the two disagree on window symmetry, boundary padding and scaling. What
//     matches, bit-for-bit, is: hann PERIODIC (sym=False), 400 zeros prepended AND
//     appended, the tail padded out to a whole number of 200-sample hops, and a divide
//     by the window SUM rather than by nperseg. mel_reference.selftest asserts it.
//
//   * The mel bank is not a textbook bank. Upstream floors its edges to int16 and builds
//     each triangle across ONE consecutive pair of edges, so the 80 filters neither
//     overlap nor sum to one, and some are a single bin wide. Substituting a proper
//     bank changes every mel value and therefore every mouth the model draws.
//
//   * extract_audio_frames indexes with an int16 array, which wraps negative past 32767
//     mel columns (~6.8 minutes of audio). Reproduced deliberately: diverging from
//     upstream silently is worse than reproducing a bug that is visible in the output.
//     `melColumnLimit()` names the boundary so a caller can refuse instead.

#pragma once
#include <cstdint>
#include <vector>

namespace ffaudio {

// audio.py: read_voice / prepare_voice
constexpr int kVoiceSampleRate = 16000;
// audio.py: create_spectrogram
constexpr int kMelBinTotal = 800;      // nperseg and nfft, which upstream sets equal
constexpr int kMelBinOverlap = 600;
constexpr int kMelHop = kMelBinTotal - kMelBinOverlap;   // 200
constexpr int kSpectrumBins = kMelBinTotal / 2 + 1;      // 401, the rfft length
// audio.py: create_mel_filter_bank
constexpr int kMelFilterTotal = 80;
constexpr float kFrequencyMin = 55.0f;
constexpr float kFrequencyMax = 7600.0f;
// audio.py: extract_audio_frames
constexpr int kAudioStepSize = 16;
// lip_syncer/core.py: the --lip-syncer-weight default
constexpr float kDefaultWeight = 0.5f;

// 80 x T, column-major over time: value(filter, column) = data[filter * columns + column].
struct Spectrogram {
  int filters = kMelFilterTotal;
  int columns = 0;
  std::vector<float> data;
  float at(int filter, int column) const {
    return data[(size_t)filter * columns + column];
  }
};

// One model input: 80 x 16, row-major over filters, ready for the `source` tensor.
struct MelWindow {
  std::vector<float> data;   // kMelFilterTotal * kAudioStepSize
};

// audio.py:prepare_audio -- interleaved channels to mono, peak normalise, pre-emphasis.
// `channels` may be 1. Returns the mono signal; does not resample.
std::vector<float> prepareAudio(const float* interleaved, size_t frames, int channels);

// Same, from the 16-bit PCM a decoder actually hands over.
std::vector<float> prepareAudio(const int16_t* interleaved, size_t frames, int channels);

// audio.py:create_mel_filter_bank -- 80 x 401, built once and cached.
const std::vector<float>& melFilterBank();

// audio.py:create_spectrogram -- expects 16 kHz mono, already through prepareAudio.
Spectrogram createSpectrogram(const std::vector<float>& audio);

// audio.py:extract_audio_frames followed by lip_syncer's prepare_audio_frame.
// One window per video frame, in video-frame order. `fps` is the TARGET frame rate.
//
// The first window upstream returns is the one ending at mel column 16, so window k
// covers the 200 ms FOLLOWING video frame k rather than surrounding it. That offset is
// upstream's alignment and the trim has to preserve it.
std::vector<MelWindow> extractWindows(const Spectrogram& spectrogram, double fps,
                                      float weight = kDefaultWeight);

// The int16 wrap in extract_audio_frames: beyond this many mel columns upstream's index
// array goes negative. 32767 columns is 6.8 minutes of audio.
constexpr int melColumnLimit() { return 32767; }

}  // namespace ffaudio
