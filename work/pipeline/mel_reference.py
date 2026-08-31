"""The lip syncer's audio front end: host reference, and the oracle for the C++ port.

Ported line for line from `facefusion/audio.py` and
`facefusion/processors/modules/lip_syncer/core.py` (3.8.2).  Every constant here is
upstream's.  This file is what `ffaudio.cpp` is measured against, exactly as
`run_reference.py` is for the swap chain and `nsfw_reference.py` is for the gate.

The STFT convention is the whole risk in this file, so it is asserted rather than
assumed.  `selftest` checks a hand-rolled rfft against `scipy.signal.stft` on random
input and requires them BIT-IDENTICAL, because the hand-rolled one is the thing C++ can
reproduce and scipy is the thing upstream actually calls.  Measured 0.0 max abs
difference at nperseg 800 / noverlap 600.  The four conventions that make it exact:

    hann window, PERIODIC (sym=False)
    boundary='zeros'  -> 400 zeros prepended AND appended (nperseg // 2)
    padded=True       -> tail zero-filled to a whole number of 200-sample hops
    scaling           -> divide the rfft by win.sum(), not by nperseg

WHAT THIS DOES NOT DO YET, and it is a real divergence from upstream:

  Upstream's lip syncer reads `read_static_voice`, not `read_static_audio`.  That path
  runs the audio through the VOICE EXTRACTOR (`kim_vocal_2`, 66.8 MB ONNX, an MDX-net)
  to isolate speech from music and noise BEFORE the mel is taken.  roadmap 9 did not
  account for it.  This reference implements the audio path and leaves a seam where the
  extractor goes; on clean speech the two agree, and on a clip with a music bed they
  will not.  Measure before shipping a claim about it.

Usage:
    py -3.10 work/pipeline/mel_reference.py selftest
    py -3.10 work/pipeline/mel_reference.py synth <out.wav> [seconds]
    py -3.10 work/pipeline/mel_reference.py dump <media> <outdir> [fps]
    py -3.10 work/pipeline/mel_reference.py verify <outdir> <cxx_frames.f32>
"""
import json
import os
import subprocess
import sys

import numpy
import scipy

# audio.py: read_audio / read_voice
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNEL_TOTAL = 2
VOICE_RESAMPLE_RATE = 16000
# audio.py: create_spectrogram
MEL_BIN_TOTAL = 800
MEL_BIN_OVERLAP = 600
MEL_HOP = MEL_BIN_TOTAL - MEL_BIN_OVERLAP
# audio.py: create_mel_filter_bank
MEL_FILTER_TOTAL = 80
AUDIO_FREQUENCY_MIN = 55.0
AUDIO_FREQUENCY_MAX = 7600.0
# audio.py: extract_audio_frames
AUDIO_STEP_SIZE = 16
# lip_syncer/core.py: register_args default
LIP_SYNCER_WEIGHT = 0.5


def decode_audio(media_path):
	"""ffmpeg.py:read_audio_buffer -- 48 kHz stereo s16le, exactly what upstream asks for."""
	cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', media_path,
	       '-vn', '-f', 's16le', '-acodec', 'pcm_s16le',
	       '-ar', str(AUDIO_SAMPLE_RATE), '-ac', str(AUDIO_CHANNEL_TOTAL), '-']
	buf = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
	return numpy.frombuffer(buf, dtype=numpy.int16).reshape(-1, AUDIO_CHANNEL_TOTAL)


def prepare_audio(audio):
	"""audio.py:prepare_audio -- mean to mono, peak normalise, pre-emphasis.

	The lfilter is y[n] = x[n] - 0.97 * x[n-1] with a zero initial condition, which is
	an FIR of two taps and needs no state in C++ beyond the previous sample.
	"""
	if audio.ndim > 1:
		audio = numpy.mean(audio, axis=1)
	audio = audio / numpy.max(numpy.abs(audio), axis=0)
	audio = scipy.signal.lfilter([1.0, -0.97], [1.0], audio)
	return audio


def prepare_voice(audio):
	"""audio.py:prepare_voice -- 48 kHz to 16 kHz, then prepare_audio.

	WARNING: `scipy.signal.resample` is FFT-based, not polyphase: it takes the rfft of
	the WHOLE signal, truncates the spectrum and inverts.  A phone will not do that; the
	app's resampler is a design decision with a measurable error, and this is the number
	to measure it against.
	"""
	factor = round(len(audio) * VOICE_RESAMPLE_RATE / AUDIO_SAMPLE_RATE)
	audio = scipy.signal.resample(audio, factor)
	return prepare_audio(audio)


def convert_hertz_to_mel(hertz):
	return 2595 * numpy.log10(1 + hertz / 700)


def convert_mel_to_hertz(mel):
	return 700 * (10 ** (mel / 2595) - 1)


def create_mel_filter_bank():
	"""audio.py:create_mel_filter_bank.

	WARNING: the bin edges are floor()ed to int16 and the triangles are built BETWEEN
	CONSECUTIVE edges, not the usual overlapping start/centre/end triple.  Filter i
	covers [indices[i], indices[i+1]) and nothing else, so the bank does not sum to one
	and adjacent filters do not overlap.  That is upstream's, and reproducing wav2lip
	means reproducing it rather than substituting a textbook bank.
	"""
	bank = numpy.zeros((MEL_FILTER_TOTAL, MEL_BIN_TOTAL // 2 + 1))
	mel_range = numpy.linspace(convert_hertz_to_mel(AUDIO_FREQUENCY_MIN),
	                           convert_hertz_to_mel(AUDIO_FREQUENCY_MAX),
	                           MEL_FILTER_TOTAL + 2)
	indices = numpy.floor((MEL_BIN_TOTAL + 1) * convert_mel_to_hertz(mel_range)
	                      / VOICE_RESAMPLE_RATE).astype(numpy.int16)
	for index in range(MEL_FILTER_TOTAL):
		start, end = indices[index], indices[index + 1]
		bank[index, start:end] = scipy.signal.windows.triang(end - start)
	return bank


def stft_manual(audio):
	"""What C++ has to implement.  selftest asserts this equals scipy.signal.stft."""
	window = scipy.signal.windows.hann(MEL_BIN_TOTAL, sym=False)
	padded = numpy.concatenate([numpy.zeros(MEL_BIN_TOTAL // 2), audio,
	                            numpy.zeros(MEL_BIN_TOTAL // 2)])
	segment_total = -((len(padded) - MEL_BIN_OVERLAP) // -MEL_HOP)
	need = MEL_BIN_OVERLAP + segment_total * MEL_HOP
	padded = numpy.concatenate([padded, numpy.zeros(need - len(padded))])
	frames = numpy.stack([padded[i * MEL_HOP:i * MEL_HOP + MEL_BIN_TOTAL] * window
	                      for i in range(segment_total)], axis=1)
	return numpy.fft.rfft(frames, n=MEL_BIN_TOTAL, axis=0) / window.sum()


def create_spectrogram(audio, manual=False):
	"""audio.py:create_spectrogram."""
	if manual:
		spectrum = stft_manual(audio)
	else:
		spectrum = scipy.signal.stft(audio, nperseg=MEL_BIN_TOTAL, nfft=MEL_BIN_TOTAL,
		                             noverlap=MEL_BIN_OVERLAP)[2]
	return numpy.dot(create_mel_filter_bank(), numpy.abs(spectrum))


def extract_audio_frames(spectrogram, fps):
	"""audio.py:extract_audio_frames -- one 80x16 window per VIDEO frame.

	WARNING: `indices` is cast to int16, so a clip long enough to pass 32767 mel columns
	wraps NEGATIVE.  At 80 columns per second of audio that is ~6.8 minutes.  Upstream's
	bug; noted because this port will hit it before upstream's users do, videos being
	what a phone holds.
	"""
	frames = []
	indices = numpy.arange(0, spectrogram.shape[1], MEL_FILTER_TOTAL / fps).astype(numpy.int16)
	indices = indices[indices >= AUDIO_STEP_SIZE]
	for index in indices:
		start = max(0, index - AUDIO_STEP_SIZE)
		frames.append(spectrogram[:, start:index])
	return frames


def prepare_audio_frame(frame, weight=LIP_SYNCER_WEIGHT):
	"""lip_syncer/core.py:prepare_audio_frame, the wav2lip branch."""
	frame = numpy.maximum(numpy.exp(-5 * numpy.log(10)), frame)
	frame = numpy.log10(frame) * 1.6 + 3.2
	frame = frame.clip(-4, 4).astype(numpy.float32)
	frame = frame * weight * 2.0
	return numpy.expand_dims(frame, axis=(0, 1))


def synth(out_path, seconds=4.0):
	"""Deterministic speech-like audio, because no clip in the repo has an audio track.

	A real voice is not needed to measure a spectrogram port; a reproducible signal with
	energy across the mel range is, and this one regenerates byte-identically from the
	seed rather than living in the repo as an asset. Harmonic stack with a wandering F0
	and two moving formants, amplitude-gated into syllables so the mel has structure in
	time as well as frequency.
	"""
	rng = numpy.random.default_rng(12345)
	n = int(AUDIO_SAMPLE_RATE * seconds)
	t = numpy.arange(n) / AUDIO_SAMPLE_RATE
	f0 = 120 + 25 * numpy.sin(2 * numpy.pi * 0.7 * t)
	phase = 2 * numpy.pi * numpy.cumsum(f0) / AUDIO_SAMPLE_RATE
	sig = numpy.zeros(n)
	for k in range(1, 41):
		formant = numpy.exp(-((k * f0 - (700 + 500 * numpy.sin(2 * numpy.pi * 0.4 * t))) ** 2)
		                    / (2 * 260.0 ** 2))
		formant += numpy.exp(-((k * f0 - (1900 + 700 * numpy.sin(2 * numpy.pi * 0.3 * t))) ** 2)
		                     / (2 * 420.0 ** 2))
		sig += formant * numpy.sin(k * phase) / k
	gate = (numpy.sin(2 * numpy.pi * 3.1 * t) > -0.35).astype(float)
	gate = scipy.signal.lfilter(numpy.ones(400) / 400, [1.0], gate)
	sig = sig * gate + 0.004 * rng.standard_normal(n)
	sig = sig / numpy.max(numpy.abs(sig)) * 0.89
	stereo = numpy.stack([sig, numpy.roll(sig, 13)], axis=1)
	pcm = (stereo * 32767).astype(numpy.int16)
	import wave
	with wave.open(out_path, 'wb') as handle:
		handle.setnchannels(AUDIO_CHANNEL_TOTAL)
		handle.setsampwidth(2)
		handle.setframerate(AUDIO_SAMPLE_RATE)
		handle.writeframes(pcm.tobytes())
	print('wrote %s  %d samples  %.1f s' % (out_path, n, seconds))
	return out_path


def selftest():
	rng = numpy.random.default_rng(0)
	audio = rng.standard_normal(48000)
	a = scipy.signal.stft(audio, nperseg=MEL_BIN_TOTAL, nfft=MEL_BIN_TOTAL,
	                      noverlap=MEL_BIN_OVERLAP)[2]
	b = stft_manual(audio)
	assert a.shape == b.shape, 'shape %s vs %s' % (a.shape, b.shape)
	diff = numpy.abs(a - b).max()
	print('stft   scipy vs hand-rolled : max abs %.3e over %s' % (diff, a.shape))
	assert diff == 0.0, 'the C++ convention is NOT what scipy does -- fix before porting'
	bank = create_mel_filter_bank()
	nonzero = (bank > 0).sum(axis=1)
	print('bank   %s, per-filter width min %d max %d, empty filters %d'
	      % (bank.shape, nonzero.min(), nonzero.max(), int((nonzero == 0).sum())))
	print('OK')


def dump(media_path, out_dir, fps):
	os.makedirs(out_dir, exist_ok=True)
	raw = decode_audio(media_path)
	audio = prepare_voice(raw)
	spectrogram = create_spectrogram(audio)
	frames = extract_audio_frames(spectrogram, fps)
	prepared = numpy.concatenate([prepare_audio_frame(f) for f in frames
	                              if f.shape[1] == AUDIO_STEP_SIZE], axis=0)
	raw.astype(numpy.int16).tofile(os.path.join(out_dir, 'pcm48_stereo.s16'))
	audio.astype(numpy.float32).tofile(os.path.join(out_dir, 'audio16k.f32'))
	spectrogram.astype(numpy.float32).tofile(os.path.join(out_dir, 'mel.f32'))
	prepared.astype(numpy.float32).tofile(os.path.join(out_dir, 'frames.f32'))
	meta = {
		'media': os.path.basename(media_path), 'fps': fps,
		'pcm48_samples': int(raw.shape[0]), 'audio16k_samples': int(audio.shape[0]),
		'mel_shape': list(spectrogram.shape), 'frames_shape': list(prepared.shape),
		'weight': LIP_SYNCER_WEIGHT,
	}
	with open(os.path.join(out_dir, 'meta.json'), 'w') as handle:
		json.dump(meta, handle, indent=1)
	print(json.dumps(meta, indent=1))
	print('mel   min %8.4f max %8.4f' % (spectrogram.min(), spectrogram.max()))
	print('frame min %8.4f max %8.4f' % (prepared.min(), prepared.max()))
	return meta


def verify(out_dir, cxx_path):
	meta = json.load(open(os.path.join(out_dir, 'meta.json')))
	ref = numpy.fromfile(os.path.join(out_dir, 'frames.f32'), dtype=numpy.float32)
	got = numpy.fromfile(cxx_path, dtype=numpy.float32)
	if ref.shape != got.shape:
		sys.exit('FAIL shape: reference %d floats, C++ %d' % (ref.size, got.size))
	err = numpy.abs(ref - got)
	denom = numpy.mean((ref - got) ** 2)
	snr = 10 * numpy.log10(numpy.mean(ref ** 2) / denom) if denom > 0 else float('inf')
	print('frames %s  max abs %.3e  mean abs %.3e  SNR %.2f dB'
	      % (meta['frames_shape'], err.max(), err.mean(), snr))
	return err.max()


if __name__ == '__main__':
	if len(sys.argv) < 2:
		sys.exit(__doc__)
	mode = sys.argv[1]
	if mode == 'selftest':
		selftest()
	elif mode == 'synth':
		synth(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 4.0)
	elif mode == 'dump':
		dump(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 25.0)
	elif mode == 'verify':
		verify(sys.argv[2], sys.argv[3])
	else:
		sys.exit(__doc__)
