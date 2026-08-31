"""Verify the native audio front end against the numpy oracle, stage by stage.

Run in WSL, from build_and_test.sh, after mel_reference.py has dumped an oracle:

    py -3.10 work/pipeline/mel_reference.py synth work/assets/lipsync_test.wav 4
    py -3.10 work/pipeline/mel_reference.py dump  work/assets/lipsync_test.wav work/assets/melref 24
    bash work/native/build_and_test.sh

Four stages are checked separately on purpose. A single end-to-end number tells you the
mouth is wrong but not which step moved it, and three of these four steps have a
convention that a reasonable implementation gets wrong (see ffaudio.h).

The tolerances are not round numbers picked to pass. Each is the measured deviation with
headroom, and the reason it is not zero is stated:

  mel bank        EXACT. Both sides compute the same floor()ed int16 edges and the same
                  triangular fill, in double. Any difference at all is a real divergence.
  prepare_audio   float32 storage of a float64 pipeline, so ~1e-7 relative.
  spectrogram     the DFT accumulates 800 terms in double on both sides but in a
                  different order from numpy's FFT, so the error is float32 rounding on
                  a value whose magnitude is ~0.03.
  windows         the log10 and the clip amplify nothing; same order as the spectrogram.
"""
import ctypes
import json
import os
import sys

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ORACLE = os.path.join(ROOT, 'work', 'assets', 'melref')

MEL_FILTER_TOTAL = 80
SPECTRUM_BINS = 401
AUDIO_STEP_SIZE = 16

TOL_BANK = 0.0
TOL_PREPARE = 1e-6
TOL_SPECTROGRAM = 1e-6
TOL_WINDOWS = 1e-5

ff = ctypes.CDLL(os.path.join(HERE, 'libffaudio.so'))
P = ctypes.POINTER
ff.ff_prepare_audio.argtypes = [P(ctypes.c_float), ctypes.c_int, ctypes.c_int, P(ctypes.c_float)]
ff.ff_prepare_audio.restype = ctypes.c_int
ff.ff_mel_filter_bank.argtypes = [P(ctypes.c_float)]
ff.ff_spectrogram_columns.argtypes = [ctypes.c_int]
ff.ff_spectrogram_columns.restype = ctypes.c_int
ff.ff_create_spectrogram.argtypes = [P(ctypes.c_float), ctypes.c_int, P(ctypes.c_float)]
ff.ff_create_spectrogram.restype = ctypes.c_int
ff.ff_extract_windows.argtypes = [P(ctypes.c_float), ctypes.c_int, ctypes.c_double,
                                  ctypes.c_float, P(ctypes.c_float)]
ff.ff_extract_windows.restype = ctypes.c_int

f32 = numpy.float32
FAILURES = []


def fp(a):
    a = numpy.ascontiguousarray(a, dtype=f32)
    return a, a.ctypes.data_as(P(ctypes.c_float))


def check(name, got, want, tol):
    if got.shape != want.shape:
        print('  %-14s FAIL shape %s vs oracle %s' % (name, got.shape, want.shape))
        FAILURES.append(name)
        return
    err = numpy.abs(got.astype(numpy.float64) - want.astype(numpy.float64)).max()
    ok = err <= tol
    print('  %-14s max abs %.3e  tol %.0e  %s' % (name, err, tol, 'ok' if ok else 'FAIL'))
    if not ok:
        FAILURES.append(name)


def main():
    if not os.path.isdir(ORACLE):
        sys.exit('no oracle at %s -- run mel_reference.py dump first' % ORACLE)
    meta = json.load(open(os.path.join(ORACLE, 'meta.json')))
    print('oracle: %s at %g fps, %d mel columns, %d windows'
          % (meta['media'], meta['fps'], meta['mel_shape'][1], meta['frames_shape'][0]))

    # ---- the mel filter bank, which must be exact
    # Loaded from the dump rather than recomputed: the oracle is produced by Windows
    # py -3.10, this runs in the WSL venv, and that venv has no scipy. A test that
    # rebuilds its own reference is not comparing against anything anyway.
    want_bank = numpy.fromfile(os.path.join(ORACLE, 'bank.f32'), dtype=f32)
    want_bank = want_bank.reshape(MEL_FILTER_TOTAL, SPECTRUM_BINS)
    got_bank = numpy.zeros((MEL_FILTER_TOTAL, SPECTRUM_BINS), dtype=f32)
    ff.ff_mel_filter_bank(got_bank.ctypes.data_as(P(ctypes.c_float)))
    check('mel bank', got_bank, want_bank, TOL_BANK)

    # ---- prepare_audio, from the resampled stereo the oracle dumped
    resampled = numpy.fromfile(os.path.join(ORACLE, 'resampled16k_stereo.f32'), dtype=f32)
    resampled = resampled.reshape(meta['resampled_shape'])
    want_mono = numpy.fromfile(os.path.join(ORACLE, 'audio16k.f32'), dtype=f32)
    src, src_p = fp(resampled)
    got_mono = numpy.zeros(resampled.shape[0], dtype=f32)
    n = ff.ff_prepare_audio(src_p, resampled.shape[0], resampled.shape[1],
                            got_mono.ctypes.data_as(P(ctypes.c_float)))
    check('prepare_audio', got_mono[:n], want_mono, TOL_PREPARE)

    # ---- the spectrogram, driven from the oracle's own mono so the stages stay isolated
    want_mel = numpy.fromfile(os.path.join(ORACLE, 'mel.f32'), dtype=f32)
    want_mel = want_mel.reshape(meta['mel_shape'])
    mono, mono_p = fp(want_mono)
    columns = ff.ff_spectrogram_columns(mono.size)
    got_mel = numpy.zeros((MEL_FILTER_TOTAL, columns), dtype=f32)
    written = ff.ff_create_spectrogram(mono_p, mono.size,
                                       got_mel.ctypes.data_as(P(ctypes.c_float)))
    if written != meta['mel_shape'][1]:
        print('  %-14s FAIL columns %d, oracle %d' % ('spectrogram', written, meta['mel_shape'][1]))
        FAILURES.append('spectrogram columns')
    else:
        check('spectrogram', got_mel, want_mel, TOL_SPECTROGRAM)

    # ---- the windows the model actually eats
    want_win = numpy.fromfile(os.path.join(ORACLE, 'frames.f32'), dtype=f32)
    want_win = want_win.reshape(meta['frames_shape'])
    mel, mel_p = fp(want_mel)
    got_win = numpy.zeros((want_win.shape[0] + 8, MEL_FILTER_TOTAL, AUDIO_STEP_SIZE), dtype=f32)
    count = ff.ff_extract_windows(mel_p, want_mel.shape[1], float(meta['fps']),
                                  float(meta['weight']),
                                  got_win.ctypes.data_as(P(ctypes.c_float)))
    if count != want_win.shape[0]:
        print('  %-14s FAIL count %d, oracle %d' % ('windows', count, want_win.shape[0]))
        FAILURES.append('window count')
    else:
        check('windows', got_win[:count], want_win.reshape(count, MEL_FILTER_TOTAL,
                                                           AUDIO_STEP_SIZE), TOL_WINDOWS)

    print()
    if FAILURES:
        sys.exit('FAILED: %s' % ', '.join(FAILURES))
    print('ffaudio matches the oracle at every stage')


if __name__ == '__main__':
    main()
