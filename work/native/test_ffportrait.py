"""Verify ffportrait::hourglassInput against the ONNX oracle.

Run in WSL, from build_and_test.sh, after lp_hg_reference.py has dumped the oracle:

    py -3.10 work/pipeline/lp_hg_reference.py
    bash work/native/build_and_test.sh

Heatmaps and the warped volume are checked separately: they come from different code and
a single number would not say which one moved. Cases 6 and 7 move keypoints far enough
that taps fall off the volume, so grid_sample's zero padding is exercised.

Tolerances are measured deviations with headroom, not round numbers picked to pass:

  heatmap   exp of a sum (ONNX) against a product of three exps (here): measured 1.5e-7,
            float rounding on values in [-1, 1].
  warped    the trilinear sum is done as three 1-D passes, so the eight corners are added
            in a different order: measured 1.2e-7, float rounding on values up to ~1.
Both are far below one 16-bit step of the NPU's input encoding (~7e-6 on [-0.06, 0.42]).
"""
import ctypes
import glob
import os
import sys

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ORACLE = os.path.join(ROOT, 'work', 'assets', 'lpref')

CHANNELS = 110
VOXELS = 16 * 64 * 64
TOL_HEATMAP = 5e-7
TOL_WARPED = 5e-7

ff = ctypes.CDLL(os.path.join(HERE, 'libffportrait.so'))
P = ctypes.POINTER(ctypes.c_float)
ff.ff_lp_hourglass_input.argtypes = [P, P, P, P]

cases = sorted(glob.glob(os.path.join(ORACLE, 'hg_*.raw')))
if not cases:
	sys.exit('no oracle in %s -- run work/pipeline/lp_hg_reference.py first' % ORACLE)
ptr = lambda a: a.ctypes.data_as(P)
failures = 0
for path in cases:
	n = os.path.basename(path)[3:6]
	load = lambda k: numpy.fromfile(os.path.join(ORACLE, '%s_%s.raw' % (k, n)), numpy.float32)
	fv, src, tgt = load('fv'), load('source'), load('target')
	ref = numpy.fromfile(path, numpy.float32).reshape(VOXELS, CHANNELS)
	got = numpy.empty((VOXELS, CHANNELS), numpy.float32)
	ff.ff_lp_hourglass_input(ptr(fv), ptr(src), ptr(tgt), ptr(got))
	heat = numpy.arange(CHANNELS) % 5 == 0
	eh = float(numpy.abs(got[:, heat] - ref[:, heat]).max())
	ew = float(numpy.abs(got[:, ~heat] - ref[:, ~heat]).max())
	ok = eh <= TOL_HEATMAP and ew <= TOL_WARPED
	failures += not ok
	print('case %s  heatmap max|err| %.2e  warped max|err| %.2e  %s'
	      % (n, eh, ew, 'ok' if ok else 'FAIL'))

# face_editor's keypoints against lp_editor.apply_edit. The ops are the same float32 ops in
# the same order, except the three MLPs (a dot product summed in a different order from
# onnxruntime's GEMM) and the rotation matrix (scipy's quaternion path against a direct
# Euler product, both rounded to float32): measured 2.4e-7 on keypoints of magnitude ~1.
TOL_EDIT = 1e-6
MLP = os.path.join(ROOT, 'work', 'onnx', 'lp_mlp.bin')
ff.ff_lp_edit_keypoints.argtypes = [P, P, P, ctypes.c_char_p, P, P]
ff.ff_lp_edit_keypoints.restype = ctypes.c_int
edits = sorted(glob.glob(os.path.join(ORACLE, 'edit_motion_*.raw')))
worst = 0.0
for path in edits:
	n = os.path.basename(path)[12:15]
	load = lambda k: numpy.fromfile(os.path.join(ORACLE, 'edit_%s_%s.raw' % (k, n)), numpy.float32)
	src, tgt = numpy.empty(63, numpy.float32), numpy.empty(63, numpy.float32)
	if not ff.ff_lp_edit_keypoints(ptr(load('motion')), ptr(load('sliders')), ptr(load('lm68')),
	                               MLP.encode(), ptr(src), ptr(tgt)):
		sys.exit('cannot load %s -- run work/export/lp_mlp_export.py' % MLP)
	e = max(float(numpy.abs(src - load('source')).max()), float(numpy.abs(tgt - load('target')).max()))
	worst = max(worst, e)
	if e > TOL_EDIT:
		failures += 1
		print('edit %s  max|err| %.2e  FAIL' % (n, e))
print('editKeypoints: %d cases, worst max|err| %.2e  %s'
      % (len(edits), worst, 'ok' if worst <= TOL_EDIT else 'FAIL'))
total = len(cases) + len(edits)
print('ffportrait: %d of %d cases FAILED' % (failures, total) if failures
      else 'ffportrait: all %d cases pass' % total)
sys.exit(1 if failures else 0)
