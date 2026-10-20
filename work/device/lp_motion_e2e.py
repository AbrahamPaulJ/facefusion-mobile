"""What a motion-extractor build costs in the PICTURE, not in its own outputs.

`expression` is a small residual (|x| < 0.07) and its SNR condemns errors the generator
never sees: face_editor derives source AND target keypoints from the same extraction, so an
error common to both cancels in the warp. Measured 2026-10-20 on the S25: a W8A16 build at
1.78 dB on `expression` gives a BIT-IDENTICAL face with no edit (300 dB) and ~71 dB with a
small mouth move. What does not cancel is an edit that moves keypoints by a pose-dependent
amount, so the second number here is a head turn (face_editor edit_head_rotation, yaw +DY):
the fraction of the edit's own change that comes out wrong.

Each run's motion outputs go through the same keypoint arithmetic as make_lp_io.py into the
ORIGINAL upstream generator (onnxruntime) with the REFERENCE feature volume, so only the
motion extractor differs. `ref` (the upstream motion outputs) must score 300 dB / 0.

    py -3.10 work/device/lp_motion_e2e.py ref out/lp_motion out/lp_motionc ...
"""
import glob
import os
import sys

import numpy
import onnxruntime as ort
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
NAMES = ['pitch', 'yaw', 'roll', 'scale', 'translation', 'expression', 'motion_points']
SHAPES = {'pitch': (1,), 'yaw': (1,), 'roll': (1,), 'scale': (1, 1), 'translation': (1, 3),
          'expression': (1, 21, 3), 'motion_points': (1, 21, 3)}
DY = float(os.environ.get('DY', '15'))


def psnr(a, b):
	return min(300.0, 10 * numpy.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-30)))


def motion(run, i):
	if run == 'ref':
		return [numpy.fromfile(os.path.join(HERE, 'io', 'lp_motion', 'ref', '%s_%03d.raw' % (k, i)),
		                       numpy.float32).reshape(SHAPES[k]) for k in NAMES]
	return [numpy.fromfile(glob.glob(os.path.join(HERE, run, 'Result_%d' % i, k + '.raw'))[0],
	                       numpy.float32).reshape(SHAPES[k]) for k in NAMES]


def keypoints(m, dyaw=0.0):
	"""make_lp_io.py keypoints(), with face_editor's head-rotation edit on the yaw."""
	pitch, yaw, roll, scale, translation, expression, mp = m
	r = Rotation.from_euler('xyz', [float(numpy.ravel(pitch)[0]), float(numpy.ravel(yaw)[0]) + dyaw,
	                                float(numpy.ravel(roll)[0])], degrees=True).as_matrix()
	return (scale * (mp @ r.astype(numpy.float32).T + expression) + translation).astype(numpy.float32)


def main():
	so = ort.SessionOptions()
	so.log_severity_level = 3
	gen = ort.InferenceSession(os.path.join(ROOT, 'models', 'live_portrait_generator.onnx'), so,
	                           providers=['CPUExecutionProvider'])
	io = os.path.join(HERE, 'io', 'lp_generator')
	n = len(glob.glob(os.path.join(io, 'ref', 'output_*.raw')))
	fvs = [numpy.fromfile(os.path.join(io, 'in', 'lp_generator_feature_volume_%03d.raw' % i),
	                      numpy.float32).reshape(1, 32, 16, 64, 64) for i in range(n)]
	face = lambda i, src, tgt: gen.run(None, {'feature_volume': fvs[i], 'source': src,
	                                          'target': tgt})[0].clip(0, 1)
	turned_ref = [face(i, keypoints(motion('ref', i), DY), keypoints(motion('ref', i))) for i in range(n)]
	plain_ref = [face(i, keypoints(motion('ref', i)), keypoints(motion('ref', i))) for i in range(n)]
	for run in sys.argv[1:]:
		same, wrong = [], []
		for i in range(n):
			m = motion(run, i)
			tgt = keypoints(m)
			src = tgt.copy()
			if i % 2:
				src[0, 14:20, 1] += 0.01   # make_lp_io.py's mouth move, so cases line up
			ref = numpy.fromfile(os.path.join(io, 'ref', 'output_%03d.raw' % i), numpy.float32)
			same.append(psnr(numpy.ravel(face(i, src, tgt)), ref.clip(0, 1)))
			y = face(i, keypoints(m, DY), tgt)
			wrong.append(float(numpy.sqrt(((y - turned_ref[i]) ** 2).mean() /
			                              ((turned_ref[i] - plain_ref[i]) ** 2).mean())))
		print('%-24s no edit / mouth: PSNR %s | yaw %+g deg: %.1f%% of the edit wrong (worst %.1f%%)'
		      % (run, numpy.round(same, 1), DY, 100 * numpy.mean(wrong), 100 * max(wrong)), flush=True)


if __name__ == '__main__':
	main()
