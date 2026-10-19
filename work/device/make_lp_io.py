"""Held-out I/O for the LivePortrait extractors (docs/roadmap.md 10), for compare.py.

Inputs are REAL face crops: the edtalk calibration crops, which are upstream's ffhq_512 crop
resized to 256, RGB, [0,1] -- the same preparation face_editor's `prepare_crop_frame` does.
face_editor scales landmark-5 by 1.5 first, so its crop is looser; that changes which
pixels are in the crop, not the input's distribution, and a conversion check needs real
faces rather than the exact framing (trap #4: white noise is what it must NOT be fed).

References come from the ORIGINAL upstream ONNX, never from the prepared graph, so the
device is scored against upstream and the surgery is inside what is measured.

    py -3.10 work/device/make_lp_io.py
"""
import glob
import os

import numpy
import onnxruntime as ort

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REMOTE = '/data/local/tmp/ff/io'
GRAPHS = {
	'lp_feature': 'live_portrait_feature_extractor',
	'lp_motion': 'live_portrait_motion_extractor',
}


def main():
	# HELD-OUT crops only: work/calib/lp_*_list.txt calibrates the W8A16 builds on
	# edtalk_target, and scoring on the calibration set cannot fail.
	crops = sorted(glob.glob(os.path.join(ROOT, 'calib', 'edtalk_target_heldout', '*.raw')))
	if not crops:
		raise SystemExit('no edtalk crops under work/calib')
	for name, onnx_name in GRAPHS.items():
		io = os.path.join(ROOT, 'device', 'io', name)
		os.makedirs(os.path.join(io, 'in'), exist_ok=True)
		os.makedirs(os.path.join(io, 'ref'), exist_ok=True)
		sess = ort.InferenceSession(os.path.join(ROOT, 'models', onnx_name + '.onnx'),
		                            providers=['CPUExecutionProvider'])
		outs = [o.name for o in sess.get_outputs()]
		lines = []
		for i, c in enumerate(crops):
			x = numpy.fromfile(c, numpy.float32).reshape(1, 3, 256, 256)
			fn = '%s_input_%03d.raw' % (name, i)
			x.tofile(os.path.join(io, 'in', fn))
			lines.append('input:=%s/%s/in/%s' % (REMOTE, name, fn))
			for oname, y in zip(outs, sess.run(None, {'input': x})):
				numpy.asarray(y, numpy.float32).reshape(-1).tofile(
					os.path.join(io, 'ref', '%s_%03d.raw' % (oname, i)))
		with open(os.path.join(io, 'input_list.txt'), 'w', newline='\n') as f:
			f.write('\n'.join(lines) + '\n')
		print('%s: %d cases, outputs %s' % (name, len(crops), outs))
	generator_io()


def keypoints(motion):
	"""face_editor `apply_edit`, with no edit applied: the keypoints the generator is fed.

	motion_points_target = scale * (motion_points @ R.T + expression) + translation, and
	with every editor slider at zero the source keypoints are the same arithmetic on the
	limited expression. A small mouth edit is added to half the cases so calibration also
	sees source != target, which is the only case the generator exists for.
	"""
	from scipy.spatial.transform import Rotation
	pitch, yaw, roll, scale, translation, expression, mp = motion
	r = Rotation.from_euler('xyz', [float(numpy.ravel(pitch)[0]), float(numpy.ravel(yaw)[0]),
	                                float(numpy.ravel(roll)[0])], degrees=True).as_matrix()
	r = r.astype(numpy.float32)
	target = scale * (mp @ r.T + expression) + translation
	return target.astype(numpy.float32)


def generator_io():
	"""Generator calibration (edtalk_target crops) and held-out I/O (edtalk_target_heldout),
	both through the ORIGINAL upstream extractors and generator."""
	fe = ort.InferenceSession(os.path.join(ROOT, 'models', 'live_portrait_feature_extractor.onnx'),
	                          providers=['CPUExecutionProvider'])
	me = ort.InferenceSession(os.path.join(ROOT, 'models', 'live_portrait_motion_extractor.onnx'),
	                          providers=['CPUExecutionProvider'])
	gen = ort.InferenceSession(os.path.join(ROOT, 'models', 'live_portrait_generator.onnx'),
	                           providers=['CPUExecutionProvider'])
	for split, sub in (('calib', 'edtalk_target'), ('io', 'edtalk_target_heldout')):
		crops = sorted(glob.glob(os.path.join(ROOT, 'calib', sub, '*.raw')))
		if split == 'calib':
			base = os.path.join(ROOT, 'calib', 'lp_generator')
			mnt = '/mnt/c/' + base[3:].replace('\\', '/')
			lines = []
		else:
			base = os.path.join(ROOT, 'device', 'io', 'lp_generator')
			os.makedirs(os.path.join(base, 'ref'), exist_ok=True)
			lines = []
		os.makedirs(os.path.join(base, 'in'), exist_ok=True)
		for i, c in enumerate(crops):
			x = numpy.fromfile(c, numpy.float32).reshape(1, 3, 256, 256)
			fv = fe.run(None, {'input': x})[0].astype(numpy.float32)
			tgt = keypoints(me.run(None, {'input': x}))
			src = tgt.copy()
			if i % 2:
				src[0, 14:20, 1] += 0.01   # a small mouth-area move
			names = {}
			for k, v in (('feature_volume', fv), ('source', src), ('target', tgt)):
				fn = 'lp_generator_%s_%03d.raw' % (k, i)
				v.tofile(os.path.join(base, 'in', fn))
				names[k] = fn
			if split == 'calib':
				lines.append(' '.join('%s:=%s/in/%s' % (k, mnt, names[k]) for k in names))
			else:
				lines.append(' '.join('%s:=%s/lp_generator/in/%s' % (k, REMOTE, names[k])
				                      for k in names))
				y = gen.run(None, {'feature_volume': fv, 'source': src, 'target': tgt})[0]
				numpy.asarray(y, numpy.float32).reshape(-1).tofile(
					os.path.join(base, 'ref', 'output_%03d.raw' % i))
		out = (os.path.join(ROOT, 'calib', 'lp_generator_list.txt') if split == 'calib'
		       else os.path.join(base, 'input_list.txt'))
		with open(out, 'w', newline='\n') as f:
			f.write('\n'.join(lines) + '\n')
		print('lp_generator %s: %d cases -> %s' % (split, len(crops), out))


if __name__ == '__main__':
	main()
