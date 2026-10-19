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


if __name__ == '__main__':
	main()
