"""Dump the oracle for work/native/test_ffportrait.py: lp_gencut's hg_input from the fp32
lp_generator.onnx itself (its dense-motion Reshape_12), channels-last, for the held-out
generator cases plus two with large keypoint moves, so grid_sample's zero padding (taps off
the volume's edge) is exercised and not only the interior. Also face_editor's keypoint
arithmetic (lp_editor.apply_edit) on real motion, for ffportrait::editKeypoints.

    py -3.10 work/pipeline/lp_hg_reference.py      -> work/assets/lpref/
"""
import os
import sys

import numpy
import onnx
import onnxruntime
from onnx import helper

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HG = '/model/dense_motion_network/Reshape_12_output_0'


def main():
	out = os.path.join(ROOT, 'assets', 'lpref')
	os.makedirs(out, exist_ok=True)
	m = onnx.load(os.path.join(ROOT, 'onnx', 'lp_generator.onnx'))
	m.graph.output.append(helper.make_tensor_value_info(HG, onnx.TensorProto.FLOAT, None))
	so = onnxruntime.SessionOptions()
	so.log_severity_level = 3
	s = onnxruntime.InferenceSession(m.SerializeToString(), so, providers=['CPUExecutionProvider'])
	io = os.path.join(ROOT, 'device', 'io', 'lp_generator', 'in')
	rng = numpy.random.default_rng(7)
	cases = []
	for i in range(6):
		f = lambda k, sh: numpy.fromfile(os.path.join(io, 'lp_generator_%s_%03d.raw' % (k, i)),
		                                 numpy.float32).reshape(sh)
		cases.append((f('feature_volume', (1, 32, 16, 64, 64)), f('source', (1, 21, 3)), f('target', (1, 21, 3))))
	for i in (0, 1):
		fv, src, tgt = cases[i]
		big = (src + rng.normal(0, 0.35, src.shape)).astype(numpy.float32)
		cases.append((fv, big, tgt))
	for n, (fv, src, tgt) in enumerate(cases):
		hg = s.run([HG], {'feature_volume': fv, 'source': src, 'target': tgt})[0]
		fv.tofile(os.path.join(out, 'fv_%03d.raw' % n))
		src.tofile(os.path.join(out, 'source_%03d.raw' % n))
		tgt.tofile(os.path.join(out, 'target_%03d.raw' % n))
		numpy.ascontiguousarray(hg.transpose(0, 2, 3, 4, 1)).tofile(os.path.join(out, 'hg_%03d.raw' % n))
		print('case %d  |source - target| max %.3f  hg range [%.3f, %.3f]'
		      % (n, float(numpy.abs(src - tgt).max()), float(hg.min()), float(hg.max())))
	editor(out)


def editor(out):
	"""face_editor's keypoints (lp_editor.apply_edit) on the held-out crops' real motion,
	for ffportrait::editKeypoints. Each case: motion packed as test_ffportrait.py reads it,
	the 14 sliders, a 68-point landmark set (synthetic: only its eye and lip RATIOS are
	read, and those are computed here the same way), and the expected source / target.
	Case 0 has every slider at zero; the rest set 1-14 sliders at up to full strength."""
	import glob as g
	sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
	import lp_editor
	me = lp_editor.session('motion_extractor')
	crops = sorted(g.glob(os.path.join(ROOT, 'calib', 'edtalk_target_heldout', '*.raw')))
	rng = numpy.random.default_rng(11)
	n = 0
	for c in crops:
		motion = me.run(None, {'input': numpy.fromfile(c, numpy.float32).reshape(1, 3, 256, 256)})
		for j in range(4):
			s = ({k: 0.0 for k in lp_editor.SLIDERS} if n == 0 else
			     lp_editor.random_sliders(rng, active=int(rng.integers(1, 15))))
			lm = (rng.random((68, 2)) * 200 + 100).astype(numpy.float32)
			ratio = lambda t, b, l, r: float(numpy.linalg.norm(lm[t] - lm[b]) /
			                                 (numpy.linalg.norm(lm[l] - lm[r]) + 1e-6))
			src, tgt = lp_editor.apply_edit(motion, s, (ratio(37, 40, 39, 36), ratio(43, 46, 45, 42)),
			                                ratio(62, 66, 54, 48))
			pitch, yaw, roll, scale, translation, expression, mp = [numpy.ravel(x).astype(numpy.float32) for x in motion]
			numpy.concatenate([pitch, yaw, roll, scale, translation, expression, mp]).tofile(
				os.path.join(out, 'edit_motion_%03d.raw' % n))
			numpy.array([s[k] for k in lp_editor.SLIDERS], numpy.float32).tofile(
				os.path.join(out, 'edit_sliders_%03d.raw' % n))
			lm.tofile(os.path.join(out, 'edit_lm68_%03d.raw' % n))
			src.tofile(os.path.join(out, 'edit_source_%03d.raw' % n))
			tgt.tofile(os.path.join(out, 'edit_target_%03d.raw' % n))
			n += 1
	print('editor: %d cases' % n)


if __name__ == '__main__':
	main()
