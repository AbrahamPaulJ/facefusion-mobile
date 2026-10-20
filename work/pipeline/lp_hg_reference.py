"""Dump the oracle for work/native/test_ffportrait.py: lp_gencut's hg_input from the fp32
lp_generator.onnx itself (its dense-motion Reshape_12), channels-last, for the held-out
generator cases plus two with large keypoint moves, so grid_sample's zero padding (taps off
the volume's edge) is exercised and not only the interior.

    py -3.10 work/pipeline/lp_hg_reference.py      -> work/assets/lpref/
"""
import os

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


if __name__ == '__main__':
	main()
