"""LivePortrait's three keypoint MLPs as one flat file for the CPU: lp_mlp.bin.

face_editor runs the eye retargeter, the lip retargeter and the stitcher on 21x3 keypoints
-- 66->256->256->128->128->64->63, 65->128->128->64->63, 126->128->128->64->65, all Gemm +
ReLU. That is ~200 K multiply-adds, so they run in ffportrait.cpp rather than as three more
context binaries, and the weights travel as this file (hosted once per tier, like every
other model file, so the downloader needs no tier-less special case).

Layout, little-endian:
    char[8]  "LPMLP1\\0\\0"
    for each net in (eye, lip, stitcher):
        uint32 layers
        for each layer: uint32 out, uint32 in, float32 W[out][in], float32 b[out]
ReLU follows every layer but the last.

    py -3.10 work/export/lp_mlp_export.py      -> work/onnx/lp_mlp.bin
"""
import hashlib
import os
import struct

import numpy
import onnx
from onnx import numpy_helper

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def gemms(name):
	m = onnx.load(os.path.join(ROOT, 'models', 'live_portrait_%s.onnx' % name))
	inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
	layers = []
	for n in m.graph.node:
		if n.op_type != 'Gemm':
			continue
		attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
		assert attrs.get('transB') == 1 and attrs.get('alpha', 1.0) == 1.0 and attrs.get('beta', 1.0) == 1.0
		w, b = inits[n.input[1]].astype(numpy.float32), inits[n.input[2]].astype(numpy.float32)
		layers.append((w, b))
	return layers


def main():
	out = os.path.join(ROOT, 'onnx', 'lp_mlp.bin')
	with open(out, 'wb') as f:
		f.write(b'LPMLP1\0\0')
		for name in ('eye_retargeter', 'lip_retargeter', 'stitcher'):
			layers = gemms(name)
			f.write(struct.pack('<I', len(layers)))
			for w, b in layers:
				f.write(struct.pack('<II', w.shape[0], w.shape[1]))
				f.write(numpy.ascontiguousarray(w).tobytes())
				f.write(b.tobytes())
			print('  %-15s %s' % (name, ' -> '.join([str(layers[0][0].shape[1])] + [str(w.shape[0]) for w, _ in layers])))
	data = open(out, 'rb').read()
	print('wrote %s  %d bytes  sha256 %s' % (out, len(data), hashlib.sha256(data).hexdigest()))


if __name__ == '__main__':
	main()
