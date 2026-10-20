"""Generator inputs made from DEVICE outputs: the whole still path on the NPU.

make_lp_io.py feeds the generator the upstream fp32 feature volume and keypoints, which
scores the generator alone. This takes the feature extractor's and the motion extractor's
own device outputs for the same held-out crops (out/<feature run>, out/<motion run>),
applies make_lp_io.py's keypoint arithmetic (and its mouth move on odd cases), and writes
io/lp_chain/ for `run_model.sh lp_chain acc lp_generator`. The references stay
io/lp_generator/ref: the upstream chain end to end.

    py -3.10 work/device/make_lp_chain.py [feature_run] [motion_run]
"""
import glob
import os
import sys

import numpy

from lp_motion_e2e import keypoints, motion

HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE = '/data/local/tmp/ff/io/lp_chain'


def main():
	feat = sys.argv[1] if len(sys.argv) > 1 else 'out/lp_feature'
	mot = sys.argv[2] if len(sys.argv) > 2 else 'out/lp_motionc'
	io = os.path.join(HERE, 'io', 'lp_chain')
	os.makedirs(os.path.join(io, 'in'), exist_ok=True)
	n = len(glob.glob(os.path.join(HERE, 'io', 'lp_generator', 'ref', 'output_*.raw')))
	lines = []
	for i in range(n):
		fv = numpy.fromfile(os.path.join(HERE, feat, 'Result_%d' % i, 'feature_volume.raw'),
		                    numpy.float32).reshape(1, 32, 16, 64, 64)
		tgt = keypoints(motion(mot, i))
		src = tgt.copy()
		if i % 2:
			src[0, 14:20, 1] += 0.01
		names = []
		for k, v in (('feature_volume', fv), ('source', src), ('target', tgt)):
			fn = 'lp_chain_%s_%03d.raw' % (k, i)
			v.tofile(os.path.join(io, 'in', fn))
			names.append('%s:=%s/in/%s' % (k, REMOTE, fn))
		lines.append(' '.join(names))
	with open(os.path.join(io, 'input_list.txt'), 'w', newline='\n') as f:
		f.write('\n'.join(lines) + '\n')
	print('%d cases from %s + %s -> %s' % (n, feat, mot, io))


if __name__ == '__main__':
	main()
