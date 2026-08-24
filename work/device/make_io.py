"""Build device I/O for one model: held-out inputs + the onnxruntime reference outputs.

Held-out on purpose -- the calibration set trained the encodings, so measuring deploy SNR
on it flatters the result.  These come from the tail of the capture, calibration from the
head/stride.

Writes:
    work/device/io/<name>/in/*.raw        float32, what qnn-net-run consumes
    work/device/io/<name>/ref/*.raw       float32, the onnxruntime reference
    work/device/io/<name>/input_list.txt  device-side paths
"""
import argparse
import glob
import os
import sys

import numpy
import onnxruntime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REMOTE = '/data/local/tmp/ff'

SPECS = {
	'arcface':   dict(onnx='onnx/arcface_w600k_r50_b1_sim.onnx',
					  inputs=[('input', 'calib/arcface', (1, 3, 112, 112))]),
	'fan2d':     dict(onnx='onnx/2dfan4_heatmaps_sim.onnx',
					  inputs=[('input', 'calib/fan2d', (1, 3, 256, 256))]),
	'yoloface':  dict(onnx='onnx/yoloface_8n_b1_sim.onnx',
					  inputs=[('input', 'calib/yoloface', (1, 3, 640, 640))]),
	'hyperswap': dict(onnx='onnx/hyperswap_1a_256_sim.onnx',
					  inputs=[('target', 'calib/swap_target', (1, 3, 256, 256)),
							  ('source', 'calib/swap_source', (1, 512))]),
	# The held-out set is the ODD stride phase, so it is DISJOINT from calibration rather
	# than merely the tail of the same sequence (nsfw_reference.py:frames).
	'nsfw':      dict(onnx='onnx/nsfw_2_sim.onnx',
					  inputs=[('input', 'calib/nsfw_heldout', (1, 3, 384, 384))]),
	'inswapper': dict(onnx='onnx/inswapper_128_split_sim.onnx',
					  inputs=[('target', 'calib/swap_target_128', (1, 3, 128, 128)),
							  ('source', 'calib/swap_source_128', (1, 512))]),
}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('name', choices=list(SPECS))
	ap.add_argument('-n', type=int, default=16, help='how many held-out cases')
	args = ap.parse_args()

	spec = SPECS[args.name]
	out = os.path.join(ROOT, 'device', 'io', args.name)
	for sub in ('in', 'ref'):
		os.makedirs(os.path.join(out, sub), exist_ok=True)

	cols = []
	for input_name, subdir, shape in spec['inputs']:
		files = sorted(glob.glob(os.path.join(ROOT, subdir, '*.raw')))
		if not files:
			sys.exit('no tensors in %s' % subdir)
		cols.append((input_name, files[-args.n:], shape))   # the TAIL: held out

	n = min(len(f) for _, f, _ in cols)
	opts = onnxruntime.SessionOptions()
	opts.log_severity_level = 3
	sess = onnxruntime.InferenceSession(os.path.join(ROOT, spec['onnx']), opts,
										providers=['CPUExecutionProvider'])
	out_names = [o.name for o in sess.get_outputs()]
	print('%s: %d held-out cases, outputs %s' % (args.name, n, out_names))

	lines = []
	multi = len(cols) > 1
	first_ref = None
	for i in range(n):
		feed, parts = {}, []
		for input_name, files, shape in cols:
			x = numpy.fromfile(files[i], numpy.float32).reshape(shape)
			feed[input_name] = x
			dst = os.path.join(out, 'in', '%s_%s_%03d.raw' % (args.name, input_name, i))
			numpy.ascontiguousarray(x).tofile(dst)
			rp = '%s/io/%s/in/%s' % (REMOTE, args.name, os.path.basename(dst))
			parts.append(('%s:=%s' % (input_name, rp)) if multi else rp)
		lines.append(' '.join(parts))

		res = sess.run(None, feed)
		for k, name in enumerate(out_names):
			safe = name.replace('/', '_').replace(':', '_')
			numpy.ascontiguousarray(res[k], numpy.float32).tofile(
				os.path.join(out, 'ref', '%s_%03d.raw' % (safe, i)))
		if i == 0:
			first_ref = res[0].copy()
		elif first_ref is not None:
			# trap #8: identical outputs from different inputs means the runner is
			# misreading the raws.  Assert the reference itself varies, so a flat device
			# result can only be the device's fault.
			if numpy.array_equal(first_ref, res[0]):
				print('  WARNING: case %d gives a bit-identical output to case 0' % i)

	with open(os.path.join(out, 'input_list.txt'), 'w', newline='\n') as fh:
		fh.write('\n'.join(lines) + '\n')
	print('  wrote %d inputs + refs -> work/device/io/%s/' % (n, args.name))


if __name__ == '__main__':
	main()
