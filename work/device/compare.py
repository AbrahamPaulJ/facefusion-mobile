"""Device-vs-host SNR for one model, over the held-out set.

Pull the device outputs first:
    adb -s <serial> pull /data/local/tmp/ff/out/<name> work/device/out/

Reports per-case and worst-case SNR.  For arcface it also reports COSINE SIMILARITY of the
512-d embedding, which is the number that actually governs whether identity survives --
SNR on an embedding is not the quantity downstream cares about (trap #18: pick the metric
before running the comparison).
"""
import argparse
import glob
import os
import sys

import numpy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def snr(ref, got):
	ref = ref.astype(numpy.float64).ravel()
	got = got.astype(numpy.float64).ravel()
	noise = ((ref - got) ** 2).sum()
	if noise == 0:
		return float('inf')
	return 10 * numpy.log10((ref ** 2).sum() / noise)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('name')
	ap.add_argument('--out-dir', default=None, help='pulled device outputs')
	args = ap.parse_args()

	io_dir = os.path.join(ROOT, 'device', 'io', args.name)
	dev_dir = args.out_dir or os.path.join(ROOT, 'device', 'out', args.name)
	if not os.path.isdir(dev_dir):
		sys.exit('no device outputs at %s -- adb pull first' % dev_dir)

	results = sorted(glob.glob(os.path.join(dev_dir, 'Result_*')),
					 key=lambda p: int(p.rsplit('_', 1)[1]))
	if not results:
		sys.exit('no Result_* dirs in %s' % dev_dir)

	refs = sorted(glob.glob(os.path.join(io_dir, 'ref', '*.raw')))
	if not refs:
		sys.exit('no reference tensors -- run make_io.py first')
	ref_names = sorted({os.path.basename(f).rsplit('_', 1)[0] for f in refs})

	print('%s: %d device results, outputs %s' % (args.name, len(results), ref_names))
	worst = {}
	dev_first = {}
	for i, rdir in enumerate(results):
		for name in ref_names:
			ref_p = os.path.join(io_dir, 'ref', '%s_%03d.raw' % (name, i))
			if not os.path.exists(ref_p):
				continue
			cand = glob.glob(os.path.join(rdir, '*.raw'))
			match = [c for c in cand if os.path.splitext(os.path.basename(c))[0]
					 .replace('/', '_').replace(':', '_') == name]
			if not match:
				match = cand if len(cand) == 1 else []
			if not match:
				print('  case %d: no device tensor matching %s (have %s)'
					  % (i, name, [os.path.basename(c) for c in cand]))
				continue
			ref = numpy.fromfile(ref_p, numpy.float32)
			got = numpy.fromfile(match[0], numpy.float32)
			if ref.size != got.size:
				print('  case %d %s: size %d vs %d' % (i, name, ref.size, got.size))
				continue
			s = snr(ref, got)
			worst[name] = min(worst.get(name, 1e9), s)
			if i == 0:
				dev_first[name] = got.copy()
			elif numpy.array_equal(dev_first.get(name, numpy.zeros(1)), got):
				print('  WARNING case %d %s: bit-identical to case 0 -- see trap #8'
					  % (i, name))
			if args.name == 'arcface':
				cos = float(ref @ got / (numpy.linalg.norm(ref) * numpy.linalg.norm(got)))
				print('  case %2d  SNR %6.2f dB   cosine %.6f' % (i, s, cos))
			else:
				print('  case %2d  %-24s SNR %6.2f dB' % (i, name, s))

	print()
	for name, v in worst.items():
		print('WORST  %-24s %6.2f dB' % (name, v))


if __name__ == '__main__':
	main()
