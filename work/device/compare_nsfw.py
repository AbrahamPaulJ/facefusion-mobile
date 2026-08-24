"""Device-vs-host comparison for the `nsfw_2` content gate.

⚠ Tensor SNR is the WRONG metric here, for the same reason it was wrong for the detector
(trap #9): the graph's job is a yes/no decision, and the number that matters is whether the
decision moves.  So this reports, in order of importance:

  1. verdict agreement -- how many of the held-out frames the device classifies the same
     way the fp32 host does, at upstream's +0.25 threshold;
  2. score shift -- |device - host| on `logit[0] - logit[1]`, in the same units as the
     threshold, which says how much margin quantisation eats;
  3. margin -- how far the held-out frames actually sit from the threshold, which is what
     decides whether (1) proves anything at all.

⚠⚠ (3) is the limitation to read first.  Our only footage is SFW and scores around -0.9,
about 1.2 below the threshold.  A shift of 0.02 cannot flip a verdict 1.2 away, so 100%
agreement here is close to guaranteed and is NOT evidence that the gate agrees on content
near the boundary.  Only a labelled set spanning the threshold could show that.  Report the
shift, not the agreement, as the finding.
"""
import glob
import os
import sys

import numpy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
THRESHOLD = 0.25          # content_analyser.py:detect_with_nsfw_2


def score(v):
	return float(v[0] - v[1])


def main():
	ref = sorted(glob.glob(os.path.join(ROOT, 'device', 'io', 'nsfw', 'ref', '*.raw')))
	outs = sorted(glob.glob(os.path.join(ROOT, 'device', 'out', 'nsfw', 'Result_*', '*.raw')),
				  key=lambda p: int(os.path.basename(os.path.dirname(p)).split('_')[1]))
	if not ref or not outs:
		sys.exit('missing ref or device outputs -- run make_io.py and run_model.sh nsfw acc')
	n = min(len(ref), len(outs))

	rows = []
	for i in range(n):
		h = numpy.fromfile(ref[i], dtype=numpy.float32)
		d = numpy.fromfile(outs[i], dtype=numpy.float32)
		rows.append((score(h), score(d)))

	hs = numpy.array([r[0] for r in rows])
	ds = numpy.array([r[1] for r in rows])
	shift = numpy.abs(ds - hs)
	agree = int(((hs > THRESHOLD) == (ds > THRESHOLD)).sum())
	margin = numpy.abs(hs - THRESHOLD)

	print('%-4s %10s %10s %9s' % ('#', 'host', 'device', 'shift'))
	for i, (h, d) in enumerate(rows):
		print('%-4d %+10.4f %+10.4f %9.4f' % (i, h, d, abs(d - h)))
	print()
	signed = ds - hs
	ratio = margin.min() / max(shift.max(), 1e-9)
	print('verdict agreement : %d/%d' % (agree, n))
	print('score shift       : mean %+.4f signed, %.4f abs, max %.4f'
		  % (signed.mean(), shift.mean(), shift.max()))
	print('                    %d/%d frames shifted TOWARD nsfw' % (int((signed > 0).sum()), n))
	print('threshold         : +%.2f' % THRESHOLD)
	print('closest frame     : %.4f away, i.e. %.1fx the largest shift'
		  % (margin.min(), ratio))
	print()
	# Deliberately no pass/fail on the agreement. With every frame this far out, agreement
	# is arithmetically forced; a label would dress up a tautology as a result.
	print('READ THIS: with the nearest frame %.1fx further from the threshold than the'
		  % ratio)
	print('           largest shift, agreement is guaranteed, not evidence. The finding is')
	print('           the shift: %.4f against a %.2f threshold, and one-directional.'
		  % (shift.max(), THRESHOLD))
	return 0 if agree == n else 1


if __name__ == '__main__':
	sys.exit(main())
