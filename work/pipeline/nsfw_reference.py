"""The `nsfw_2` content gate: host reference, calibration capture, and surgery check.

Ported line for line from `facefusion/content_analyser.py` (3.8.2).  Every constant here
is upstream's, and the source line is cited beside it -- this file is the golden reference
the device build is measured against, exactly as `run_reference.py` is for the swap chain.

⚠ Upstream does NOT decide with one model.  `detect_nsfw` is a 2-of-3 vote:

    is_nsfw_1 and is_nsfw_2 or is_nsfw_1 and is_nsfw_3 or is_nsfw_2 and is_nsfw_3

over `nsfw_1` (80.4 MB, 640x640), `nsfw_2` (22.5 MB, 384x384) and `nsfw_3` (358.2 MB,
448x448).  Shipping all three is 461 MB of payload against a 266 MB app, so this port
gates on `nsfw_2` ALONE.  That is a deliberate divergence, not an oversight: a single
model cannot reproduce a majority vote, and the direction of the difference is unmeasured
until there is a labelled set to measure it on.  See docs/roadmap.md 2.

⚠ The preprocessing is NOT the detector's.  `fit_contain_frame` scales to fit and pads
CENTRED; `detect_faces` scales and pads into the top-left corner.  Reusing the detector's
input path here would letterbox the frame differently and move every score.

Usage:
    py -3.10 work/pipeline/nsfw_reference.py score  work/assets/target-720p.mp4
    py -3.10 work/pipeline/nsfw_reference.py calib  work/assets/target-720p.mp4 work/calib/nsfw 2 128
    py -3.10 work/pipeline/nsfw_reference.py calib  work/assets/target-720p.mp4 work/calib/nsfw_heldout 2 16 1
    py -3.10 work/pipeline/nsfw_reference.py verify work/assets/target-720p.mp4
"""
import os
import sys

import cv2
import numpy
import onnxruntime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# content_analyser.py:create_static_model_set -- nsfw_2
MODEL_SIZE = (384, 384)
# ⚠ 0.5/0.5, NOT 0/1.  facefusion 3.8.2 content_analyser.py gives nsfw_2
# `mean (0.5,0.5,0.5)` and `standard_deviation (0.5,0.5,0.5)`, i.e. [0,1] -> [-1,1].
# This port used 0/1 from the gate's first commit and fed the model [0,1] instead.
#
# It survived every check because ffpipe.cpp did it the same wrong way, so device-vs-host
# agreement was never evidence about UPSTREAM -- both sides were wrong together. Measured
# cost on 12 real frames: the decision statistic moves by -1.15 on average, against a
# threshold of 0.25.  Corrected 2026-08-30.
MODEL_MEAN = (0.5, 0.5, 0.5)
MODEL_STD = (0.5, 0.5, 0.5)
# content_analyser.py:detect_with_nsfw_2
SCORE_THRESHOLD = 0.25
# content_analyser.py:analyse_video -- a video is refused when more than this percentage
# of the SAMPLED frames trip the gate.  Sampling is one frame per second, not every frame.
VIDEO_RATE_THRESHOLD = 10.0

RAW = os.path.join(ROOT, 'models', 'nsfw_2.onnx')
SIM = os.path.join(ROOT, 'onnx', 'nsfw_2_sim.onnx')


def fit_contain_frame(vision_frame, resolution):
	"""vision.py:fit_contain_frame -- scale to fit, then pad CENTRED with zeros."""
	contain_width, contain_height = resolution
	height, width = vision_frame.shape[:2]
	scale = min(contain_height / height, contain_width / width)
	new_width = int(width * scale)
	new_height = int(height * scale)
	start_x = max(0, (contain_width - new_width) // 2)
	start_y = max(0, (contain_height - new_height) // 2)
	end_x = max(0, contain_width - new_width - start_x)
	end_y = max(0, contain_height - new_height - start_y)
	temp = cv2.resize(vision_frame, (new_width, new_height))
	return numpy.pad(temp, ((start_y, end_y), (start_x, end_x), (0, 0)))


def prepare_detect_frame(vision_frame):
	"""content_analyser.py:prepare_detect_frame for nsfw_2.  BGR in, NCHW float32 out."""
	frame = fit_contain_frame(vision_frame, MODEL_SIZE)
	frame = frame[:, :, ::-1] / 255.0          # BGR -> RGB, to [0, 1]
	frame -= MODEL_MEAN
	frame /= MODEL_STD
	return numpy.expand_dims(frame.transpose(2, 0, 1), axis=0).astype(numpy.float32)


def score_of(detection):
	"""content_analyser.py:detect_with_nsfw_2 -- logit[0] - logit[1], NSFW when > 0.25.

	`forward_nsfw` takes detection[0] for nsfw_2, i.e. the single batch row of the [1, 2]
	output, so `detection` here is the length-2 logit vector.
	"""
	return float(detection[0] - detection[1])


def session(path=SIM):
	return onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])


def run(sess, frame_bgr):
	x = prepare_detect_frame(frame_bgr)
	out = sess.run(None, {'input': x})[0]
	return score_of(out[0]), x


def frames(video, stride=1, limit=None, offset=0):
	"""`offset` selects a phase of the stride, which is how the held-out set is made
	DISJOINT from the calibration set rather than merely later in the same clip: even
	frames calibrate, odd frames are never seen by the quantiser."""
	cap = cv2.VideoCapture(video)
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	i = kept = 0
	while limit is None or kept < limit:
		ok, frame = cap.read()
		if not ok:
			break
		if i % stride == offset % stride:
			yield i, fps, frame
			kept += 1
		i += 1
	cap.release()


def cmd_score(video):
	"""Upstream's video verdict: sample one frame per second, refuse above a 10% rate."""
	sess = session()
	scores = []
	for i, fps, frame in frames(video):
		if i % int(fps) == 0:
			scores.append((i, run(sess, frame)[0]))
	flagged = [s for _, s in scores if s > SCORE_THRESHOLD]
	rate = 100.0 * len(flagged) / max(1, len(scores))
	print('sampled %d frames at 1/s' % len(scores))
	for i, s in scores:
		print('  frame %4d  score %+8.4f  %s' % (i, s, 'NSFW' if s > SCORE_THRESHOLD else ''))
	print('rate %.1f%% (threshold %.1f%%) -> %s'
		  % (rate, VIDEO_RATE_THRESHOLD,
			 'REFUSE' if rate > VIDEO_RATE_THRESHOLD else 'allow'))


def cmd_calib(video, outdir, stride, limit, offset=0):
	"""trap #4: calibrate on REAL letterboxed frames, sampled across the whole clip."""
	os.makedirs(outdir, exist_ok=True)
	sess = session()
	scores = []
	kept = 0
	for _, _, frame in frames(video, stride, limit, offset):
		s, x = run(sess, frame)
		numpy.ascontiguousarray(x[0]).tofile(os.path.join(outdir, 'nsfw_%04d.raw' % kept))
		scores.append(s)
		kept += 1
	a = numpy.array(scores)
	print('wrote %d tensors of shape (3, 384, 384) to %s' % (kept, outdir))
	print('score min %+.4f  max %+.4f  mean %+.4f  flagged %d/%d'
		  % (a.min(), a.max(), a.mean(), int((a > SCORE_THRESHOLD).sum()), kept))


def cmd_verify(video, n=16):
	"""Assert the simplified graph is the graph we downloaded (verify_surgery.py's job).

	Both the logits and the derived verdict are compared: a fold that moved a logit by
	1e-6 is fine, one that flips a decision is not, and only the second is visible in the
	metric that matters.
	"""
	raw, sim = session(RAW), session(SIM)
	worst = 0.0
	flips = 0
	rows = 0
	for _, _, frame in frames(video, stride=8, limit=n):
		x = prepare_detect_frame(frame)
		a = raw.run(None, {'input': x})[0][0]
		b = sim.run(None, {'input': x})[0][0]
		worst = max(worst, float(numpy.abs(a - b).max()))
		if (score_of(a) > SCORE_THRESHOLD) != (score_of(b) > SCORE_THRESHOLD):
			flips += 1
		rows += 1
	print('%d frames: max abs logit diff %.3e, verdict flips %d' % (rows, worst, flips))
	ok = worst < 1e-4 and flips == 0
	print('nsfw surgery:', 'OK' if ok else 'FAILED')
	return 0 if ok else 1


if __name__ == '__main__':
	cmd = sys.argv[1] if len(sys.argv) > 1 else 'score'
	if cmd == 'score':
		cmd_score(sys.argv[2])
	elif cmd == 'calib':
		cmd_calib(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]),
				  int(sys.argv[6]) if len(sys.argv) > 6 else 0)
	elif cmd == 'verify':
		sys.exit(cmd_verify(sys.argv[2]))
	else:
		sys.exit(__doc__)
