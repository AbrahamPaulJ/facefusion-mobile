"""Calibration capture for the lip syncer, from real frames and real audio.

`make_lipsync_calib.py` built stand-in inputs so the CONVERSION could be verified before
the crop chain existed: fan2d crops downscaled to 96, which are whole faces rather than
mouth boxes. Those are fine for asking "does the context binary compute onnxruntime's
function" and useless for quantisation, where the encodings are fitted to the input
distribution and a wrong distribution is a wrong graph (trap #4).

This produces the REAL thing, and the two models take a DIFFERENT crop -- checked against
lip_syncer/core.py, not assumed, after a wrong assumption here fed a wrong distribution to
edtalk's W8A16 calibration for two sessions running (2026-09-04):

  wav2lip  frame -> ffhq_512 by landmark-5 -> the 68 points transformed INTO that crop ->
           create_bounding_box -> warp to 96x96 -> masked half concatenated with the
           reference half, BGR throughout
  edtalk   frame -> ffhq_512 by landmark-5 -> the WHOLE 512 crop resized (not warped) to
           256x256 -> RGB, no mask, no concatenation -- upstream's `sync_lip` never
           computes a bounding box for this model at all
  source   the mel windows mel_reference.py produces, which are upstream's own arithmetic

Everything about the face comes from `run_reference`, which is the golden host path -- this
file must never reimplement a stage of it. What it does add is the three upstream helpers
the swap chain never needed, each cited.

⚠ The audio is `synth`'s speech-like signal, not a recording, and it has nothing to do with
the faces in the frames. That is acceptable for CALIBRATION, which needs the input RANGE
rather than semantic agreement, and it is not acceptable for an accuracy claim. Say which
you are making.

    py -3.10 work/pipeline/capture_lipsync_calib.py \\
        --target work/device/raw/target12.bgr --tw 1366 --th 720 --frames 12 \\
        --audio work/assets/lipsync_test.wav --fps 24 --out work/calib
"""
import argparse
import os
import sys

import cv2
import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mel_reference as mel
import run_reference as ref

CROP_SIZE = 512
# name -> (the size create_bounding_box's box is warped to, the tensor's channel count,
#          the calib subdir stem)
MODELS = {
	'wav2lip': (96, 6, 'lipsync'),
	'edtalk': (256, 3, 'edtalk'),
}


def create_bounding_box(landmark_68):
	"""face_helper.py:150 -- min/max over all 68, then normalise so x1 <= x2."""
	x1, y1 = numpy.min(landmark_68, axis=0)
	x2, y2 = numpy.max(landmark_68, axis=0)
	return numpy.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])


def warp_face_by_bounding_box(frame, bounding_box, crop_size):
	"""face_helper.py:83. INTER_AREA when the box is bigger than the crop, which
	cv2.warpAffine silently ignores -- see ffcv.h's header, measured bit-identical."""
	source_points = numpy.array([
		[bounding_box[0], bounding_box[1]],
		[bounding_box[2], bounding_box[1]],
		[bounding_box[0], bounding_box[3]]]).astype(numpy.float32)
	target_points = numpy.array([[0, 0], [crop_size, 0], [0, crop_size]]).astype(numpy.float32)
	affine = cv2.getAffineTransform(source_points, target_points)
	method = cv2.INTER_AREA if (bounding_box[2] - bounding_box[0] > crop_size or
	                            bounding_box[3] - bounding_box[1] > crop_size) \
		else cv2.INTER_LINEAR
	return cv2.warpAffine(frame, affine, (crop_size, crop_size), flags=method), affine


def prepare_crop_frame(area_frame, model):
	"""lip_syncer/core.py:prepare_crop_frame.

	wav2lip: the masked copy comes FIRST and the reference second, and the mask is the
	BOTTOM half -- upstream zeroes `prepare_vision_frame[:, 48:]` on an HWC array. BGR
	throughout -- upstream never flips it for this model.

	edtalk: no mask and no concatenation. It is a generator over the WHOLE face, not an
	inpainter for a hidden mouth. ⚠ RGB, not BGR -- `crop_vision_frame[:, :, ::-1] / 255.0`
	in the real lip_syncer/core.py. A session that ported this without checking upstream
	fed it BGR for two sessions and calibrated THIS SCRIPT the same wrong way; both are
	fixed together here, since a calibration set in the wrong channel order fits the W8A16
	encodings to a distribution the corrected app never sends it.
	"""
	size, _, _ = MODELS[model]
	if model == 'edtalk':
		area_frame = area_frame[:, :, ::-1]
	frame = numpy.expand_dims(area_frame, axis=0)
	if model == 'wav2lip':
		masked = frame.copy()
		masked[:, size // 2:] = 0
		frame = numpy.concatenate((masked, frame), axis=3)
	return frame.transpose(0, 3, 1, 2).astype(numpy.float32) / 255.0


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--model', choices=sorted(MODELS), default='wav2lip')
	ap.add_argument('--target', required=True)
	ap.add_argument('--tw', type=int, required=True)
	ap.add_argument('--th', type=int, required=True)
	ap.add_argument('--frames', type=int, default=12)
	ap.add_argument('--audio', required=True)
	ap.add_argument('--fps', type=float, default=24.0)
	ap.add_argument('--out', required=True)
	args = ap.parse_args()
	size, _, stem = MODELS[args.model]

	models = ref.Models('hyperswap_1a_256')
	raw = numpy.fromfile(args.target, numpy.uint8)
	need = args.tw * args.th * 3 * args.frames
	if raw.size < need:
		sys.exit('%s holds %d bytes, %d frames needs %d' % (args.target, raw.size,
		                                                    args.frames, need))
	frames = raw[:need].reshape(args.frames, args.th, args.tw, 3)

	targets = []
	for i in range(args.frames):
		frame = frames[i].copy()
		faces = ref.get_many_faces(models, frame)
		if not faces:
			print('  frame %2d: no face' % (i + 1))
			continue
		face = max(faces, key=lambda f: (f.bounding_box[2] - f.bounding_box[0]) *
		                                (f.bounding_box[3] - f.bounding_box[1]))
		crop, affine = ref.warp_face_by_face_landmark_5(
			frame, face.landmark_5_68, 'ffhq_512', (CROP_SIZE, CROP_SIZE))
		if args.model == 'edtalk':
			# No bounding box for this model -- upstream resizes the WHOLE 512 crop.
			area = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
			targets.append(prepare_crop_frame(area, args.model))
			print('  frame %2d: whole crop -> %dx%d' % (i + 1, size, size))
		else:
			# cv2.transform(landmark_68, affine_matrix): the 68 points INTO crop space.
			lm = cv2.transform(face.landmark_68.reshape(1, -1, 2), affine).reshape(-1, 2)
			box = create_bounding_box(lm)
			area, _ = warp_face_by_bounding_box(crop, box, size)
			targets.append(prepare_crop_frame(area, args.model))
			print('  frame %2d: box %.0f,%.0f %.0fx%.0f' %
			      (i + 1, box[0], box[1], box[2] - box[0], box[3] - box[1]))

	if not targets:
		sys.exit('captured nothing -- no faces in any frame')

	# The mel windows, strided across the clip: consecutive ones are 200 ms apart and
	# nearly identical, and a calibration set of near-duplicates fits the encodings to a
	# narrower range than the device will ever see.
	raw_audio = mel.decode_audio(args.audio)
	import scipy.signal
	resampled = scipy.signal.resample(
		raw_audio, round(len(raw_audio) * mel.VOICE_RESAMPLE_RATE / mel.AUDIO_SAMPLE_RATE))
	audio = mel.prepare_audio(resampled)
	spectrogram = mel.create_spectrogram(audio)
	windows = [w for w in mel.extract_audio_frames(spectrogram, args.fps)
	           if w.shape[1] == mel.AUDIO_STEP_SIZE]
	if not windows:
		sys.exit('no mel windows -- audio too short for %g fps' % args.fps)

	# Split by PARITY, the rule gpen and nsfw already use: the held-out set is the odd
	# frames, so it is disjoint from calibration rather than merely the tail of the same
	# sequence. Consecutive video frames are correlated and this is still a friendly test,
	# but it is at least not the set that trained the encodings.
	n = len(targets)
	# edtalk takes a third input, the lip-direction scale, which upstream drives at 1.0.
	# It is written per case rather than once because --input_list wants one path per
	# input per line, and a shared file would make a case's inputs no longer self-describing.
	names = ['source', 'target'] + (['weight'] if args.model == 'edtalk' else [])
	subs = ['%s_%s%s' % (stem, k, h) for k in names for h in ('', '_heldout')]
	for sub in subs:
		os.makedirs(os.path.join(args.out, sub), exist_ok=True)
	kept = {'': 0, '_heldout': 0}
	for i in range(n):
		w = mel.prepare_audio_frame(windows[(i * len(windows)) // n])
		suffix = '_heldout' if i % 2 else ''
		# The index in the filename stays the CAPTURE index, so a held-out file still
		# says which frame it came from.
		def out(kind):
			return os.path.join(args.out, '%s_%s%s' % (stem, kind, suffix),
			                    '%s_%s_%03d.raw' % (stem, kind, i))
		numpy.ascontiguousarray(targets[i], numpy.float32).tofile(out('target'))
		numpy.ascontiguousarray(w, numpy.float32).tofile(out('source'))
		if args.model == 'edtalk':
			numpy.array([1.0], numpy.float32).tofile(out('weight'))
		kept[suffix] += 1
	print('calibration %d, held out %d' % (kept[''], kept['_heldout']))

	stack = numpy.concatenate(targets, axis=0)
	print('%d cases -> %s/%s_{%s}/' % (n, args.out, stem, ','.join(names)))
	print('target  min %.4f max %.4f mean %.4f' % (stack.min(), stack.max(), stack.mean()))
	print('source  %d mel windows available, %d used' % (len(windows), n))


if __name__ == '__main__':
	main()
