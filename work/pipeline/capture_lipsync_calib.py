"""Calibration capture for the lip syncer, from real frames and real audio.

`make_lipsync_calib.py` built stand-in inputs so the CONVERSION could be verified before
the crop chain existed: fan2d crops downscaled to 96, which are whole faces rather than
mouth boxes. Those are fine for asking "does the context binary compute onnxruntime's
function" and useless for quantisation, where the encodings are fitted to the input
distribution and a wrong distribution is a wrong graph (trap #4).

This produces the REAL thing:

  target  the frame -> ffhq_512 by landmark-5 -> the 68 points transformed INTO that crop
          -> create_bounding_box -> warp to 96x96 -> masked half concatenated with the
          reference half, exactly as lip_syncer/core.py:prepare_crop_frame does
  source  the mel windows mel_reference.py produces, which are upstream's own arithmetic

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

MODEL_SIZE = 96
CROP_SIZE = 512


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


def prepare_crop_frame(area_frame):
	"""lip_syncer/core.py:prepare_crop_frame, the wav2lip branch.

	The masked copy comes FIRST and the reference second, and the mask is the BOTTOM half
	-- upstream zeroes `prepare_vision_frame[:, 48:]` on an HWC array.
	"""
	frame = numpy.expand_dims(area_frame, axis=0)
	masked = frame.copy()
	masked[:, MODEL_SIZE // 2:] = 0
	both = numpy.concatenate((masked, frame), axis=3)
	return both.transpose(0, 3, 1, 2).astype(numpy.float32) / 255.0


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--target', required=True)
	ap.add_argument('--tw', type=int, required=True)
	ap.add_argument('--th', type=int, required=True)
	ap.add_argument('--frames', type=int, default=12)
	ap.add_argument('--audio', required=True)
	ap.add_argument('--fps', type=float, default=24.0)
	ap.add_argument('--out', required=True)
	args = ap.parse_args()

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
		# cv2.transform(landmark_68, affine_matrix): the 68 points INTO crop space.
		lm = cv2.transform(face.landmark_68.reshape(1, -1, 2), affine).reshape(-1, 2)
		box = create_bounding_box(lm)
		area, _ = warp_face_by_bounding_box(crop, box, MODEL_SIZE)
		targets.append(prepare_crop_frame(area))
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

	n = len(targets)
	for sub in ('lipsync_source', 'lipsync_target'):
		os.makedirs(os.path.join(args.out, sub), exist_ok=True)
	for i in range(n):
		w = mel.prepare_audio_frame(windows[(i * len(windows)) // n])
		numpy.ascontiguousarray(targets[i], numpy.float32).tofile(
			os.path.join(args.out, 'lipsync_target', 'lipsync_target_%03d.raw' % i))
		numpy.ascontiguousarray(w, numpy.float32).tofile(
			os.path.join(args.out, 'lipsync_source', 'lipsync_source_%03d.raw' % i))

	stack = numpy.concatenate(targets, axis=0)
	print('%d cases -> %s/lipsync_{source,target}/' % (n, args.out))
	print('target  min %.4f max %.4f mean %.4f' % (stack.min(), stack.max(), stack.mean()))
	print('source  %d mel windows available, %d used' % (len(windows), n))


if __name__ == '__main__':
	main()
