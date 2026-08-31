"""Build the lip syncer's two device inputs from what the repo already has.

wav2lip takes `source` (1, 1, 80, 16), a mel window, and `target` (1, 6, 96, 96), the
masked mouth crop concatenated with the reference crop.  Neither exists as a capture yet,
because the mouth crop is roadmap 9 step 5 and has not been written.  This makes
REPRESENTATIVE inputs so the CONVERSION can be verified before the geometry lands:

  source  the real mel windows mel_reference.py dumped, which are upstream's own numbers
  target  the fan2d calibration crops, which are real 256x256 face crops already in
          [0, 1], area-downscaled to 96 and stacked the way lip_syncer/core.py does

⚠ These are NOT the crops the app will feed.  The real one comes from the 68-point
landmarks through create_bounding_box + warp_face_by_bounding_box, and it is tighter and
lower on the face than a full fan2d crop.  What this set can prove is that the context
binary computes the same function as onnxruntime; what it cannot prove is that the mouth
lands in the right place.  Trap #9 is exactly that distinction.

    py -3.10 work/device/make_lipsync_calib.py
"""
import glob
import os
import sys

import numpy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ORACLE = os.path.join(ROOT, 'assets', 'melref')

MODEL_SIZE = 96
MEL_FILTER_TOTAL = 80
AUDIO_STEP_SIZE = 16


def area_downscale(chw, size):
	"""256 -> 96 by box average over fractional bounds. Representative, not a reference:
	the app resizes with ffcv, and this file never feeds the app."""
	c, h, w = chw.shape
	out = numpy.zeros((c, size, size), numpy.float32)
	edges = numpy.linspace(0, h, size + 1)
	for y in range(size):
		y0, y1 = int(numpy.floor(edges[y])), int(numpy.ceil(edges[y + 1]))
		for x in range(size):
			x0, x1 = int(numpy.floor(edges[x])), int(numpy.ceil(edges[x + 1]))
			out[:, y, x] = chw[:, y0:y1, x0:x1].mean(axis=(1, 2))
	return out


def main():
	crops = sorted(glob.glob(os.path.join(ROOT, 'calib', 'fan2d', '*.raw')))
	if not crops:
		sys.exit('no fan2d crops -- nothing to build a target from')
	frames_path = os.path.join(ORACLE, 'frames.f32')
	if not os.path.exists(frames_path):
		sys.exit('no oracle at %s -- run mel_reference.py dump first' % ORACLE)

	windows = numpy.fromfile(frames_path, numpy.float32)
	windows = windows.reshape(-1, MEL_FILTER_TOTAL, AUDIO_STEP_SIZE)
	n = min(len(crops), len(windows))
	print('%d cases: %d fan2d crops, %d mel windows available' % (n, len(crops), len(windows)))

	for sub in ('lipsync_source', 'lipsync_target'):
		os.makedirs(os.path.join(ROOT, 'calib', sub), exist_ok=True)

	for i in range(n):
		crop = numpy.fromfile(crops[i], numpy.float32).reshape(3, 256, 256)
		small = area_downscale(crop, MODEL_SIZE)

		# lip_syncer/core.py:prepare_crop_frame, wav2lip branch. Upstream works in HWC and
		# zeroes prepare_vision_frame[:, 48:], which is the BOTTOM half of the image, then
		# concatenates (masked, reference) on the channel axis.
		masked = small.copy()
		masked[:, MODEL_SIZE // 2:, :] = 0.0
		target = numpy.concatenate([masked, small], axis=0)[None]

		# The windows are strided across the clip rather than taken consecutively: mel
		# windows 200 ms apart are nearly identical, and a set of near-duplicates would
		# hide a device fault the way trap #8 describes.
		source = windows[(i * len(windows)) // n][None, None]

		target.astype(numpy.float32).tofile(
			os.path.join(ROOT, 'calib', 'lipsync_target', 'lipsync_target_%03d.raw' % i))
		source.astype(numpy.float32).tofile(
			os.path.join(ROOT, 'calib', 'lipsync_source', 'lipsync_source_%03d.raw' % i))

	print('  target (1, 6, %d, %d), source (1, 1, %d, %d) -> work/calib/lipsync_*/'
	      % (MODEL_SIZE, MODEL_SIZE, MEL_FILTER_TOTAL, AUDIO_STEP_SIZE))


if __name__ == '__main__':
	main()
