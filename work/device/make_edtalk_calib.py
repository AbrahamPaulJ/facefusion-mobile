"""Build `edtalk_256`'s three device inputs from what the repo already has.

edtalk takes a DIFFERENT set from wav2lip, and the difference is the whole reason it is
being converted:

    source  (1, 1, 80, 16)    the mel window -- same tensor wav2lip takes
    target  (1, 3, 256, 256)  the FULL face crop, at 256, not a masked 96 mouth box
    weight  (1,)              the lip-direction scale; upstream drives it at 1.0

So the fan2d calibration crops feed it DIRECTLY: they are already real 256x256 face crops
in [0, 1], and no downscale or channel-stacking is needed.  wav2lip needed
`make_lipsync_calib.py` to build a 6-channel 96x96 masked/reference pair; edtalk needs
none of that, because 256x256 IS the crop.

⚠ Same caveat as wav2lip's, and it still matters: a full fan2d crop is not the crop the app
will feed.  The real one comes from the 68 landmarks through create_bounding_box, and it is
tighter and lower on the face.  What this set proves is that the context binary computes the
same function as onnxruntime; it says nothing about where the mouth lands (trap #9).

    py -3.10 work/device/make_edtalk_calib.py
"""
import glob
import os
import sys

import numpy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ORACLE = os.path.join(ROOT, 'assets', 'melref')

CROP_SIZE = 256
MEL_FILTER_TOTAL = 80
AUDIO_STEP_SIZE = 16
LIP_WEIGHT = 1.0


def main():
	crops = sorted(glob.glob(os.path.join(ROOT, 'calib', 'fan2d', '*.raw')))
	if not crops:
		sys.exit('no fan2d crops -- nothing to build a target from')
	frames_path = os.path.join(ORACLE, 'frames.f32')
	if not os.path.exists(frames_path):
		sys.exit('no oracle at %s -- run mel_reference.py dump first' % frames_path)

	windows = numpy.fromfile(frames_path, numpy.float32)
	windows = windows.reshape(-1, MEL_FILTER_TOTAL, AUDIO_STEP_SIZE)
	n = min(len(crops), len(windows))
	print('%d cases: %d fan2d crops, %d mel windows available' % (n, len(crops), len(windows)))

	for sub in ('edtalk_source', 'edtalk_target', 'edtalk_weight'):
		os.makedirs(os.path.join(ROOT, 'calib', sub), exist_ok=True)

	for i in range(n):
		target = numpy.fromfile(crops[i], numpy.float32).reshape(1, 3, CROP_SIZE, CROP_SIZE)
		# Strided across the clip, not consecutive: mel windows 200 ms apart are nearly
		# identical and a set of near-duplicates would hide a device fault (trap #8).
		source = windows[(i * len(windows)) // n][None, None]
		weight = numpy.array([LIP_WEIGHT], numpy.float32)

		target.astype(numpy.float32).tofile(
			os.path.join(ROOT, 'calib', 'edtalk_target', 'edtalk_target_%03d.raw' % i))
		source.astype(numpy.float32).tofile(
			os.path.join(ROOT, 'calib', 'edtalk_source', 'edtalk_source_%03d.raw' % i))
		weight.tofile(
			os.path.join(ROOT, 'calib', 'edtalk_weight', 'edtalk_weight_%03d.raw' % i))

	print('  target (1, 3, %d, %d), source (1, 1, %d, %d), weight (1,) -> work/calib/edtalk_*/'
		  % (CROP_SIZE, CROP_SIZE, MEL_FILTER_TOTAL, AUDIO_STEP_SIZE))
	print('  now: py -3.10 work/qnn/make_calib_lists.py edtalk   (it writes WSL paths)')


if __name__ == '__main__':
	main()
