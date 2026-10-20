"""A WIDER calibration set for the LivePortrait extractors (docs/roadmap.md 10).

The first pass calibrated on the 6 `edtalk_target` crops, all one face in one pose, and the
held-out pose fell OUTSIDE what that calibrated: every held-out yaw (23.3 deg) is above the
encoded [0, 22.4] and every roll (-2.5) below [-0.9, 0], so the quantised outputs clip.
Six crops of one head cannot cover a pose output whatever the bit width.

This takes every face in the repo's own media, every Nth frame of the clips, at BOTH
framings -- face_editor's (landmark-5 scaled 1.5 about the nose, which is what the app would
feed) and edtalk's unscaled one (which is what the held-out set is) -- and the horizontal
mirror of each, which is the cheapest way to see the opposite yaw and roll. Crops within
a hair of a held-out crop are dropped, so scoring stays held-out.

    py -3.10 work/device/make_lp_calib.py [--stride 6]
"""
import argparse
import glob
import os
import sys

import cv2
import numpy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'pipeline'))
import run_reference as ref  # noqa: E402

MEDIA = os.path.join(ROOT, '..', 'media', 'source')
OUT = os.path.join(ROOT, 'calib', 'lp_crops')


def scale_face_landmark_5(landmark_5, scale):
	"""facefusion/face_helper.py scale_face_landmark_5: about the nose tip."""
	return (landmark_5 - landmark_5[2]) * scale + landmark_5[2]


def crop(frame, landmark_5):
	"""face_editor prepare_crop_frame: ffhq_512, INTER_AREA to 256, RGB, [0,1], NCHW."""
	c, _ = ref.warp_face_by_face_landmark_5(frame, landmark_5, 'ffhq_512', (512, 512))
	c = cv2.resize(c, (256, 256), interpolation=cv2.INTER_AREA)
	return (c[:, :, ::-1].astype(numpy.float32) / 255.0).transpose(2, 0, 1)


def frames(stride):
	for p in sorted(glob.glob(os.path.join(MEDIA, '*.jpg'))):
		f = cv2.imread(p)
		if f is not None:
			yield os.path.basename(p), f
	for p in sorted(glob.glob(os.path.join(MEDIA, '*.mp4'))):
		cap = cv2.VideoCapture(p)
		i = 0
		while True:
			ok, f = cap.read()
			if not ok:
				break
			if i % stride == 0:
				yield '%s#%d' % (os.path.basename(p), i), f
			i += 1


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--stride', type=int, default=6)
	args = ap.parse_args()
	held = [numpy.fromfile(c, numpy.float32).reshape(3, 256, 256) for c in
	        sorted(glob.glob(os.path.join(ROOT, 'calib', 'edtalk_target_heldout', '*.raw')))]
	if not held:
		raise SystemExit('no held-out crops under work/calib/edtalk_target_heldout')
	os.makedirs(OUT, exist_ok=True)
	for f in glob.glob(os.path.join(OUT, '*.raw')):
		os.remove(f)
	models = ref.Models()
	n = dropped = 0
	for tag, frame in frames(args.stride):
		for face in ref.get_many_faces(models, frame):
			for scale in (1.0, 1.5):
				lm = scale_face_landmark_5(face.landmark_5_68.astype(numpy.float64), scale)
				x = crop(frame, lm)
				for mirrored in (False, True):
					y = numpy.ascontiguousarray(x[:, :, ::-1]) if mirrored else x
					if min(float(((y - h) ** 2).mean()) for h in held) < 1e-4:
						dropped += 1
						continue
					y.tofile(os.path.join(OUT, 'lp_crop_%04d.raw' % n))
					n += 1
		print('\r%-24s %d crops' % (tag, n), end='', flush=True)
	print('\n%d crops -> %s (%d dropped as near a held-out crop)' % (n, OUT, dropped))
	mnt = '/mnt/c/' + OUT[3:].replace('\\', '/')
	with open(os.path.join(ROOT, 'calib', 'lp_motion_list.txt'), 'w', newline='\n') as f:
		f.write(''.join('input:=%s/lp_crop_%04d.raw\n' % (mnt, i) for i in range(n)))


if __name__ == '__main__':
	main()
