"""Where the non-NPU time goes, per frame.

The NPU side measures 5.63 ms/frame for detect + landmark + recognise, so the CPU-side
geometry and video I/O are now the suspected bottleneck -- and none of it was measured.

This is host x86, NOT the phone's CPU, so the absolute numbers do not transfer.  What DOES
transfer is the relative cost of the stages and the per-frame pixel volume each one moves,
which is what decides whether a stage needs to be C++/NEON or can stay in Kotlin.

    py -3.10 work/pipeline/profile_cpu.py --frames 30
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy

sys.path.insert(0, os.path.dirname(__file__))
import run_reference as R  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class T:
	def __init__(self):
		self.t = defaultdict(float)
		self.n = defaultdict(int)

	def __call__(self, key):
		return _Ctx(self, key)

	def report(self, frames):
		total = sum(self.t.values())
		print('\n%-26s %10s %8s %9s' % ('stage', 'ms/frame', 'calls', '% of CPU'))
		print('-' * 58)
		for k in sorted(self.t, key=lambda k: -self.t[k]):
			print('%-26s %10.2f %8d %8.1f%%' %
				  (k, self.t[k] * 1000 / frames, self.n[k], 100 * self.t[k] / total))
		print('-' * 58)
		print('%-26s %10.2f' % ('TOTAL', total * 1000 / frames))


class _Ctx:
	def __init__(self, parent, key):
		self.p, self.k = parent, key

	def __enter__(self):
		self.s = time.perf_counter()

	def __exit__(self, *a):
		self.p.t[self.k] += time.perf_counter() - self.s
		self.p.n[self.k] += 1


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--source', default=os.path.join(ROOT, 'assets', 'source.jpg'))
	ap.add_argument('--target', default=os.path.join(ROOT, 'assets', 'target-720p.mp4'))
	ap.add_argument('--frames', type=int, default=30)
	args = ap.parse_args()

	models = R.Models('hyperswap_1a_256')
	t = T()
	spec = R.SWAPPER_SPEC['hyperswap_1a_256']

	src = cv2.imread(args.source)
	sfaces = R.get_many_faces(models, src)
	source_face = max(sfaces, key=lambda f: (f.bounding_box[2] - f.bounding_box[0]) *
											(f.bounding_box[3] - f.bounding_box[1]))

	cap = cv2.VideoCapture(args.target)
	frames_done = 0
	while frames_done < args.frames:
		with t('video decode (cv2)'):
			ok, frame = cap.read()
		if not ok:
			break

		with t('detect: resize+pad+norm'):
			temp = R.restrict_frame(frame, R.FACE_DETECTOR_SIZE)
			det = numpy.zeros((640, 640, 3))
			det[:temp.shape[0], :temp.shape[1], :] = temp
			det = numpy.expand_dims(det.transpose(2, 0, 1), 0).astype(numpy.float32) / 255.0
		with t('NN yoloface (ORT cpu)'):
			out = models.yoloface.run(None, {'input': det})[0]
		with t('detect: decode 8400'):
			detection = numpy.squeeze(out).T
			b, s, l = numpy.split(detection, [4, 5], axis=1)
			keep = numpy.where(s > R.FACE_DETECTOR_SCORE)[0]
		boxes, scores, lms = [], [], []
		if keep.size:
			ratio_h = frame.shape[0] / temp.shape[0]
			ratio_w = frame.shape[1] / temp.shape[1]
			b, s, l = b[keep], s[keep], l[keep]
			for bb in b:
				boxes.append(numpy.array([(bb[0] - bb[2] / 2) * ratio_w, (bb[1] - bb[3] / 2) * ratio_h,
										  (bb[0] + bb[2] / 2) * ratio_w, (bb[1] + bb[3] / 2) * ratio_h]))
			scores = s.ravel().tolist()
			l[:, 0::3] *= ratio_w
			l[:, 1::3] *= ratio_h
			for lm in l:
				lms.append(numpy.array(lm.reshape(-1, 3)[:, :2]))
		with t('NMS'):
			keep2 = R.apply_nms(boxes, scores, R.FACE_DETECTOR_SCORE, 0.4) if boxes else []

		for idx in numpy.array(keep2).ravel().tolist():
			box, lm5 = boxes[idx], lms[idx]
			with t('NN fan_68_5 (ORT cpu)'):
				lm68_5 = models.fan_68_5.run(None, {'input': [lm5.astype(numpy.float32)]})[0][0]
			with t('angle + rot matrix'):
				angle = R.estimate_face_angle(lm68_5)
				rot, rot_size = R.create_rotation_matrix_and_size(angle, (256, 256))
			with t('warp 256 (landmarker)'):
				scale = 195 / numpy.subtract(box[2:], box[:2]).max().clip(1, None)
				tr = (256 - numpy.add(box[2:], box[:2]) * scale) * 0.5
				crop, affine = R.warp_face_by_translation(frame, tr, scale, (256, 256))
				crop = cv2.warpAffine(crop, rot, rot_size)
			with t('CLAHE check (Lab x2)'):
				crop = R.conditional_optimize_contrast(crop)
			with t('NN 2dfan4 (ORT cpu)'):
				_, hm = models.fan2d.run(None, {'input': [crop.transpose(2, 0, 1).astype(numpy.float32) / 255.0]})
			with t('heatmap soft-argmax'):
				lm64, peaks = R.decode_heatmaps(hm)
				lm68 = lm64 / 64 * 256
				lm68 = R.transform_points(lm68, cv2.invertAffineTransform(rot))
				lm68 = R.transform_points(lm68, cv2.invertAffineTransform(affine))
				score68 = numpy.interp(numpy.mean(peaks), [0, 0.9], [0, 1])
				lm5_68 = R.convert_to_face_landmark_5(lm68) if score68 > R.FACE_LANDMARKER_SCORE else lm5
			with t('warp 112 (arcface)'):
				ac, _ = R.warp_face_by_face_landmark_5(frame, lm5_68, 'arcface_112_v2', (112, 112))
				ac = ((ac / 127.5 - 1.0)[:, :, ::-1].transpose(2, 0, 1)).astype(numpy.float32)[None]
			with t('NN arcface (ORT cpu)'):
				emb = models.arcface.run(None, {'input': ac})[0].ravel()
			with t('warp 256 (swapper)'):
				sc, aff = R.warp_face_by_face_landmark_5(frame, lm5_68, spec['template'], spec['size'])
				prep = ((sc[:, :, ::-1] / 255.0 - spec['mean']) / spec['std'])
				prep = numpy.expand_dims(prep.transpose(2, 0, 1), 0).astype(numpy.float32)
			with t('embedding blend'):
				se = source_face.embedding_norm.reshape(1, -1)
				se = R.balance_source_embedding(se, emb)
			with t('NN hyperswap (ORT cpu)'):
				o = models.swapper.run(None, {'source': se, 'target': prep})[0][0]
			with t('box mask + blur'):
				mask = R.create_box_mask(spec['size'], R.FACE_MASK_BLUR, R.FACE_MASK_PADDING)
			with t('paste_back'):
				o = o.transpose(1, 2, 0) * spec['std'] + spec['mean']
				o = o.clip(0, 1)[:, :, ::-1] * 255
				frame = R.paste_back(frame, o, mask, aff)
		frames_done += 1
	cap.release()

	print('frames: %d  (%dx%d)' % (frames_done, frame.shape[1], frame.shape[0]))
	t.report(frames_done)
	nn = sum(v for k, v in t.t.items() if k.startswith('NN '))
	print('\nNN on host CPU: %.1f%% of the total -- on device these move to the NPU'
		  % (100 * nn / sum(t.t.values())))
	print('everything else is the CPU work the port still has to carry: %.2f ms/frame here'
		  % ((sum(t.t.values()) - nn) * 1000 / frames_done))


if __name__ == '__main__':
	main()
