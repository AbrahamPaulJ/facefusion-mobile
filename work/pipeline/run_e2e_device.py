"""End-to-end swap where every neural network runs on the NPU.

This is the acceptance test for the port: it produces swapped frames using ONLY device
model outputs, and scores them against the host golden.

The stages are sequential, but frames are independent within a stage, so the whole clip
costs four device invocations rather than four per frame:

    frames --> [yoloface on NPU] --> boxes/landmarks (host decode + NMS)
           --> [2dfan4 on NPU]  --> heatmaps --> landmark_68 (host soft-argmax)
           --> [arcface on NPU] --> embeddings
           --> [hyperswap on NPU] --> crops --> paste_back (host)

The source image's embedding is computed on the device too, so the identity fed to the
swapper is the quantised one -- otherwise the test would flatter itself.

    py -3.10 work/pipeline/run_e2e_device.py --frames 24
"""
import argparse
import os
import sys

import cv2
import numpy

sys.path.insert(0, os.path.dirname(__file__))
import run_reference as R                      # noqa: E402
from device_runner import DeviceRunner          # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


DETECTOR_CTX = 'yoloface'   # override to 'yolofacef' for the fp32 detector


def detect_on_device(dev, frames):
	"""yoloface: batch every frame, then decode + NMS on the host."""
	dets, metas = [], []
	for frame in frames:
		temp = R.restrict_frame(frame, R.FACE_DETECTOR_SIZE)
		det = numpy.zeros((R.FACE_DETECTOR_SIZE[1], R.FACE_DETECTOR_SIZE[0], 3))
		det[:temp.shape[0], :temp.shape[1], :] = temp
		dets.append(det.transpose(2, 0, 1).astype(numpy.float32) / 255.0)
		metas.append((frame.shape[0] / temp.shape[0], frame.shape[1] / temp.shape[1]))

	outs = dev.run('yoloface', {'input': dets}, ctx=DETECTOR_CTX)
	per_frame = []
	for (ratio_h, ratio_w), case in zip(metas, outs):
		raw = list(case.values())[0].reshape(1, 20, 8400)
		detection = numpy.squeeze(raw).T
		boxes_raw, scores_raw, lmk_raw = numpy.split(detection, [4, 5], axis=1)
		keep = numpy.where(scores_raw > R.FACE_DETECTOR_SCORE)[0]
		boxes, scores, lms = [], [], []
		if keep.size:
			boxes_raw, scores_raw, lmk_raw = boxes_raw[keep], scores_raw[keep], lmk_raw[keep]
			for b in boxes_raw:
				boxes.append(numpy.array([
					(b[0] - b[2] / 2) * ratio_w, (b[1] - b[3] / 2) * ratio_h,
					(b[0] + b[2] / 2) * ratio_w, (b[1] + b[3] / 2) * ratio_h]))
			scores = scores_raw.ravel().tolist()
			lmk_raw[:, 0::3] *= ratio_w
			lmk_raw[:, 1::3] *= ratio_h
			for lm in lmk_raw:
				lms.append(numpy.array(lm.reshape(-1, 3)[:, :2]))
		per_frame.append((boxes, scores, lms))
	return per_frame


def faces_on_device(dev, models, frames, detections):
	"""2dfan4 + arcface, batched across every face in every frame."""
	jobs = []           # (frame_index, box, landmark_5, det_score)
	for fi, (boxes, scores, lms) in enumerate(detections):
		if not boxes:
			continue
		keep = R.apply_nms(boxes, scores, R.FACE_DETECTOR_SCORE, 0.4)
		for idx in numpy.array(keep).ravel().tolist():
			jobs.append((fi, boxes[idx], lms[idx], scores[idx]))
	if not jobs:
		return []

	# host: 5 -> 68 (fan_68_5 is 0.9 MB and stays on the CPU), then the 256x256 crop
	crops, geoms = [], []
	for fi, box, lm5, _ in jobs:
		lm68_5 = models.fan_68_5.run(None, {'input': [lm5.astype(numpy.float32)]})[0][0]
		angle = R.estimate_face_angle(lm68_5)
		model_size = (256, 256)
		scale = 195 / numpy.subtract(box[2:], box[:2]).max().clip(1, None)
		translation = (model_size[0] - numpy.add(box[2:], box[:2]) * scale) * 0.5
		rot, rot_size = R.create_rotation_matrix_and_size(angle, model_size)
		crop, affine = R.warp_face_by_translation(frames[fi], translation, scale, model_size)
		crop = cv2.warpAffine(crop, rot, rot_size)
		crop = R.conditional_optimize_contrast(crop)
		crops.append(crop.transpose(2, 0, 1).astype(numpy.float32) / 255.0)
		geoms.append((rot, affine))

	hm_out = dev.run('fan2d', {'input': crops})

	arc_inputs, faces = [], []
	for (fi, box, lm5, dscore), (rot, affine), case in zip(jobs, geoms, hm_out):
		hm = list(case.values())[0].reshape(1, 68, 64, 64)
		lm64, peaks = R.decode_heatmaps(hm)
		lm68 = lm64 / 64 * 256
		lm68 = R.transform_points(lm68, cv2.invertAffineTransform(rot))
		lm68 = R.transform_points(lm68, cv2.invertAffineTransform(affine))
		score68 = numpy.interp(numpy.mean(peaks), [0, 0.9], [0, 1])
		lm5_68 = R.convert_to_face_landmark_5(lm68) if score68 > R.FACE_LANDMARKER_SCORE else lm5

		arc_crop, _ = R.warp_face_by_face_landmark_5(frames[fi], lm5_68, 'arcface_112_v2', (112, 112))
		arc_crop = arc_crop / 127.5 - 1.0
		arc_crop = arc_crop[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)
		arc_inputs.append(arc_crop)
		faces.append(dict(frame=fi, landmark_5_68=lm5_68))

	emb_out = dev.run('arcface', {'input': arc_inputs})
	for face, case in zip(faces, emb_out):
		e = list(case.values())[0].ravel()
		face['embedding'] = e
		face['embedding_norm'] = e / numpy.linalg.norm(e)
	return faces


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--source', default=os.path.join(ROOT, 'assets', 'source.jpg'))
	ap.add_argument('--target', default=os.path.join(ROOT, 'assets', 'target-720p.mp4'))
	ap.add_argument('--frames', type=int, default=24)
	ap.add_argument('--swapper', default='hyperswap_1a_256', choices=list(R.SWAPPER_SPEC))
	ap.add_argument('--detector-ctx', default=None,
					help="device context for yoloface; 'yolofacef' is the fp32 build")
	ap.add_argument('--ctx', default='hyperswap', help='device context name for the swapper')
	ap.add_argument('--out', default=os.path.join(ROOT, 'output', 'e2e_device.mp4'))
	ap.add_argument('--golden', default=None, help='host reference video to score against')
	args = ap.parse_args()
	if args.detector_ctx:
		global DETECTOR_CTX
		DETECTOR_CTX = args.detector_ctx

	models = R.Models(args.swapper)
	dev = DeviceRunner()

	# ---- source identity, computed on the device as well
	source_frame = cv2.imread(args.source)
	sdet = detect_on_device(dev, [source_frame])
	sfaces = faces_on_device(dev, models, [source_frame], sdet)
	if not sfaces:
		sys.exit('no face found in source')
	source_face = sfaces[0]
	print('source embedding norm %.4f (device)' % numpy.linalg.norm(source_face['embedding']))

	# ---- target frames
	cap = cv2.VideoCapture(args.target)
	frames = []
	while len(frames) < args.frames:
		ok, f = cap.read()
		if not ok:
			break
		frames.append(f)
	cap.release()
	print('%d target frames' % len(frames))

	dets = detect_on_device(dev, frames)
	faces = faces_on_device(dev, models, frames, dets)
	print('%d faces across %d frames' % (len(faces), len(frames)))

	# ---- swapper, batched
	spec = R.SWAPPER_SPEC[args.swapper]
	targets, sources, geom = [], [], []
	for face in faces:
		crop, affine = R.warp_face_by_face_landmark_5(
			frames[face['frame']], face['landmark_5_68'], spec['template'], spec['size'])
		prepared = crop[:, :, ::-1] / 255.0
		prepared = (prepared - spec['mean']) / spec['std']
		targets.append(prepared.transpose(2, 0, 1).astype(numpy.float32))

		src_emb = (source_face['embedding_norm'].reshape(1, -1)
				   if args.swapper.startswith('hyperswap')
				   else R.prepare_source_embedding(models, _Wrap(source_face)))
		sources.append(R.balance_source_embedding(src_emb, face['embedding'])[0])
		geom.append(affine)

	swap_out = dev.run(args.ctx, {'target': targets, 'source': sources})

	mask = R.create_box_mask(spec['size'], R.FACE_MASK_BLUR, R.FACE_MASK_PADDING)
	result = [f.copy() for f in frames]
	for face, affine, case in zip(faces, geom, swap_out):
		out = list(case.values())[0].reshape(3, *spec['size'][::-1]).transpose(1, 2, 0)
		if spec['denorm']:
			out = out * spec['std'] + spec['mean']
		out = out.clip(0, 1)[:, :, ::-1] * 255
		result[face['frame']] = R.paste_back(result[face['frame']], out, mask, affine)

	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	h, w = frames[0].shape[:2]
	writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (w, h))
	for f in result:
		writer.write(f)
	writer.release()
	print('wrote', args.out)

	# ---- score against the host golden, frame by frame
	golden = args.golden or os.path.join(ROOT, 'output', 'golden-720p.mp4')
	if os.path.exists(golden):
		cap = cv2.VideoCapture(golden)
		psnrs = []
		for i in range(len(result)):
			ok, g = cap.read()
			if not ok:
				break
			d = (g.astype(numpy.float64) - result[i].astype(numpy.float64)) ** 2
			mse = d.mean()
			psnrs.append(10 * numpy.log10(255.0 ** 2 / mse) if mse > 0 else float('inf'))
		cap.release()
		if psnrs:
			print('\nEND-TO-END vs host golden: mean PSNR %.2f dB, worst %.2f dB, over %d frames'
				  % (numpy.mean(psnrs), min(psnrs), len(psnrs)))
			cv2.imwrite(os.path.join(ROOT, 'output', 'e2e_device_frame.png'), result[len(result)//2])


class _Wrap:
	"""adapt the dict-shaped face to what prepare_source_embedding expects"""
	def __init__(self, d):
		self.embedding = d['embedding']
		self.embedding_norm = d['embedding_norm']


if __name__ == '__main__':
	main()
