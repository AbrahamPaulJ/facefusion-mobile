"""Device-vs-host end-to-end comparison on RAW frames, with no codec in the path.

`run_e2e_device.py` scored its output against a golden MP4, and both sides had been through
cv2's lossy `mp4v` encoder.  That made the reported PSNR a floor set partly by the codec:
re-decoding the golden and diffing it against the untouched PNG target showed 66% of the
frame "changed" when only the faces were.

This runs the host reference and the device pipeline over the same frames in one process
and diffs the arrays directly, so the number reflects only quantisation.  It also reports
PSNR restricted to the swapped region, which is the quantity that actually governs how the
result looks -- a full-frame PSNR is dominated by untouched background and flatters itself
(Neodragon trap #18: pick the metric before running the comparison).

    py -3.10 work/pipeline/compare_e2e_raw.py --frames 12
"""
import argparse
import os
import sys

import cv2
import numpy

sys.path.insert(0, os.path.dirname(__file__))
import run_reference as R          # noqa: E402
import run_e2e_device as E         # noqa: E402
from device_runner import DeviceRunner   # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def psnr(a, b, mask=None):
	a = a.astype(numpy.float64)
	b = b.astype(numpy.float64)
	se = (a - b) ** 2
	mse = se[mask].mean() if mask is not None else se.mean()
	return float('inf') if mse == 0 else 10 * numpy.log10(255.0 ** 2 / mse)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--source', default=os.path.join(ROOT, 'assets', 'source.jpg'))
	ap.add_argument('--target', default=os.path.join(ROOT, 'assets', 'target-720p.mp4'))
	ap.add_argument('--frames', type=int, default=12)
	ap.add_argument('--swapper', default='hyperswap_1a_256', choices=list(R.SWAPPER_SPEC))
	ap.add_argument('--detector-ctx', default=None,
					help="device context for yoloface; 'yolofacef' is the fp32 build")
	ap.add_argument('--ctx', default='hyperswap')
	args = ap.parse_args()
	if args.detector_ctx:
		E.DETECTOR_CTX = args.detector_ctx

	models = R.Models(args.swapper)
	dev = DeviceRunner(verbose=False)

	cap = cv2.VideoCapture(args.target)
	frames = []
	while len(frames) < args.frames:
		ok, f = cap.read()
		if not ok:
			break
		frames.append(f)
	cap.release()

	# ---- host reference, in memory
	src = cv2.imread(args.source)
	sfaces = R.get_many_faces(models, src)
	source_face = max(sfaces, key=lambda f: (f.bounding_box[2] - f.bounding_box[0]) *
											(f.bounding_box[3] - f.bounding_box[1]))
	host = [R.process_frame(models, source_face, f.copy()) for f in frames]

	# ---- device pipeline, same frames
	sdet = E.detect_on_device(dev, [src])
	sfaces_d = E.faces_on_device(dev, models, [src], sdet)
	sfd = sfaces_d[0]
	dets = E.detect_on_device(dev, frames)
	faces = E.faces_on_device(dev, models, frames, dets)

	spec = R.SWAPPER_SPEC[args.swapper]
	targets, sources, geom = [], [], []
	for face in faces:
		crop, affine = R.warp_face_by_face_landmark_5(
			frames[face['frame']], face['landmark_5_68'], spec['template'], spec['size'])
		prep = (crop[:, :, ::-1] / 255.0 - spec['mean']) / spec['std']
		targets.append(prep.transpose(2, 0, 1).astype(numpy.float32))
		se = (sfd['embedding_norm'].reshape(1, -1) if args.swapper.startswith('hyperswap')
			  else R.prepare_source_embedding(models, E._Wrap(sfd)))
		sources.append(R.balance_source_embedding(se, face['embedding'])[0])
		geom.append(affine)

	swap_out = dev.run(args.ctx, {'target': targets, 'source': sources})
	mask = R.create_box_mask(spec['size'], R.FACE_MASK_BLUR, R.FACE_MASK_PADDING)
	device = [f.copy() for f in frames]
	for face, affine, case in zip(faces, geom, swap_out):
		o = list(case.values())[0].reshape(3, spec['size'][1], spec['size'][0]).transpose(1, 2, 0)
		if spec['denorm']:
			o = o * spec['std'] + spec['mean']
		o = o.clip(0, 1)[:, :, ::-1] * 255
		device[face['frame']] = R.paste_back(device[face['frame']], o, mask, affine)

	print('%d frames, %d faces, swapper=%s' % (len(frames), len(faces), args.swapper))
	full, region, areas = [], [], []
	for h, d, orig in zip(host, device, frames):
		touched = numpy.abs(h.astype(int) - orig.astype(int)).sum(2) > 0
		full.append(psnr(h, d))
		areas.append(100.0 * touched.sum() / touched.size)
		if touched.any():
			region.append(psnr(h, d, mask=touched))
	print('\n%-34s %8s %8s' % ('', 'mean', 'worst'))
	print('%-34s %8.2f %8.2f' % ('PSNR, full frame', numpy.mean(full), min(full)))
	if region:
		print('%-34s %8.2f %8.2f' % ('PSNR, swapped region only', numpy.mean(region), min(region)))
	print('%-34s %8.2f %%' % ('swapped area', numpy.mean(areas)))
	mx = max(int(numpy.abs(h.astype(int) - d.astype(int)).max()) for h, d in zip(host, device))
	mn = numpy.mean([numpy.abs(h.astype(int) - d.astype(int)).mean() for h, d in zip(host, device)])
	print('%-34s %8d' % ('max abs pixel diff', mx))
	print('%-34s %8.4f' % ('mean abs pixel diff', mn))


if __name__ == '__main__':
	main()
