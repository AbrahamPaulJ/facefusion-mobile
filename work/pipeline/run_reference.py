"""Golden reference for the FaceFusion default swap path.

A self-contained replication of FaceFusion 3.8.2's default face-swap chain using
onnxruntime + cv2 only.  Deliberately NOT facefusion.py: that pins ort 1.28 / numpy 2.4.6 /
opencv 5.0 and drags in gradio, none of which this box has or needs.

Every function here mirrors a named function in the upstream source and cites it, so a
divergence can be bisected against the real thing.

Roles:
  * golden reference for every device-vs-host SNR comparison
  * source of every calibration tensor  (trap #4: calibration must be REAL pipeline
    tensors, never synthetic)

Usage:
    py -3.10 work/pipeline/run_reference.py --source work/assets/source.jpg \
        --target work/assets/target-720p.mp4 --out work/output/swapped.mp4
    py -3.10 work/pipeline/run_reference.py --source ... --target frame.jpg --out out.jpg
    ... --dump-calib work/calib --calib-frames 64
"""
import argparse
import os
import sys

import cv2
import numpy
import onnxruntime

MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')

# facefusion/face_helper.py:10  WARP_TEMPLATE_SET
WARP_TEMPLATE_SET = {
	'arcface_112_v2': numpy.array([
		[0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.50022500, 0.64050536],
		[0.37097589, 0.82469196], [0.63151696, 0.82325089]]),
	'arcface_128': numpy.array([
		[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219],
		[0.38710391, 0.72160547], [0.61507734, 0.72034453]]),
	'ffhq_512': numpy.array([
		[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
		[0.39308822, 0.72541100], [0.61150205, 0.72490465]]),
}

# facefusion/program.py defaults, for the default swap path
FACE_DETECTOR_SIZE = (640, 640)
FACE_DETECTOR_SCORE = 0.5
FACE_LANDMARKER_SCORE = 0.5
FACE_MASK_BLUR = 0.3
FACE_MASK_PADDING = (0, 0, 0, 0)
FACE_SWAPPER_WEIGHT = 0.5
CALIB_ONLY = False


def session(name):
	path = os.path.join(MODELS, name + '.onnx')
	opts = onnxruntime.SessionOptions()
	opts.log_severity_level = 3
	return onnxruntime.InferenceSession(path, opts, providers=['CPUExecutionProvider'])


class Models:
	def __init__(self, swapper='hyperswap_1a_256'):
		self.yoloface = session('yoloface_8n')
		self.fan_68_5 = session('fan_68_5')
		self.fan2d = session('2dfan4')
		self.arcface = session('arcface_w600k_r50')
		self.swapper = session(swapper)
		self.swapper_name = swapper


# ---------------------------------------------------------------- geometry helpers

def estimate_matrix_by_face_landmark_5(landmark_5, template, crop_size):
	"""facefusion/face_helper.py:71"""
	normed = WARP_TEMPLATE_SET[template] * crop_size
	return cv2.estimateAffinePartial2D(
		landmark_5.astype(numpy.float32), normed.astype(numpy.float32),
		method=cv2.RANSAC, ransacReprojThreshold=100)[0]


def warp_face_by_face_landmark_5(frame, landmark_5, template, crop_size):
	"""facefusion/face_helper.py:77"""
	matrix = estimate_matrix_by_face_landmark_5(landmark_5, template, crop_size)
	crop = cv2.warpAffine(frame, matrix, crop_size,
						  borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
	return crop, matrix


def warp_face_by_translation(frame, translation, scale, crop_size):
	"""facefusion/face_helper.py:95"""
	matrix = numpy.array([[scale, 0.0, translation[0]], [0.0, scale, translation[1]]])
	return cv2.warpAffine(frame, matrix, crop_size), matrix


def create_rotation_matrix_and_size(angle, size):
	"""facefusion/face_helper.py:142"""
	matrix = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1)
	rot_size = numpy.dot(numpy.abs(matrix[:, :2]), size)
	matrix[:, -1] += (rot_size - size) * 0.5
	return matrix, (int(rot_size[0]), int(rot_size[1]))


def transform_points(points, matrix):
	"""facefusion/face_helper.py — cv2.transform on an Nx2"""
	return cv2.transform(points.reshape(1, -1, 2).astype(numpy.float32), matrix).reshape(-1, 2)


def estimate_face_angle(landmark_68):
	"""facefusion/face_helper.py:220 — snapped to {0, 90, 180, 270}"""
	x1, y1 = landmark_68[0]
	x2, y2 = landmark_68[16]
	theta = numpy.degrees(numpy.arctan2(y2 - y1, x2 - x1)) % 360
	angles = numpy.linspace(0, 360, 5)
	return int(angles[numpy.argmin(numpy.abs(angles - theta))] % 360)


def convert_to_face_landmark_5(landmark_68):
	"""facefusion/face_helper.py — the 68 -> 5 reduction"""
	return numpy.array([
		numpy.mean(landmark_68[36:42], axis=0),
		numpy.mean(landmark_68[42:48], axis=0),
		landmark_68[30], landmark_68[48], landmark_68[54]])


# ---------------------------------------------------------------- detection

def restrict_frame(frame, resolution):
	"""facefusion/vision.py:222"""
	h, w = frame.shape[:2]
	rw, rh = resolution
	if h > rh or w > rw:
		scale = min(rh / h, rw / w)
		return cv2.resize(frame, (int(w * scale), int(h * scale)))
	return frame


def detect_faces(models, frame):
	"""facefusion/face_detector.py:298 detect_with_yolo_face"""
	temp = restrict_frame(frame, FACE_DETECTOR_SIZE)
	ratio_h = frame.shape[0] / temp.shape[0]
	ratio_w = frame.shape[1] / temp.shape[1]

	# prepare_detect_frame (face_detector.py:445) + normalize to [0,1]
	det = numpy.zeros((FACE_DETECTOR_SIZE[1], FACE_DETECTOR_SIZE[0], 3))
	det[:temp.shape[0], :temp.shape[1], :] = temp
	det = numpy.expand_dims(det.transpose(2, 0, 1), axis=0).astype(numpy.float32) / 255.0

	out = models.yoloface.run(None, {'input': det})[0]
	detection = numpy.squeeze(out).T
	boxes_raw, scores_raw, lmk_raw = numpy.split(detection, [4, 5], axis=1)
	keep = numpy.where(scores_raw > FACE_DETECTOR_SCORE)[0]

	boxes, scores, landmarks = [], [], []
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
			landmarks.append(numpy.array(lm.reshape(-1, 3)[:, :2]))
	return boxes, scores, landmarks


def apply_nms(boxes, scores, score_threshold, nms_threshold):
	"""facefusion/face_helper.py:231"""
	norm = [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in boxes]
	return cv2.dnn.NMSBoxes(norm, scores, score_threshold=score_threshold,
							nms_threshold=nms_threshold)


# ---------------------------------------------------------------- landmarks

def conditional_optimize_contrast(crop):
	"""facefusion/face_landmarker.py:191"""
	crop = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab)
	if numpy.mean(crop[:, :, 0]) < 30:
		crop[:, :, 0] = cv2.createCLAHE(clipLimit=2).apply(crop[:, :, 0])
	return cv2.cvtColor(crop, cv2.COLOR_Lab2RGB)


def decode_heatmaps(heatmaps):
	"""The twelve nodes 2dfan4 runs between `heatmaps` and `landmarks`, on the host.

	See docs/model-audit.md §1.  This is what lets the NPU graph be cut at `heatmaps`,
	removing ArgMax/Mod/Greater/Not/Expand/Tile/ReduceSum/Sqrt/Clip/Div from it.

	    ArgMax(heatmaps) -> peak;  mask = ||index - peak|| <= threshold
	    windowed = clip(heatmaps * mask, 0, inf)
	    m00 = clip(sum(windowed), eps, inf)
	    xs  = sum(windowed * x_index) / m00      (ys likewise)

	Returns landmark_68 in 64x64 heatmap coordinates, plus the per-point peak value.
	"""
	hm = heatmaps[0]                                   # [68, 64, 64]
	n, h, w = hm.shape
	flat = hm.reshape(n, -1)
	peak_idx = numpy.argmax(flat, axis=1)
	peak_y, peak_x = numpy.divmod(peak_idx, w)         # the graph's ArgMax + Mod

	# The window is a radius-6.4 disc around each peak, so only a 13x13 neighbourhood can
	# contribute.  Scanning the full 68x64x64 is ~20x more work than the answer needs, and
	# the device port would inherit that.  Build the mask on the small offset grid instead;
	# the result is identical because everything outside the disc is multiplied by zero.
	r = int(numpy.ceil(DECODE_WINDOW))
	off = numpy.arange(-r, r + 1)
	dy, dx = numpy.meshgrid(off, off, indexing='ij')
	disc = (dx * dx + dy * dy) <= DECODE_WINDOW * DECODE_WINDOW

	windowed = numpy.zeros_like(hm)
	for k in range(n):
		y0, x0 = peak_y[k] - r, peak_x[k] - r
		ys = slice(max(y0, 0), min(y0 + 2 * r + 1, h))
		xs = slice(max(x0, 0), min(x0 + 2 * r + 1, w))
		sub = disc[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
		windowed[k, ys, xs] = hm[k, ys, xs] * sub
	windowed = numpy.clip(windowed, DECODE_CLIP_MIN, None)

	# The graph's `x_indices`/`y_indices` initializers are PIXEL CENTRES (0.5 .. 63.5),
	# not 0-based indices.  Using i instead of i+0.5 costs exactly 0.5 heatmap px.
	ys_i, xs_i = numpy.mgrid[0:h, 0:w]
	m00 = numpy.clip(windowed.sum(axis=(1, 2)), DECODE_M00_EPS, None)
	xs = (windowed * (xs_i[None] + 0.5)).sum(axis=(1, 2)) / m00
	ys = (windowed * (ys_i[None] + 0.5)).sum(axis=(1, 2)) / m00
	return numpy.stack([xs, ys], axis=1), flat.max(axis=1)


# Read out of the graph's own initializers, not fitted:
#   Greater_867 compares the peak distance against  onnx::Greater_1889 = 6.4
#   Clip_899 lower-bounds the windowed heatmap at   onnx::Range_1986   = 0.0
#   Clip_901 lower-bounds m00 at                    onnx::Clip_1999    = 1.1920929e-07
DECODE_WINDOW = 6.4
DECODE_CLIP_MIN = 0.0
DECODE_M00_EPS = 1.1920929e-07


def verify_decode(models, crop_chw):
	"""Assert the host decode reproduces the graph's own `landmarks` output.

	Phase 2 cuts the NPU graph at `heatmaps` and runs decode_heatmaps() instead, so this
	equivalence is the thing that makes the cut safe.  Checked, never assumed.
	Returns max abs error in 64x64 heatmap units.
	"""
	lm_onnx, hm = models.fan2d.run(None, {'input': [crop_chw]})
	lm_host, _ = decode_heatmaps(hm)
	return numpy.abs(lm_host - lm_onnx[0, :, :2]).max()


def detect_landmark_68(models, frame, bounding_box, face_angle):
	"""facefusion/face_landmarker.py:153 detect_with_2dfan4"""
	model_size = (256, 256)
	scale = 195 / numpy.subtract(bounding_box[2:], bounding_box[:2]).max().clip(1, None)
	translation = (model_size[0] - numpy.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
	rot_matrix, rot_size = create_rotation_matrix_and_size(face_angle, model_size)
	crop, affine = warp_face_by_translation(frame, translation, scale, model_size)
	crop = cv2.warpAffine(crop, rot_matrix, rot_size)
	crop = conditional_optimize_contrast(crop)
	crop_chw = crop.transpose(2, 0, 1).astype(numpy.float32) / 255.0

	_, heatmaps = models.fan2d.run(None, {'input': [crop_chw]})
	landmark_64, peaks = decode_heatmaps(heatmaps)

	landmark_68 = landmark_64 / 64 * 256
	landmark_68 = transform_points(landmark_68, cv2.invertAffineTransform(rot_matrix))
	landmark_68 = transform_points(landmark_68, cv2.invertAffineTransform(affine))
	score = numpy.interp(numpy.mean(peaks), [0, 0.9], [0, 1])
	return landmark_68, score, crop_chw


# ---------------------------------------------------------------- recognition

def calculate_face_embedding(models, frame, landmark_5):
	"""facefusion/face_recognizer.py:71"""
	crop, _ = warp_face_by_face_landmark_5(frame, landmark_5, 'arcface_112_v2', (112, 112))
	crop = crop / 127.5 - 1.0
	crop = crop[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)
	crop = numpy.expand_dims(crop, axis=0)
	embedding = models.arcface.run(None, {'input': crop})[0].ravel()
	return embedding, embedding / numpy.linalg.norm(embedding), crop


# ---------------------------------------------------------------- face assembly

class Face:
	__slots__ = ('bounding_box', 'landmark_5', 'landmark_5_68', 'landmark_68',
				 'angle', 'embedding', 'embedding_norm', 'score')


def create_faces(models, frame, boxes, scores, landmarks_5, calib=None):
	"""facefusion/face_creator.py:16"""
	faces = []
	keep = apply_nms(boxes, scores, FACE_DETECTOR_SCORE, 0.4)
	for index in numpy.array(keep).ravel().tolist():
		box, landmark_5 = boxes[index], landmarks_5[index]

		landmark_68_5 = models.fan_68_5.run(None, {'input': [landmark_5.astype(numpy.float32)]})[0][0]
		angle = estimate_face_angle(landmark_68_5)

		landmark_68, score_68, fan_crop = detect_landmark_68(models, frame, box, angle)
		landmark_5_68 = convert_to_face_landmark_5(landmark_68) if score_68 > FACE_LANDMARKER_SCORE else landmark_5

		embedding, embedding_norm, arc_crop = calculate_face_embedding(models, frame, landmark_5_68)

		if calib is not None:
			calib.setdefault('fan2d', []).append(fan_crop)
			calib.setdefault('arcface', []).append(arc_crop[0])

		face = Face()
		face.bounding_box, face.landmark_5 = box, landmark_5
		face.landmark_5_68, face.landmark_68, face.angle = landmark_5_68, landmark_68, angle
		face.embedding, face.embedding_norm, face.score = embedding, embedding_norm, scores[index]
		faces.append(face)
	return faces


def get_many_faces(models, frame, calib=None):
	boxes, scores, landmarks_5 = detect_faces(models, frame)
	if not boxes:
		return []
	return create_faces(models, frame, boxes, scores, landmarks_5, calib)


# ---------------------------------------------------------------- swapping

def create_box_mask(crop_size, blur, padding):
	"""facefusion/face_masker.py:188"""
	blur_amount = int(crop_size[0] * 0.5 * blur)
	blur_area = max(blur_amount // 2, 1)
	mask = numpy.ones(crop_size, dtype=numpy.float32)
	mask[:max(blur_area, int(crop_size[1] * padding[0] / 100)), :] = 0
	mask[-max(blur_area, int(crop_size[1] * padding[2] / 100)):, :] = 0
	mask[:, :max(blur_area, int(crop_size[0] * padding[3] / 100))] = 0
	mask[:, -max(blur_area, int(crop_size[0] * padding[1] / 100)):] = 0
	if blur_amount > 0:
		mask = cv2.GaussianBlur(mask, (0, 0), blur_amount * 0.25)
	return mask


def paste_back(frame, crop, crop_mask, affine_matrix):
	"""facefusion/face_helper.py:101"""
	inverse = cv2.invertAffineTransform(affine_matrix)
	ch, cw = crop.shape[:2]
	th, tw = frame.shape[:2]
	pts = transform_points(numpy.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], numpy.float32), inverse)
	x1, y1 = numpy.clip(numpy.floor(pts.min(axis=0)).astype(int), 0, [tw, th])
	x2, y2 = numpy.clip(numpy.ceil(pts.max(axis=0)).astype(int), 0, [tw, th])
	if x2 <= x1 or y2 <= y1:
		return frame
	paste_matrix = inverse.copy()
	paste_matrix[:, -1] -= [x1, y1]
	w, h = x2 - x1, y2 - y1
	inv_mask = cv2.warpAffine(crop_mask, paste_matrix, (w, h)).clip(0, 1)[:, :, None]
	inv_crop = cv2.warpAffine(crop, paste_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
	frame = frame.copy()
	region = frame[y1:y2, x1:x2]
	frame[y1:y2, x1:x2] = (region * (1 - inv_mask) + inv_crop * inv_mask).astype(frame.dtype)
	return frame


SWAPPER_SPEC = {
	'hyperswap_1a_256': dict(template='arcface_128', size=(256, 256), mean=0.5, std=0.5, denorm=True),
	'inswapper_128':    dict(template='arcface_128', size=(128, 128), mean=0.0, std=1.0, denorm=False),
}


def prepare_source_embedding(models, source_face):
	"""facefusion/processors/modules/face_swapper/core.py:689"""
	if models.swapper_name.startswith('hyperswap'):
		return source_face.embedding_norm.reshape((1, -1)).astype(numpy.float32)
	# inswapper: emap matmul against the model's own initializer
	import onnx
	from onnx import numpy_helper
	if not hasattr(models, '_emap'):
		proto = onnx.load(os.path.join(MODELS, 'inswapper_128.onnx'))
		models._emap = numpy_helper.to_array(proto.graph.initializer[-1])
	e = source_face.embedding.reshape((1, -1))
	return (numpy.dot(e, models._emap) / numpy.linalg.norm(e)).astype(numpy.float32)


def balance_source_embedding(source_embedding, target_embedding):
	"""facefusion/processors/modules/face_swapper/core.py:715"""
	weight = numpy.interp(FACE_SWAPPER_WEIGHT, [0, 1], [0.35, -0.35]).astype(numpy.float32)
	target = target_embedding / numpy.linalg.norm(target_embedding)
	return (source_embedding.reshape(1, -1) * (1 - weight) +
			target.reshape(1, -1) * weight).astype(numpy.float32)


def swap_face(models, source_face, target_face, frame, calib=None):
	"""facefusion/processors/modules/face_swapper/core.py:610 swap_face"""
	spec = SWAPPER_SPEC[models.swapper_name]
	crop, affine = warp_face_by_face_landmark_5(
		frame, target_face.landmark_5_68, spec['template'], spec['size'])
	mask = create_box_mask(spec['size'], FACE_MASK_BLUR, FACE_MASK_PADDING)

	prepared = crop[:, :, ::-1] / 255.0
	prepared = (prepared - spec['mean']) / spec['std']
	prepared = numpy.expand_dims(prepared.transpose(2, 0, 1), axis=0).astype(numpy.float32)

	embedding = balance_source_embedding(
		prepare_source_embedding(models, source_face), target_face.embedding)

	if calib is not None:
		suffix = '_128' if models.swapper_name.startswith('inswapper') else ''
		calib.setdefault('swap_target' + suffix, []).append(prepared[0])
		calib.setdefault('swap_source' + suffix, []).append(embedding[0])

	if CALIB_ONLY:
		# Capturing the swapper's INPUTS does not require running the swapper.  inswapper
		# is 174.58 GMAC per face on CPU; skipping it makes its calibration capture as
		# cheap as hyperswap's.
		#
		# ⚠ It also skips the ENHANCER capture below, which is downstream of the swapper's
		# output and therefore cannot be had for free.  --calib-only produces no `gpen` set.
		return frame

	out = models.swapper.run(None, {'source': embedding, 'target': prepared})[0][0]

	out = out.transpose(1, 2, 0)
	if spec['denorm']:
		out = out * spec['std'] + spec['mean']
	out = out.clip(0, 1)[:, :, ::-1] * 255

	# The face enhancer's calibration set.
	#
	# gpen_bfr_256 only ever sees a face the SWAPPER has already written, so its input
	# distribution is this crop -- not the target crop above.  Calibrating it on target
	# crops would quantise it against a distribution it never meets at runtime; trap #4 is
	# about using real tensors, and this is that rule one stage further down.
	#
	# Captured in the enhancer's own normalisation (face_enhancer/core.py:prepare_crop_frame
	# -- BGR->RGB, /255, then (x-0.5)/0.5), which happens to match the swapper's mean/std
	# but is written out explicitly rather than shared: the two are independent upstream and
	# a future enhancer with different constants must not silently inherit these.
	#
	# Only at 256: gpen is a fixed 256x256 graph, and an inswapper run produces a 128 crop
	# that is not a valid enhancer input.
	if calib is not None and out.shape[0] == 256:
		g = out[:, :, ::-1] / 255.0
		g = (g - 0.5) / 0.5
		calib.setdefault('gpen', []).append(
			g.transpose(2, 0, 1).astype(numpy.float32))

	return paste_back(frame, out, mask, affine)


def process_frame(models, source_face, frame, calib=None):
	for face in get_many_faces(models, frame, calib):
		frame = swap_face(models, source_face, face, frame, calib)
	return frame


# ---------------------------------------------------------------- entry point

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--source', required=True)
	ap.add_argument('--target', required=True)
	ap.add_argument('--out', required=True)
	ap.add_argument('--swapper', default='hyperswap_1a_256', choices=list(SWAPPER_SPEC))
	ap.add_argument('--max-frames', type=int, default=0)
	ap.add_argument('--dump-calib', default=None,
					help='directory to write real pipeline tensors into (trap #4)')
	ap.add_argument('--calib-frames', type=int, default=64)
	ap.add_argument('--calib-stride', type=int, default=1)
	ap.add_argument('--calib-only', action='store_true',
					help='capture swapper inputs without running the swapper')
	args = ap.parse_args()

	global CALIB_ONLY
	CALIB_ONLY = args.calib_only
	models = Models(args.swapper)

	source_frame = cv2.imread(args.source)
	if source_frame is None:
		sys.exit('cannot read source: ' + args.source)

	source_faces = get_many_faces(models, source_frame)
	if not source_faces:
		sys.exit('no face found in source')
	source_face = max(source_faces, key=lambda f: (f.bounding_box[2] - f.bounding_box[0]) *
												  (f.bounding_box[3] - f.bounding_box[1]))
	print('source: %d face(s), embedding norm %.4f' % (len(source_faces),
													   numpy.linalg.norm(source_face.embedding)))

	# The Phase 2 graph cut depends on this equivalence; check it on a real crop.
	_, _, probe_crop = detect_landmark_68(models, source_frame, source_face.bounding_box,
										  source_face.angle)
	err = verify_decode(models, probe_crop)
	print('host heatmap decode vs graph `landmarks`: max err %.3e heatmap px' % err)
	if err > 1e-3:
		sys.exit('host decode diverges from the graph — do NOT cut 2dfan4 at `heatmaps` yet')

	calib = {} if args.dump_calib else None

	if args.target.lower().endswith(('.jpg', '.jpeg', '.png')):
		frame = cv2.imread(args.target)
		out = process_frame(models, source_face, frame, calib)
		os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
		cv2.imwrite(args.out, out)
		print('wrote', args.out)
	else:
		cap = cv2.VideoCapture(args.target)
		fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
		total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		limit = args.max_frames or total
		os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
		writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
		print('target: %dx%d @ %.2f fps, %d frames (processing %d)' % (w, h, fps, total, limit))
		i = 0
		while i < limit:
			ok, frame = cap.read()
			if not ok:
				break
			# Sample calibration across the WHOLE clip, not the first N frames — a
			# contiguous head gives real tensors with artificially narrow range coverage.
			take = (calib is not None and i % args.calib_stride == 0 and
					len(calib.get('arcface', [])) < args.calib_frames)
			c = calib if take else None
			writer.write(process_frame(models, source_face, frame, c))
			i += 1
			if i % 25 == 0:
				print('  %d/%d' % (i, limit), flush=True)
		cap.release()
		writer.release()
		print('wrote %s (%d frames)' % (args.out, i))

	if calib:
		os.makedirs(args.dump_calib, exist_ok=True)
		for name, tensors in calib.items():
			d = os.path.join(args.dump_calib, name)
			os.makedirs(d, exist_ok=True)
			for k, t in enumerate(tensors):
				numpy.ascontiguousarray(t, dtype=numpy.float32).tofile(
					os.path.join(d, '%s_%04d.raw' % (name, k)))
			print('calib %-12s %4d tensors  shape %s' % (name, len(tensors), tensors[0].shape))


if __name__ == '__main__':
	main()
