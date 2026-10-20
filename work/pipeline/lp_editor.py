"""FaceFusion's face_editor keypoint arithmetic, without its state manager: the oracle for
the app's port, and what the generator's calibration and held-out edits are made with.

Every constant is face_editor/core.py's and processors/live_portrait.py's (FaceFusion v4,
the same arithmetic as 3.8.2's). `apply_edit` returns the generator's (source, target) in
the ONNX's naming: `source` is the EDITED keypoints and `target` the face as it is --
face_editor's forward_generate_frame(feature_volume, motion_points_source,
motion_points_target).

The eye and lip openers read 68-point landmark ratios. A caller without landmarks passes
typical ratios (an open eye ~0.3, a closed mouth ~0.05), which is enough for calibration;
the app passes the real ones.
"""
import os

import numpy
import onnxruntime
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS = os.path.join(ROOT, 'models')

SLIDERS = ('eyebrow_direction', 'eye_gaze_horizontal', 'eye_gaze_vertical', 'eye_open_ratio',
           'lip_open_ratio', 'mouth_grim', 'mouth_pout', 'mouth_purse', 'mouth_smile',
           'mouth_position_horizontal', 'mouth_position_vertical', 'head_pitch', 'head_yaw',
           'head_roll')

EXPRESSION_MIN = numpy.array([[
	[-2.88067125e-02, -8.12731311e-02, -1.70541159e-03], [-4.88598682e-02, -3.32196616e-02, -1.67431499e-04],
	[-6.75425082e-02, -4.28681746e-02, -1.98950816e-04], [-7.23103955e-02, -3.28503326e-02, -7.31324719e-04],
	[-3.87073644e-02, -6.01546466e-02, -5.50269964e-04], [-6.38048723e-02, -2.23840728e-01, -7.13261834e-04],
	[-3.02710701e-02, -3.93195450e-02, -8.24086510e-06], [-2.95799859e-02, -5.39318882e-02, -1.74219604e-04],
	[-2.92359516e-02, -1.53050944e-02, -6.30460854e-05], [-5.56493877e-03, -2.34344602e-02, -1.26858242e-04],
	[-4.37593013e-02, -2.77768299e-02, -2.70503685e-02], [-1.76926646e-02, -1.91676542e-02, -1.15090821e-04],
	[-8.34268332e-03, -3.99775570e-03, -3.27481248e-05], [-3.40162888e-02, -2.81868968e-02, -1.96679524e-04],
	[-2.91855410e-02, -3.97511162e-02, -2.81230678e-05], [-1.50395725e-02, -2.49494594e-02, -9.42573533e-05],
	[-1.67938769e-02, -2.00953931e-02, -4.00750607e-04], [-1.86435618e-02, -2.48535164e-02, -2.74416432e-02],
	[-4.61211195e-03, -1.21660791e-02, -2.93173041e-04], [-4.10017073e-02, -7.43824020e-02, -4.42762971e-02],
	[-1.90370996e-02, -3.74363363e-02, -1.34740388e-02]]], numpy.float32)
EXPRESSION_MAX = numpy.array([[
	[4.46682945e-02, 7.08772913e-02, 4.08344204e-04], [2.14308221e-02, 6.15894832e-02, 4.85319615e-05],
	[3.02363783e-02, 4.45043296e-02, 1.28298725e-05], [3.05869691e-02, 3.79812494e-02, 6.57040102e-04],
	[4.45670523e-02, 3.97259220e-02, 7.10966764e-04], [9.43699256e-02, 9.85926315e-02, 2.02551950e-04],
	[1.61131397e-02, 2.92906128e-02, 3.44733417e-06], [5.23825921e-02, 1.07065082e-01, 6.61510974e-04],
	[2.85718683e-03, 8.32320191e-03, 2.39314613e-04], [2.57947259e-02, 1.60935968e-02, 2.41853559e-05],
	[4.90833223e-02, 3.43903080e-02, 3.22353356e-02], [1.44766076e-02, 3.39248963e-02, 1.42291479e-04],
	[8.75749043e-04, 6.82212645e-03, 2.76097053e-05], [1.86958015e-02, 3.84016186e-02, 7.33085908e-05],
	[2.01714113e-02, 4.90544215e-02, 2.34028921e-05], [2.46518422e-02, 3.29151377e-02, 3.48571630e-05],
	[2.22457591e-02, 1.21796541e-02, 1.56396593e-04], [1.72109623e-02, 3.01626958e-02, 1.36556877e-02],
	[1.83460284e-02, 1.61141958e-02, 2.87440169e-04], [3.57594155e-02, 1.80554688e-01, 2.75554154e-02],
	[2.17450950e-02, 8.66811201e-02, 3.34241726e-02]]], numpy.float32)

_SESSIONS = {}


def session(name):
	if name not in _SESSIONS:
		so = onnxruntime.SessionOptions()
		so.log_severity_level = 3
		_SESSIONS[name] = onnxruntime.InferenceSession(
			os.path.join(MODELS, 'live_portrait_%s.onnx' % name), so, providers=['CPUExecutionProvider'])
	return _SESSIONS[name]


def interp(v, lo, hi):
	return float(numpy.interp(v, [-1, 1], [lo, hi]))


def create_rotation(pitch, yaw, roll):
	return Rotation.from_euler('xyz', [pitch, yaw, roll], degrees=True).as_matrix().astype(numpy.float32)


def limit_angle(pitch, yaw, roll, out_pitch, out_yaw, out_roll):
	pmin, pmax, ymin, ymax, rmin, rmax = -30.0, 30.0, -60.0, 60.0, -20.0, 20.0
	if pitch < 0: pmin = min(pitch, pmin)
	else: pmax = max(pitch, pmax)
	if yaw < 0: ymin = min(yaw, ymin)
	else: ymax = max(yaw, ymax)
	if roll < 0: rmin = min(roll, rmin)
	else: rmax = max(roll, rmax)
	return numpy.clip(out_pitch, pmin, pmax), numpy.clip(out_yaw, ymin, ymax), numpy.clip(out_roll, rmin, rmax)


def edit_expression(e, s):
	"""face_editor's expression edits, in its own order: gaze, grim, position, pout, purse,
	smile, eyebrow. `e` is modified in place and returned."""
	g = s['eye_gaze_horizontal']
	if g > 0:
		e[0, 11, 0] += interp(g, -0.015, 0.015); e[0, 15, 0] += interp(g, -0.020, 0.020)
	else:
		e[0, 11, 0] += interp(g, -0.020, 0.020); e[0, 15, 0] += interp(g, -0.015, 0.015)
	v = s['eye_gaze_vertical']
	e[0, 1, 1] += interp(v, -0.0025, 0.0025); e[0, 2, 1] -= interp(v, -0.0025, 0.0025)
	e[0, 11, 1] -= interp(v, -0.010, 0.010); e[0, 13, 1] -= interp(v, -0.005, 0.005)
	e[0, 15, 1] -= interp(v, -0.010, 0.010); e[0, 16, 1] -= interp(v, -0.005, 0.005)
	m = s['mouth_grim']
	if m > 0:
		e[0, 17, 2] -= interp(m, -0.005, 0.005); e[0, 19, 2] += interp(m, -0.01, 0.01)
		e[0, 20, 1] -= interp(m, -0.06, 0.06); e[0, 20, 2] -= interp(m, -0.03, 0.03)
	else:
		e[0, 19, 1] -= interp(m, -0.05, 0.05); e[0, 19, 2] -= interp(m, -0.02, 0.02)
		e[0, 20, 2] -= interp(m, -0.03, 0.03)
	h, v = s['mouth_position_horizontal'], s['mouth_position_vertical']
	e[0, 19, 0] += interp(h, -0.05, 0.05); e[0, 20, 0] += interp(h, -0.04, 0.04)
	if v > 0:
		e[0, 19, 1] -= interp(v, -0.04, 0.04); e[0, 20, 1] -= interp(v, -0.02, 0.02)
	else:
		e[0, 19, 1] -= interp(v, -0.05, 0.05); e[0, 20, 1] -= interp(v, -0.04, 0.04)
	p = s['mouth_pout']
	if p > 0:
		e[0, 19, 1] -= interp(p, -0.022, 0.022); e[0, 19, 2] += interp(p, -0.025, 0.025)
		e[0, 20, 2] -= interp(p, -0.002, 0.002)
	else:
		e[0, 19, 1] += interp(p, -0.022, 0.022); e[0, 19, 2] += interp(p, -0.025, 0.025)
		e[0, 20, 2] -= interp(p, -0.002, 0.002)
	p = s['mouth_purse']
	if p > 0:
		e[0, 19, 1] -= interp(p, -0.04, 0.04); e[0, 19, 2] -= interp(p, -0.02, 0.02)
	else:
		e[0, 14, 1] -= interp(p, -0.02, 0.02); e[0, 17, 2] += interp(p, -0.01, 0.01)
		e[0, 19, 2] -= interp(p, -0.015, 0.015); e[0, 20, 2] -= interp(p, -0.002, 0.002)
	m = s['mouth_smile']
	if m > 0:
		e[0, 20, 1] -= interp(m, -0.015, 0.015); e[0, 14, 1] -= interp(m, -0.025, 0.025)
		e[0, 17, 1] += interp(m, -0.01, 0.01); e[0, 17, 2] += interp(m, -0.004, 0.004)
		e[0, 3, 1] -= interp(m, -0.0045, 0.0045); e[0, 7, 1] -= interp(m, -0.0045, 0.0045)
	else:
		e[0, 14, 1] -= interp(m, -0.02, 0.02); e[0, 17, 1] += interp(m, -0.003, 0.003)
		e[0, 19, 1] += interp(m, -0.02, 0.02); e[0, 19, 2] -= interp(m, -0.005, 0.005)
		e[0, 20, 2] += interp(m, -0.01, 0.01); e[0, 3, 1] += interp(m, -0.0045, 0.0045)
		e[0, 7, 1] += interp(m, -0.0045, 0.0045)
	b = s['eyebrow_direction']
	if b > 0:
		e[0, 1, 1] += interp(b, -0.015, 0.015); e[0, 2, 1] -= interp(b, -0.020, 0.020)
	else:
		e[0, 1, 0] -= interp(b, -0.015, 0.015); e[0, 2, 0] += interp(b, -0.020, 0.020)
		e[0, 1, 1] += interp(b, -0.005, 0.005); e[0, 2, 1] -= interp(b, -0.005, 0.005)
	return e


def apply_edit(motion, sliders, eye_ratios=(0.3, 0.3), lip_ratio=0.05):
	"""face_editor.apply_edit's keypoints: returns (source, target), float32 [1,21,3].
	`motion` is the motion extractor's seven outputs; `sliders` maps SLIDERS to [-1, 1]."""
	s = {k: float(sliders.get(k, 0.0)) for k in SLIDERS}
	pitch, yaw, roll, scale, translation, expression, mp = [numpy.array(x, numpy.float32) for x in motion]
	p, y, r = float(numpy.ravel(pitch)[0]), float(numpy.ravel(yaw)[0]), float(numpy.ravel(roll)[0])
	rotation = create_rotation(p, y, r)
	target = scale * (mp @ rotation.T + expression) + translation
	expression = edit_expression(expression.copy(), s)
	expression = numpy.clip(expression, EXPRESSION_MIN, EXPRESSION_MAX)
	ep = p + interp(s['head_pitch'], 20, -20)
	ey = y + interp(s['head_yaw'], 60, -60)
	er = r + interp(s['head_roll'], -15, 15)
	ep, ey, er = limit_angle(p, y, r, ep, ey, er)
	rotation = create_rotation(float(ep), float(ey), float(er))
	source = mp @ rotation.T
	source += expression
	source *= scale
	source += translation
	eye = numpy.concatenate([target.ravel(), [eye_ratios[0], eye_ratios[1], 0.0 if s['eye_open_ratio'] < 0 else 0.6]])
	eye = session('eye_retargeter').run(None, {'input': eye.reshape(1, -1).astype(numpy.float32)})[0]
	source += (eye * abs(s['eye_open_ratio'])).reshape(-1, 21, 3)
	lip = numpy.concatenate([target.ravel(), [lip_ratio, 0.0 if s['lip_open_ratio'] < 0 else 1.0]])
	lip = session('lip_retargeter').run(None, {'input': lip.reshape(1, -1).astype(numpy.float32)})[0]
	source += (lip * abs(s['lip_open_ratio'])).reshape(-1, 21, 3)
	source = session('stitcher').run(None, {'source': source.astype(numpy.float32),
	                                        'target': target.astype(numpy.float32)})[0]
	return source.astype(numpy.float32), target.astype(numpy.float32)


def random_sliders(rng, strength=1.0, active=4):
	"""A face_editor-like edit: `active` sliders set, each uniform in [-strength, strength]."""
	s = {k: 0.0 for k in SLIDERS}
	for k in rng.choice(SLIDERS, size=active, replace=False):
		s[str(k)] = float(rng.uniform(-strength, strength))
	return s
