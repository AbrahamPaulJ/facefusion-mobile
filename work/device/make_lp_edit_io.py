"""Calibration and held-out I/O for lp_gencut made with REAL face_editor edits.

The first generator calibration (make_lp_io.py) had one edit, a 0.01 mouth move, so the
hourglass input encoded as [-0.06, 0.42]; a head turn or a smile puts its heatmaps near
+-1, which the NPU would clip. These edits are face_editor's own arithmetic
(work/pipeline/lp_editor.py), sliders at up to full strength.

  calib  12 crops from calib/lp_crops (the wide motion set): 8 random four-slider edits,
         head yaw at -1 and +1, a smile + open mouth, one unedited.
         -> calib/lp_gencut_edit/, calib/lp_gencut_list.txt (hg_input channels-last)
  io     the 6 held-out crops x 2 random edits, each with a SAME-frame unedited twin, refs
         from the ORIGINAL upstream generator. hg_input is NOT written: the device builds it
         with lp_hg_bench from fv/source/target, which is the chain the app will run.
         -> device/io/lp_edit/

    py -3.10 work/device/make_lp_edit_io.py
"""
import glob
import json
import os
import sys

import numpy
import onnx
import onnxruntime
from onnx import helper

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'pipeline'))
import lp_editor  # noqa: E402

HG = '/model/dense_motion_network/Reshape_12_output_0'
MNT = '/mnt/c/' + ROOT[3:].replace('\\', '/')


def main():
	fe, me = lp_editor.session('feature_extractor'), lp_editor.session('motion_extractor')
	gen = lp_editor.session('generator')
	m = onnx.load(os.path.join(ROOT, 'onnx', 'lp_generator.onnx'))
	m.graph.output.append(helper.make_tensor_value_info(HG, onnx.TensorProto.FLOAT, None))
	so = onnxruntime.SessionOptions()
	so.log_severity_level = 3
	hgs = onnxruntime.InferenceSession(m.SerializeToString(), so, providers=['CPUExecutionProvider'])
	rng = numpy.random.default_rng(2026)

	crops = sorted(glob.glob(os.path.join(ROOT, 'calib', 'lp_crops', '*.raw')))
	picks = [crops[int(i)] for i in numpy.linspace(0, len(crops) - 1, 12)]
	edits = [lp_editor.random_sliders(rng) for _ in range(8)]
	edits += [{'head_yaw': -1.0}, {'head_yaw': 1.0}, {'mouth_smile': 1.0, 'lip_open_ratio': 1.0}, {}]
	cal = os.path.join(ROOT, 'calib', 'lp_gencut_edit')
	os.makedirs(cal, exist_ok=True)
	lines, lo, hi = [], 0.0, 0.0
	for i, (c, s) in enumerate(zip(picks, edits)):
		x = numpy.fromfile(c, numpy.float32).reshape(1, 3, 256, 256)
		fv = fe.run(None, {'input': x})[0].astype(numpy.float32)
		src, tgt = lp_editor.apply_edit(me.run(None, {'input': x}), s)
		hg = hgs.run([HG], {'feature_volume': fv, 'source': src, 'target': tgt})[0]
		lo, hi = min(lo, float(hg.min())), max(hi, float(hg.max()))
		files = {'feature_volume': fv, 'source': src, 'target': tgt,
		         'hg_input': numpy.ascontiguousarray(hg.transpose(0, 2, 3, 4, 1))}
		parts = []
		for k, v in files.items():
			fn = '%s_%03d.raw' % (k, i)
			v.astype(numpy.float32).tofile(os.path.join(cal, fn))
			parts.append('%s:=%s/calib/lp_gencut_edit/%s' % (k, MNT, fn))
		lines.append(' '.join(parts))
	with open(os.path.join(ROOT, 'calib', 'lp_gencut_list.txt'), 'w', newline='\n') as f:
		f.write('\n'.join(lines) + '\n')
	print('calib: %d cases, hg_input range [%.3f, %.3f]' % (len(lines), lo, hi))

	io = os.path.join(ROOT, 'device', 'io', 'lp_edit')
	for d in ('in', 'ref'):
		os.makedirs(os.path.join(io, d), exist_ok=True)
	held = sorted(glob.glob(os.path.join(ROOT, 'calib', 'edtalk_target_heldout', '*.raw')))
	cases = []
	for i, c in enumerate(held):
		x = numpy.fromfile(c, numpy.float32).reshape(1, 3, 256, 256)
		fv = fe.run(None, {'input': x})[0].astype(numpy.float32)
		motion = me.run(None, {'input': x})
		for j in range(2):
			s = lp_editor.random_sliders(rng)
			src, tgt = lp_editor.apply_edit(motion, s)
			n = len(cases)
			for k, v in (('source', src), ('target', tgt)):
				v.tofile(os.path.join(io, 'in', '%s_%03d.raw' % (k, n)))
			gen.run(None, {'feature_volume': fv, 'source': src, 'target': tgt})[0].astype(
				numpy.float32).tofile(os.path.join(io, 'ref', 'edit_%03d.raw' % n))
			gen.run(None, {'feature_volume': fv, 'source': tgt, 'target': tgt})[0].astype(
				numpy.float32).tofile(os.path.join(io, 'ref', 'noedit_%03d.raw' % n))
			cases.append({'case': n, 'crop': i, 'sliders': {k: round(v, 3) for k, v in s.items() if v}})
	with open(os.path.join(io, 'cases.json'), 'w') as f:
		json.dump(cases, f, indent=1)
	print('io: %d edit cases on %d held-out crops -> %s' % (len(cases), len(held), io))


if __name__ == '__main__':
	main()
