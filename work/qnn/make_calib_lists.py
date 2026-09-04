"""Write qnn-onnx-converter --input_list files, with WSL paths.

Multi-input graphs need `name:=path` pairs on one line; single-input graphs take a bare
path.  Every file is stat'd here so a missing raw fails now rather than 40 minutes into a
conversion.
"""
import glob
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# The same tree as ROOT, addressed the way WSL sees it.
WSL_ROOT = os.environ.get('FF_WSL_ROOT') or (
	'/mnt/' + ROOT[0].lower() + ROOT[2:].replace(os.sep, '/'))

SPECS = {
	# name        : [(input_name, calib_subdir, expected_elems)]
	'arcface':   [('input',  'arcface',     1 * 3 * 112 * 112)],
	'fan2d':     [('input',  'fan2d',       1 * 3 * 256 * 256)],
	'yoloface':  [('input',  'yoloface',    1 * 3 * 640 * 640)],
	'hyperswap': [('target', 'swap_target', 1 * 3 * 256 * 256),
				  ('source', 'swap_source', 1 * 512)],
	'inswapper': [('target', 'swap_target_128', 1 * 3 * 128 * 128),
				  ('source', 'swap_source_128', 1 * 512)],
	# The content gate.  Its frames are letterboxed by fit_contain_frame -- centred pad,
	# NOT the detector's corner pad -- so they come from their own capture.
	'nsfw':      [('input',  'nsfw',        1 * 3 * 384 * 384)],
	# The face enhancer. Its calibration is the SWAPPER'S OUTPUT crop, captured downstream
	# of the swapper in run_reference.swap_face -- gpen never sees a target crop, only a
	# face hyperswap has already written, so `swap_target` would be the wrong distribution.
	'gpen':      [('input',  'gpen',        1 * 3 * 256 * 256)],
	# The lip syncer. `source` is the mel window and `target` is the masked crop
	# concatenated with the reference crop, both captured by
	# work/pipeline/capture_lipsync_calib.py from REAL frames through the real crop chain
	# -- make_lipsync_calib.py's fan2d stand-ins were for verifying the conversion and are
	# the wrong distribution to quantise against (trap #4).
	'wav2lip':   [('source', 'lipsync_source', 1 * 1 * 80 * 16),
				  ('target', 'lipsync_target', 1 * 6 * 96 * 96)],
	# The 256x256 lip syncer. Three inputs, and `target` is the WHOLE face crop rather
	# than wav2lip's masked/reference 96 pair -- 256x256 IS the crop, which is the point
	# of converting it. `weight` is the lip-direction scale, driven at 1.0.
	# ⚠ Same trap #4 warning as wav2lip's: make_edtalk_calib.py's fan2d stand-ins verify
	# that the CONTEXT BINARY computes what onnxruntime does. Quantising against them
	# would be quantising against the wrong distribution -- the real set has to come
	# through edtalk's own crop chain, the way capture_lipsync_calib.py does for wav2lip.
	'edtalk':    [('source', 'edtalk_source', 1 * 1 * 80 * 16),
				  ('target', 'edtalk_target', 1 * 3 * 256 * 256),
				  ('weight', 'edtalk_weight', 1)],
}


def build(name, limit=0, required=True):
	"""`required=False` reports a missing set and returns instead of exiting.

	Bare `make_calib_lists.py` walks every spec, and `inswapper` has no calibration on this
	machine -- so the unconditional exit killed the run at inswapper and silently never
	reached `nsfw` or `gpen`, which come after it in SPECS order.  A set asked for BY NAME
	still exits: there the absence is the answer to the question.
	"""
	spec = SPECS[name]
	cols = []
	for input_name, subdir, elems in spec:
		files = sorted(glob.glob(os.path.join(ROOT, 'calib', subdir, '*.raw')))
		if not files:
			msg = 'no calibration files in calib/%s -- run the reference harness first' % subdir
			if required:
				sys.exit(msg)
			print('%-10s SKIP (%s)' % (name, msg))
			return None
		for f in files:
			got = os.path.getsize(f) // 4
			if got != elems:
				sys.exit('%s: %s has %d float32s, expected %d' % (name, f, got, elems))
		cols.append((input_name, files))

	n = min(len(f) for _, f in cols)
	if limit:
		n = min(n, limit)
	multi = len(cols) > 1

	out_dir = os.path.join(ROOT, 'calib')
	path = os.path.join(out_dir, '%s_list.txt' % name)
	with open(path, 'w', newline='\n') as fh:
		for i in range(n):
			parts = []
			for input_name, files in cols:
				wsl = WSL_ROOT + '/calib/' + os.path.relpath(files[i], os.path.join(ROOT, 'calib')).replace('\\', '/')
				parts.append(('%s:=%s' % (input_name, wsl)) if multi else wsl)
			fh.write(' '.join(parts) + '\n')
	print('%-10s %3d samples -> calib/%s_list.txt' % (name, n, name))
	return path


if __name__ == '__main__':
	explicit = bool(sys.argv[1:])
	names = sys.argv[1:] or list(SPECS)
	limit = 0
	if names and names[0].isdigit():
		limit = int(names[0]); names = names[1:] or list(SPECS)
		explicit = bool(names) and explicit
	for n in names:
		if n in SPECS:
			build(n, limit, required=explicit)
