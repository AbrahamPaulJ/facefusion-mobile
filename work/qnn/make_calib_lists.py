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
}


def build(name, limit=0):
	spec = SPECS[name]
	cols = []
	for input_name, subdir, elems in spec:
		files = sorted(glob.glob(os.path.join(ROOT, 'calib', subdir, '*.raw')))
		if not files:
			sys.exit('no calibration files in calib/%s -- run the reference harness first' % subdir)
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
	names = sys.argv[1:] or list(SPECS)
	limit = 0
	if names and names[0].isdigit():
		limit = int(names[0]); names = names[1:] or list(SPECS)
	for n in names:
		if n in SPECS:
			build(n, limit)
