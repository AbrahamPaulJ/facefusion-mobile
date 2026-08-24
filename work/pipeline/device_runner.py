"""Run one converted graph on the phone for a BATCH of inputs, from Python.

The pipeline is sequential across stages (detector -> landmarker -> recognizer -> swapper)
but every frame is independent *within* a stage.  So a whole clip needs only four device
invocations, not four per frame -- which matters because each qnn-net-run call reloads the
context binary, and that load dominates a single inference.

    r = DeviceRunner()
    outs = r.run('yoloface', {'input': [arr0, arr1, ...]})   # -> list of dicts
"""
import os
import shutil
import subprocess
import tempfile

import numpy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ADB = os.environ.get('FF_ADB', 'adb')
SERIAL = os.environ.get('FF_ADB_SERIAL', '')
REMOTE = '/data/local/tmp/ff'


class DeviceError(RuntimeError):
	pass


class DeviceRunner:
	def __init__(self, verbose=True):
		self.verbose = verbose
		self._connect()

	def _adb(self, *args, check=True):
		p = subprocess.run([ADB, '-s', SERIAL] + list(args),
						   capture_output=True, text=True)
		# adb writes progress to stderr on SUCCESS (trap #14) -- returncode is the only
		# honest status.
		if check and p.returncode != 0:
			raise DeviceError('adb %s failed: %s' % (' '.join(args), p.stderr.strip()))
		return p.stdout + p.stderr

	def _connect(self):
		subprocess.run([ADB, 'connect', SERIAL], capture_output=True, text=True)
		out = subprocess.run([ADB, 'devices'], capture_output=True, text=True).stdout
		if SERIAL not in out or 'offline' in out:
			raise DeviceError('phone not reachable at %s\n%s' % (SERIAL, out))

	def run(self, name, feeds, ctx=None):
		"""feeds: {input_name: list-of-arrays}, all lists the same length.

		Returns a list of {output_name: array}, one per case.
		"""
		ctx = ctx or name
		# Stage under io/e2e_<name>, NOT io/<name>: the measurement harness owns io/<name>
		# and holds the held-out set that every deploy-SNR number is computed against.
		# Sharing the directory silently replaced those inputs with this run's frames and
		# made a later A/B compare device outputs to references from different inputs.
		io_name = 'e2e_' + name
		names = list(feeds)
		n = len(feeds[names[0]])
		for k in names:
			if len(feeds[k]) != n:
				raise ValueError('input %s has %d cases, expected %d' % (k, len(feeds[k]), n))

		staging = tempfile.mkdtemp(prefix='ffdev_')
		try:
			indir = os.path.join(staging, 'in')
			os.makedirs(indir)
			lines = []
			for i in range(n):
				parts = []
				for k in names:
					fn = '%s_%s_%04d.raw' % (name, k, i)
					numpy.ascontiguousarray(feeds[k][i], numpy.float32).tofile(
						os.path.join(indir, fn))
					rp = '%s/io/%s/in/%s' % (REMOTE, io_name, fn)
					parts.append(('%s:=%s' % (k, rp)) if len(names) > 1 else rp)
				lines.append(' '.join(parts))
			list_path = os.path.join(staging, 'input_list.txt')
			with open(list_path, 'w', newline='\n') as fh:
				fh.write('\n'.join(lines) + '\n')

			self._adb('shell', 'rm -rf %s/io/%s %s/out/%s' % (REMOTE, io_name, REMOTE, ctx))
			self._adb('shell', 'mkdir -p %s/io/%s' % (REMOTE, io_name))
			self._adb('push', indir, '%s/io/%s/in' % (REMOTE, io_name))
			self._adb('push', list_path, '%s/io/%s/input_list.txt' % (REMOTE, io_name))

			log = self._adb('shell', 'sh %s/run_model.sh %s acc %s' % (REMOTE, io_name, ctx))
			if 'Finished Executing Graphs' not in log:
				raise DeviceError('%s did not finish:\n%s' % (name, log[-1500:]))
			if self.verbose:
				print('   device %-10s %3d cases' % (name, n))

			outdir = os.path.join(staging, 'out')
			self._adb('pull', '%s/out/%s' % (REMOTE, ctx), outdir)

			results = []
			for i in range(n):
				rdir = os.path.join(outdir, 'Result_%d' % i)
				if not os.path.isdir(rdir):
					raise DeviceError('missing %s' % rdir)
				case = {}
				for f in os.listdir(rdir):
					if f.endswith('.raw'):
						case[os.path.splitext(f)[0]] = numpy.fromfile(
							os.path.join(rdir, f), numpy.float32)
				results.append(case)
			return results
		finally:
			shutil.rmtree(staging, ignore_errors=True)
