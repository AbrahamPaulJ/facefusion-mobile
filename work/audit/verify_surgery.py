"""Assert every prepared/simplified graph reproduces its original on REAL calibration data.

Surgery that silently changes numerics is the failure mode this whole project cannot
afford, so it is checked before a single conversion runs.
"""
import glob, os, sys
import numpy, onnxruntime

def sess(p):
    o = onnxruntime.SessionOptions(); o.log_severity_level = 3
    return onnxruntime.InferenceSession(p, o, providers=['CPUExecutionProvider'])

def snr(ref, got):
    ref, got = ref.astype(numpy.float64).ravel(), got.astype(numpy.float64).ravel()
    n = ((ref - got) ** 2).sum()
    return float('inf') if n == 0 else 10 * numpy.log10((ref ** 2).sum() / n)

CASES = [
    ('arcface',   'work/models/arcface_w600k_r50.onnx', 'work/onnx/arcface_w600k_r50_b1_sim.onnx',
     'work/calib/arcface/*.raw',   (1,3,112,112),  ['input'], None),
    ('2dfan4',    'work/models/2dfan4.onnx',            'work/onnx/2dfan4_heatmaps_sim.onnx',
     'work/calib/fan2d/*.raw',     (1,3,256,256),  ['input'], 'heatmaps'),
    ('yoloface',  'work/models/yoloface_8n.onnx',       'work/onnx/yoloface_8n_b1_sim.onnx',
     'work/calib/yoloface/*.raw',  (1,3,640,640),  ['input'], None),
    ('hyperswap', 'work/models/hyperswap_1a_256.onnx',  'work/onnx/hyperswap_1a_256_sim.onnx',
     'work/calib/swap_target/*.raw', (1,3,256,256), ['target','source'], 'output'),
    # The enhancer's calibration crops are the SWAPPER'S OUTPUT, not raw target frames --
    # it only ever sees a face that hyperswap already wrote.  Capture them from the stage
    # boundary in work/pipeline, or this measures a distribution the model never meets.
    ('gpen',      'work/models/gpen_bfr_256.onnx',       'work/onnx/gpen_bfr_256_sim.onnx',
     'work/calib/gpen/*.raw',      (1,3,256,256),  ['input'], 'output'),
]

n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 8
worst_overall = None
n_checked = 0
for name, orig_p, new_p, pattern, shape, inputs, out_name in CASES:
    if not os.path.exists(new_p):
        print('%-11s SKIP (no %s)' % (name, new_p)); continue
    # A graph WITH no calibration data used to reach the print below with worst=None and
    # die on the %-format.  Say what is missing instead: an absent calib dir is a thing to
    # go capture, not a crash.
    if not sorted(glob.glob(pattern)):
        print('%-11s SKIP (graph is built, but no calibration data at %s)' % (name, pattern))
        continue
    a, b = sess(orig_p), sess(new_p)
    a_names = [i.name for i in a.get_inputs()]
    a_outs  = [o.name for o in a.get_outputs()]
    b_outs  = [o.name for o in b.get_outputs()]
    files = sorted(glob.glob(pattern))[:n_cases]
    src_files = sorted(glob.glob('work/calib/swap_source/*.raw'))[:n_cases]
    worst = None
    for k, f in enumerate(files):
        x = numpy.fromfile(f, numpy.float32).reshape(shape)
        feed = {inputs[0]: x}
        if 'source' in inputs:
            feed['source'] = numpy.fromfile(src_files[k], numpy.float32).reshape(1, 512)
        ra = a.run(None, {n: feed[n] for n in a_names})
        rb = b.run(None, {n: feed[n] for n in [i.name for i in b.get_inputs()]})
        tgt = out_name or a_outs[0]
        ref = ra[a_outs.index(tgt)]
        got = rb[b_outs.index(tgt)]
        s = snr(ref, got)
        worst = s if worst is None else min(worst, s)
    print('%-11s %2d cases  worst SNR vs original: %s dB   (output `%s`)'
          % (name, len(files), ('inf' if worst == float('inf') else '%.1f' % worst),
             out_name or a_outs[0]))
    n_checked += 1
    if worst != float('inf'):
        worst_overall = worst if worst_overall is None else min(worst_overall, worst)

# Distinguish "every surgery was exact" from "nothing ran".  The summary used to print
# `all bit-exact` for both, which reads as a pass when in fact nothing was checked.
if not n_checked:
    print('\nNOTHING CHECKED - no graph had both a build and calibration data')
else:
    print('\n%d surgeries checked, worst finite SNR: %s dB' %
          (n_checked, 'none - all bit-exact' if worst_overall is None
           else '%.1f' % worst_overall))
