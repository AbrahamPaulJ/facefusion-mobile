"""ncnn on device vs the ONNX host reference, on byte-identical inputs.

This is the check that ncnn-against-ncnn cannot make.  Every earlier measurement compared
ncnn fp16 to ncnn fp32, which isolates precision but is blind to a conversion that is
self-consistently wrong -- pnnx mis-lowering an op would score a perfect SNR against
itself.  Here the reference is the same ONNX graph `prepare_onnx.py` feeds to QAIRT.

The inputs come from `ncnn_run`'s own dumps (`<out>.inN.raw`), written channel by channel,
so they are already NCHW and are the exact bytes the device saw.  Reconstructing them here
from the same LCG would be measuring my reconstruction instead of the port.

    py -3.10 work/device/ncnn_vs_onnx.py <dir-with-ref_*.raw>
"""
import os
import sys

import numpy as np
import onnxruntime as ort

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ONNX = os.path.join(ROOT, 'onnx')

# name -> (onnx file, [input shapes NCHW], output shape as ncnn dumped it)
CASES = {
    'yoloface':  ('yoloface_8n_b1',        [(1, 3, 640, 640)],                (1, 20, 8400)),
    'fan2d':     ('2dfan4_heatmaps',       [(1, 3, 256, 256)],                (1, 68, 64, 64)),
    'arcface':   ('arcface_w600k_r50_b1',  [(1, 3, 112, 112)],                (1, 512)),
    # hyperswap's fp32-weight build; source[1,512] BEFORE target[1,3,256,256].
    'hyperswap': ('hyperswap_1a_256_fp32', [(1, 512), (1, 3, 256, 256)],      (1, 3, 256, 256)),
}


def snr(ref, got):
    err = ref - got
    denom = float((err ** 2).sum())
    if denom <= 0:
        return float('inf')
    return 10.0 * np.log10(float((ref ** 2).sum()) / denom)


def main(d):
    print('%-10s %10s %12s %12s   %s' % ('model', 'SNR dB', 'max|err|', 'rel_rms', 'shape'))
    print('-' * 74)
    worst = None
    for name, (onnx_name, in_shapes, out_shape) in CASES.items():
        path = os.path.join(ONNX, onnx_name + '.onnx')
        if not os.path.exists(path):
            print('%-10s  MISSING %s' % (name, path))
            continue

        feeds = {}
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        for i, (inp, shape) in enumerate(zip(sess.get_inputs(), in_shapes)):
            raw = np.fromfile(os.path.join(d, 'ref_%s.raw.in%d.raw' % (name, i)),
                              dtype=np.float32)
            assert raw.size == int(np.prod(shape)), \
                '%s in%d: %d floats, want %d' % (name, i, raw.size, int(np.prod(shape)))
            feeds[inp.name] = raw.reshape(shape)

        ref = sess.run(None, feeds)[0].astype(np.float32).ravel()
        got = np.fromfile(os.path.join(d, 'ref_%s.raw' % name), dtype=np.float32)

        # ncnn drops the leading batch and pads channel strides; the dump is already
        # channel-major, so a ravel on both sides lines up element for element.
        n = min(ref.size, got.size)
        if ref.size != got.size:
            print('%-10s  size mismatch: onnx %d vs ncnn %d -- comparing first %d'
                  % (name, ref.size, got.size, n))
        ref, got = ref[:n], got[:n]

        s = snr(ref, got)
        rel = float(np.sqrt((( ref - got) ** 2).mean()) / (np.sqrt((ref ** 2).mean()) + 1e-30))
        print('%-10s %10.2f %12.6f %12.6f   %s'
              % (name, s, float(np.abs(ref - got).max()), rel, out_shape))
        worst = s if worst is None else min(worst, s)
    print('-' * 74)
    if worst is not None:
        print('worst: %.2f dB' % worst)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
