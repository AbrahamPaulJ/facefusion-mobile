"""onnxsim over the prepared graphs, producing the `_sim.onnx` files convert.sh reads.

This step existed nowhere in the repo.  `prepare_onnx.py` writes the SURGERY output
(`arcface_w600k_r50_b1.onnx`, `2dfan4_heatmaps.onnx`, `yoloface_8n_b1.onnx`), and
`convert.sh` reads `..._sim.onnx` -- so a clean rebuild from the documented commands
stopped at `ERROR: no .../arcface_w600k_r50_b1_sim.onnx`, with nothing to say what made
it.  Three of the six graphs run onnxsim inside their own task (`nsfw`, `gpen`, and
`hyperswap`'s fp32 demotion); these three did not, and the gap was invisible until every
intermediate was deleted at once.

    py -3.10 work/export/simplify_onnx.py

Node counts are printed because they are the check: session 2 recorded yoloface 350 -> 268
and 2dfan4 -> 730, and a run that does not reproduce those is simplifying a different graph.
"""
import collections
import os
import sys

import onnx
import onnxsim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

CASES = [
    ('onnx/arcface_w600k_r50_b1.onnx', 'onnx/arcface_w600k_r50_b1_sim.onnx',
     {'input': [1, 3, 112, 112]}),
    ('onnx/2dfan4_heatmaps.onnx',      'onnx/2dfan4_heatmaps_sim.onnx',
     {'input': [1, 3, 256, 256]}),
    ('onnx/yoloface_8n_b1.onnx',       'onnx/yoloface_8n_b1_sim.onnx',
     {'input': [1, 3, 640, 640]}),
]


def main():
    missing = 0
    for src, dst, shapes in CASES:
        s_p, d_p = os.path.join(ROOT, src), os.path.join(ROOT, dst)
        if not os.path.exists(s_p):
            print('%-34s MISSING -- run prepare_onnx.py first' % os.path.basename(src))
            missing += 1
            continue
        m = onnx.load(s_p)
        before = len(m.graph.node)
        s, ok = onnxsim.simplify(m, overwrite_input_shapes=shapes)
        if not ok:
            sys.exit('onnxsim reported failure on %s' % src)
        onnx.save(s, d_p)
        c = collections.Counter(n.op_type for n in s.graph.node)
        print('%-34s %4d -> %4d nodes, %2d op types  %6.1f MB'
              % (os.path.basename(dst), before, len(s.graph.node), len(c),
                 os.path.getsize(d_p) / 1e6))
    sys.exit(1 if missing else 0)


if __name__ == '__main__':
    main()
