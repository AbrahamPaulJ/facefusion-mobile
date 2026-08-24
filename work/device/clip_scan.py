"""Does the device CLIP? Compare every activation's real fp32 range against the
encoding the converter calibrated for it.

Why this is not already ruled out. `quant_ablate.py` fake-quantises with `q_minmax`,
which derives the range from the tensor **it is quantising** -- so by construction it
can never saturate. `resid_ablate.py` does use the real `net.json` encodings, but only
on the 35 block boundaries. Nothing has yet asked the one question that separates a
faithful host simulation from the hardware: **on the held-out cases, does any activation
leave the range the calibration set taught the graph to expect?**

If it does, the device clamps and the host simulation cannot reproduce it -- which is
exactly the shape of the 43 dB vs 19.48 dB discrepancy.

Encoding convention, verified: real = scale * (q + offset), q in [0, 2**bits - 1],
so the representable interval is [scale*offset, scale*(offset + 2**bits - 1)].

  usage: py -3.10 work/device/clip_scan.py --stage 0 --cases 0,12,8 [--chunk 400]
         (case 0 = 14.51 dB and case 12 = 12.16 dB are the two worst; 8 = 22.40 dB
          is the best, so it doubles as a control)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
ND = W.parent.parent
sys.path.insert(0, str(W))
from layer_snr import sanitize                       # noqa: E402


def encodings(net_path):
    """sanitised tensor name -> (lo, hi, bits) for every quantised activation."""
    g = json.load(open(net_path))["graph"]
    out = {}
    for name, t in g["tensors"].items():
        qp = t.get("quant_params") or {}
        se = qp.get("scale_offset")
        if not se:
            continue
        # net.json states the calibrated interval outright -- no need to reconstruct it
        # from scale/offset, and no chance of getting trap #12's dtype byte wrong.
        lo, hi = se.get("minimum"), se.get("maximum")
        if lo is None or hi is None or hi <= lo:
            continue
        out[name] = (lo, hi, se.get("bitwidth", 16), se.get("scale"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--cases", default="0,12,8")
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    import onnx
    import onnxruntime as ort

    net = W / "mmdit_s{}_net.json".format(a.stage)
    enc = encodings(net)
    print("quantised tensors with an encoding: {}".format(len(enc)))

    onnx_path = ND / "work/onnx/mmdit_s{}_env/mmdit_s{}_env_raw.onnx".format(a.stage, a.stage)
    model = onnx.load(str(onnx_path), load_external_data=False)
    g = model.graph
    onnx_of, order = {}, {}
    for i, node in enumerate(g.node):
        for o in node.output:
            s = sanitize(o)
            if s in enc and s not in onnx_of:
                onnx_of[s], order[s] = o, i
    have = sorted(onnx_of, key=lambda s: order[s])
    print("joinable to ONNX node outputs: {}".format(len(have)))

    io_dir = W / "mio_s{}".format(a.stage)
    dims = {}
    for line in open(io_dir / "dims.txt"):
        m = re.search(r"--input_dim\s+(\S+)\s+([0-9,]+)", line)
        if m:
            dims[m.group(1)] = tuple(int(x) for x in m.group(2).split(","))

    cases = [int(c) for c in a.cases.split(",")]
    worst = {}                                        # name -> (overflow, case, amin, amax)
    base_out = [o.name for o in g.output]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    tmp = onnx_path.parent / "_clip_scan_tmp.onnx"

    for start in range(0, len(have), a.chunk):
        batch = have[start:start + a.chunk]
        del g.output[:]
        for nm in base_out + [onnx_of[s] for s in batch]:
            g.output.extend([onnx.helper.make_empty_tensor_value_info(nm)])
        onnx.save(model, str(tmp))
        sess = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"])
        names = [o.name for o in sess.get_outputs()]
        for c in cases:
            feed = {nm: np.fromfile(io_dir / "{}_{:04d}.raw".format(nm, c),
                                    np.float32).reshape(shp)
                    for nm, shp in dims.items()}
            got = dict(zip(names, sess.run(None, feed)))
            for s in batch:
                v = np.asarray(got[onnx_of[s]], np.float64)
                lo, hi, bits, sc = enc[s]
                amin, amax = float(v.min()), float(v.max())
                span = hi - lo
                over = max((amax - hi) / span if amax > hi else 0.0,
                           (lo - amin) / span if amin < lo else 0.0)
                nclip = int((v > hi).sum() + (v < lo).sum())
                prev = worst.get(s)
                if prev is None or over > prev[0]:
                    worst[s] = (over, c, amin, amax, lo, hi, nclip, v.size, bits)
            del got
        del sess
        print("  ... {}/{}".format(min(start + a.chunk, len(have)), len(have)))
    tmp.unlink(missing_ok=True)

    rows = sorted(worst.items(), key=lambda kv: -kv[1][0])
    n_over = sum(1 for _, r in rows if r[0] > 0)
    n_clipped_elems = sum(r[6] for _, r in rows)
    print("")
    print("=" * 100)
    print("tensors whose real range LEAVES the calibrated encoding: {} of {}".format(
        n_over, len(rows)))
    print("total clipped elements across the scanned cases: {}".format(n_clipped_elems))
    print("=" * 100)
    print("  {:>9} {:>5} {:>4} {:>12} {:>12} {:>12} {:>12} {:>10}  {}".format(
        "over(span)", "case", "bits", "act min", "act max", "enc lo", "enc hi",
        "clipped", "tensor"))
    for s, r in rows[:a.top]:
        over, c, amin, amax, lo, hi, nclip, size, bits = r
        print("  {:>9.3f} {:>5} {:>4} {:>12.4g} {:>12.4g} {:>12.4g} {:>12.4g} "
              "{:>5}/{:<6}  {}".format(over, c, bits, amin, amax, lo, hi, nclip, size, s))


if __name__ == "__main__":
    main()
