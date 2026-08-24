"""Layerwise device-vs-fp32 SNR for a QNN graph, matched automatically by tensor name.

`block_snr.py` answers "which block", using a torch reference and 35 hand-derived
boundary tensors. This answers "which OP", using the ONNX the graph was converted from
and every tensor the device dumped -- because the converter's tensor names are the ONNX
names with the non-identifier characters replaced:

    /norm1/norm.12/Constant_1_output_0   ->   _norm1_norm_12_Constant_1_output_0

so the two sides join on a mechanical sanitisation and nothing has to be matched by hand.

The reference is onnxruntime on the SAME ONNX file that was converted, fed the SAME
input raws the device was fed, so any difference is the deployed graph's -- quantisation,
op-internal precision, or a prep-time rewrite -- and not an export discrepancy.

Memory is the real constraint: this ONNX is 5.6 GB of fp32 weights and marking every
intermediate as a graph output keeps them all alive at once. --chunk runs the model
repeatedly with a subset of extra outputs each time; 300 is comfortable on 16 GB.

  usage:
    py -3.10 work/device/layer_snr.py --dump work/device/ddbg_s0 --stage 0
    py -3.10 work/device/layer_snr.py --dump ... --filter norm --chunk 200
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
ND = W.parent.parent
TEXT_TOKENS = 128


def sanitize(name):
    """The converter's tensor-name mangling: anything not [A-Za-z0-9_] becomes '_'."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def snr(ref, got):
    ref = np.asarray(ref, np.float64).ravel()
    got = np.asarray(got, np.float64).ravel()
    if ref.shape != got.shape:
        return None
    d = np.linalg.norm(ref - got)
    if d == 0:
        return float("inf")
    r = np.linalg.norm(ref)
    return float("inf") if r == 0 else 20 * np.log10(r / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="device output dir (Result_0 inside)")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--chunk", type=int, default=300)
    ap.add_argument("--filter", default=None, help="only tensors whose name contains this")
    ap.add_argument("--top", type=int, default=60, help="worst-N table size")
    ap.add_argument("--name", default="",
                    help="variant, e.g. mmdit_s0f -- picks its ONNX and its mio_ dir")
    a = ap.parse_args()
    suf = (a.name[6:] if a.name.startswith("mmdit_") else a.name) or "s{}".format(a.stage)

    import onnx
    import onnxruntime as ort

    onnx_path = Path(a.onnx or ND / "work/onnx/mmdit_{0}_env/mmdit_{0}_env_raw.onnx"
                     .format(suf))
    dump = Path(a.dump)
    res = dump / "Result_{}".format(a.case)
    if not res.is_dir():
        res = dump                                  # some dumps are flat
    raws = {p.stem: p for p in res.rglob("*.raw")}
    if not raws:
        sys.exit("no .raw files under {}".format(res))
    print("device dump: {} tensors under {}".format(len(raws), res))

    model = onnx.load(str(onnx_path), load_external_data=False)
    g = model.graph
    order = {}                                      # sanitized -> topological index
    onnx_of = {}
    for i, node in enumerate(g.node):
        for o in node.output:
            s = sanitize(o)
            if s not in order:
                order[s], onnx_of[s] = i, o

    have = [s for s in raws if s in onnx_of]
    if a.filter:
        have = [s for s in have if a.filter in s]
    have.sort(key=lambda s: order[s])
    print("joinable to ONNX: {} of {}".format(len(have), len(raws)))
    miss = sorted(set(raws) - set(onnx_of))
    if miss:
        print("  unmatched device tensors (converter-inserted, expected): {}"
              .format(len(miss)))

    # ---- inputs: exactly the raws the device was fed --------------------------
    io_dir = W / "mio_{}".format(suf)
    dims = {}
    for line in open(io_dir / "dims.txt"):
        m = re.search(r"--input_dim\s+(\S+)\s+([0-9,]+)", line)
        if m:
            dims[m.group(1)] = tuple(int(x) for x in m.group(2).split(","))
    feed = {}
    for nm, shp in dims.items():
        p = io_dir / "{}_{:04d}.raw".format(nm, a.case)
        feed[nm] = np.fromfile(p, np.float32).reshape(shp)
    print("inputs: {}".format(", ".join("{}{}".format(k, list(v.shape))
                                        for k, v in feed.items())))

    base_out = [o.name for o in g.output]
    rows = []
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for start in range(0, len(have), a.chunk):
        batch = have[start:start + a.chunk]
        del g.output[:]
        for nm in base_out:
            g.output.extend([onnx.helper.make_empty_tensor_value_info(nm)])
        for s in batch:
            g.output.extend([onnx.helper.make_empty_tensor_value_info(onnx_of[s])])
        tmp = onnx_path.parent / "_layer_snr_tmp.onnx"
        onnx.save(model, str(tmp))                  # external data stays where it is
        sess = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"])
        names = [o.name for o in sess.get_outputs()]
        outs = sess.run(None, feed)
        got = dict(zip(names, outs))
        for s in batch:
            ref = np.asarray(got[onnx_of[s]])
            dev = np.fromfile(raws[s], np.float32)
            v = snr(ref.ravel(), dev)
            rows.append((order[s], s, ref.size, dev.size, v,
                         float(np.abs(ref).max()) if ref.size else 0.0))
        del sess, outs, got
        print("  ... {}/{}".format(min(start + a.chunk, len(have)), len(have)))
        tmp.unlink(missing_ok=True)

    print("")
    print("=" * 92)
    print("layerwise SNR, topological order (device vs onnxruntime fp32)")
    print("=" * 92)
    print("  {:>5} {:>10} {:>12} {:>11}  {}".format("idx", "SNR dB", "max|ref|", "elems", "tensor"))
    prev = None
    for idx, s, n, dn, v, mx in rows:
        if v is None:
            note = "SHAPE {} vs {}".format(n, dn)
            print("  {:>5} {:>10} {:>12.4g} {:>11}  {}  {}".format(idx, "-", mx, n, s, note))
            continue
        drop = "" if prev is None else "{:+.1f}".format(v - prev)
        prev = v
        print("  {:>5} {:>10.2f} {:>12.4g} {:>11}  {} {}".format(idx, v, mx, n, s, drop))

    ok = [r for r in rows if r[4] is not None]
    if ok:
        ok.sort(key=lambda r: r[4])
        print("")
        print("worst {} tensors by SNR:".format(min(a.top, len(ok))))
        for idx, s, n, dn, v, mx in ok[:a.top]:
            print("  {:>8.2f} dB  max|ref| {:>11.4g}  {:>9} elems  {}".format(v, mx, n, s))


if __name__ == "__main__":
    main()
