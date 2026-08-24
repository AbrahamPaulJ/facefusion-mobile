"""Report the quantised bitwidth of EVERY operand of every MatMul / FullyConnected.

Why this exists (Phase 5, trap #26): QNN treats a MatMul's *second* operand as a
"weight", and 16-bit dynamic weights are off by default -- so K and V, which are pure
activations, silently land at 8 bits in every attention block. That is invisible in
analyze_net.py (shapes and transposes are fine) and invisible in analyze_encodings.py
(it only ranks ranges). It is only visible in the operand dtypes.

htp.json DOES list u16 x u16 -> u16 MatMul as a supported kernel, so the 8-bit operand
is a converter default, not a hardware limit. `--use_dynamic_16_bit_weights` plus
`--restrict_quantization_steps "-0x8000 0x7F7F"` is the fix.

  usage: py -3.10 work/device/matmul_bits.py <net.json> [<net.json> ...]

Prints one row per (op type, operand bitwidth pattern) with counts, so a whole 18-block
graph collapses to a handful of lines. Pass two files to A/B a conversion.

Note the QNN dtype encoding (trap #12): UFIXED_16 is 0x0416 -- the low byte spells the
width in DECIMAL, so it must be read with %02x, not as hex.
"""

import json
import sys
from collections import Counter

TARGET = ("MatMul", "FullyConnected", "Conv2d", "DepthWiseConv2d")


def width(t):
    dt = t["data_type"]
    if dt in (0x0032, 0x0216):  # FLOAT_32 / FLOAT_16
        return "f32" if dt == 0x0032 else "f16"
    bits = int("{:02x}".format(dt & 0xFF))
    sign = "s" if (dt & 0xFF00) in (0x0008, 0x0016, 0x0032, 0x0300) else "u"
    return "{}{}".format(sign, bits)


def report(path):
    g = json.load(open(path))["graph"]
    tensors, nodes = g["tensors"], g["nodes"]
    static = {n for n, t in tensors.items() if t.get("type") == 4}

    pat = Counter()
    for name, n in nodes.items():
        if n["type"] not in TARGET:
            continue
        ins = []
        for i in n.get("input_names", []):
            t = tensors.get(i)
            if t is None:
                continue
            ins.append(width(t) + ("*" if i in static else ""))
        out = [width(tensors[o]) for o in n.get("output_names", []) if o in tensors]
        pat[(n["type"], tuple(ins), tuple(out))] += 1

    print("=" * 78)
    print(path.split("/")[-1])
    print("  '*' marks a STATIC tensor (a real weight). No star = an activation.")
    print("=" * 78)
    for (op, ins, out), c in sorted(pat.items(), key=lambda kv: -kv[1]):
        print("  {:>4}x  {:<16} in {:<28} -> out {}".format(
            c, op, " x ".join(ins), ",".join(out)))

    dyn8 = sum(c for (op, ins, _), c in pat.items()
               if op == "MatMul" and any(i == "u8" for i in ins))
    print("")
    print("  MatMuls with an 8-bit DYNAMIC operand: {}".format(dyn8))
    return dyn8


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
        print("")
