#!/usr/bin/env python3
"""Static analysis of a QNN *_net.json (converter output) -- no device needed.

Answers "where does this graph spend its bytes?" before spending a device session
on profiling. Written 2026-08-21 while chasing the Phase 4 24x latency gap; the
finding was that the VAE decoder spends more data movement on layout Transposes
than on its convolutions.

  usage: py -3.10 work/device/analyze_net.py <net.json> [<net.json> ...]
"""
import json
import sys
import collections

# QNN encodes datatype as 0xCCBB where CC is the category and the BB hex digits
# spell the bit width in *decimal* (QnnTypes.h:130-180). 0x0416 is UFIXED_16, so
# the width is 16 bits, not 0x16=22. Reading it as hex silently doubles every
# 16-bit tensor, so parse the digits as text.
CATEGORY = {0x00: "INT", 0x01: "UINT", 0x02: "FLOAT", 0x03: "SFIXED",
            0x04: "UFIXED", 0x05: "BOOL", 0x06: "STRING"}
COMPUTE = ("Conv2d", "Conv3d", "MatMul", "FullyConnected", "DepthWiseConv2d")


def dtype_name(code):
    cat = CATEGORY.get(code >> 8, hex(code >> 8))
    return f"{cat}_{code & 0xFF:02x}"


def dtype_bits(code):
    return int(f"{code & 0xFF:02x}")


def tensor_bytes(tensors, name):
    t = tensors[name]
    n = 1
    for d in t["dims"]:
        n *= d
    return n * dtype_bits(t["data_type"]) // 8


def report(path):
    graph = json.load(open(path))["graph"]
    nodes, tensors = graph["nodes"], graph["tensors"]

    print(f"\n===== {path} =====")
    print(f"nodes={len(nodes)}  tensors={len(tensors)}")

    hist = collections.Counter(dtype_name(t["data_type"]) for t in tensors.values())
    print("tensor dtypes:", dict(hist.most_common()))
    floats = [n for n, t in tensors.items() if t["data_type"] >> 8 == 0x02]
    print(f"float tensors (unquantised fallback): {len(floats)}")

    by_type = collections.Counter(n["type"] for n in nodes.values())
    print("\nop histogram:")
    for op, count in by_type.most_common():
        print(f"    {op:24} {count:4}")

    def total(pred):
        return sum(tensor_bytes(tensors, n["output_names"][0])
                   for n in nodes.values() if pred(n))

    moved = total(lambda n: n["type"] == "Transpose")
    computed = total(lambda n: n["type"] in COMPUTE)
    print(f"\nTranspose output bytes : {moved/1e6:10.2f} MB")
    print(f"compute  output bytes  : {computed/1e6:10.2f} MB")
    if computed:
        print(f"ratio transpose/compute: {moved/computed:10.2f}")

    rows = sorted(((tensor_bytes(tensors, n["output_names"][0]), name, n)
                   for name, n in nodes.items() if n["type"] == "Transpose"),
                  reverse=True)
    print("\nlargest Transposes:")
    for nbytes, name, node in rows[:10]:
        dims = tensors[node["output_names"][0]]["dims"]
        print(f"    {nbytes/1e6:8.2f} MB  {str(dims):24} {name[:52]}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        report(arg)
