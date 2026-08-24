"""Walk a QNN *_net.json around its biggest Transposes and print the local dataflow.

analyze_net.py says *how much* layout traffic a graph has. This says *why*: for the
worst offenders it prints the producer, the transpose's perm, and the consumers, so you
can see which op pair the converter is bridging and what axis format each side wants.

  usage: py -3.10 work/device/trace_transposes.py <net.json> [top_n]
"""

import json
import sys
from collections import defaultdict


def dtype_bits(code):
    return int("{:02x}".format(code & 0xFF))


def main():
    path = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    g = json.load(open(path))["graph"]
    nodes, tensors = g["nodes"], g["tensors"]

    producer, consumers = {}, defaultdict(list)
    for name, n in nodes.items():
        for o in n.get("output_names", []):
            producer[o] = name
        for i in n.get("input_names", []):
            consumers[i].append(name)

    def size_mb(t):
        n = 1
        for d in tensors[t]["dims"]:
            n *= d
        return n * dtype_bits(tensors[t]["data_type"]) / 8 / 1e6

    tr = []
    for name, n in nodes.items():
        if n["type"] != "Transpose":
            continue
        out = n["output_names"][0]
        tr.append((size_mb(out), name, n))
    tr.sort(reverse=True)

    # group identical (perm, in-shape, out-shape, producer-type, consumer-types)
    groups = defaultdict(lambda: [0, 0.0])
    for mb, name, n in tr:
        inp = n["input_names"][0]
        out = n["output_names"][0]
        perm = tuple(n.get("tensor_params", {}).get("perm", {})
                     .get("data", n.get("scalar_params", {}).get("perm", [])))
        ptype = nodes[producer[inp]]["type"] if inp in producer else "INPUT"
        ctypes = tuple(sorted({nodes[c]["type"] for c in consumers[out]})) or ("OUTPUT",)
        key = (perm, tuple(tensors[inp]["dims"]), tuple(tensors[out]["dims"]),
               ptype, ctypes)
        groups[key][0] += 1
        groups[key][1] += mb

    print("=" * 88)
    print("Transpose groups in {} (by total MB)".format(path.split("/")[-1]))
    print("=" * 88)
    ranked = sorted(groups.items(), key=lambda kv: -kv[1][1])
    total = sum(v[1] for v in groups.values())
    for (perm, ish, osh, ptype, ctypes), (cnt, mb) in ranked[:top]:
        print("")
        print("  {:8.1f} MB total   x{:<4} {:5.2f} MB each   ({:.0f}% of all transposes)"
              .format(mb, cnt, mb / cnt, 100 * mb / total))
        print("    {} -> Transpose(perm={}) -> {}".format(ptype, list(perm),
                                                          ", ".join(ctypes)))
        print("    {}  ->  {}".format(list(ish), list(osh)))
    print("")
    print("  total transpose bytes: {:.1f} MB across {} nodes".format(total, len(tr)))


if __name__ == "__main__":
    main()
