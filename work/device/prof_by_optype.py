"""Aggregate a QNN detailed profile by OP TYPE, joining the op names against net.json.

analyze_prof.py groups by normalised op *name*, which was right for DistilT5's 12
identical encoder layers. The MMDiT has 2446 nodes of 14 distinct types and the question
is different: which *kind* of op is eating the 248 ms? 45% of the nodes are Reshape and
StridedSlice, i.e. pure data movement -- this says what that actually costs.

  usage: py -3.10 work/device/prof_by_optype.py <prof.csv> <net.json>

Reads the SUB-EVENT CYCLES rows of one EXECUTE event (the last complete one, so the
first-inference warm-up is excluded) and buckets them by the node type in net.json.
"""

import json
import sys
from collections import defaultdict


def main():
    csv_path, net_path = sys.argv[1], sys.argv[2]
    g = json.load(open(net_path))["graph"]
    nodes, tensors = g["nodes"], g["tensors"]

    # op name -> type. QNN suffixes the profile identifier with ":OpId_N".
    ntype = {name: n["type"] for name, n in nodes.items()}

    # output tensor name -> node, because the profile often names an op by its output
    for name, n in nodes.items():
        for o in n.get("output_names", []):
            ntype.setdefault(o, n["type"])

    execs, cur, accel = [], None, []
    for line in open(csv_path, encoding="utf-8", errors="replace"):
        p = line.split(",")
        if len(p) < 7:
            continue
        msg, val, unit, lvl, ident = p[1], p[2], p[3], p[5], ",".join(p[6:]).strip()
        if msg != "EXECUTE":
            continue
        try:
            v = int(val)
        except ValueError:
            continue
        if lvl == "ROOT" and "Accelerator (execute) time (cycles)" in ident:
            accel.append(v)
            cur = defaultdict(int)
            execs.append(cur)
        elif lvl == "SUB-EVENT" and unit == "CYCLES" and cur is not None:
            nm = ident.replace(" (cycles)", "").strip()
            cur[nm] += v

    if not execs:
        print("no EXECUTE sub-events found in", csv_path)
        return
    ev = execs[-1]                      # last = warmed up
    total = sum(ev.values())

    by_type = defaultdict(lambda: [0, 0])   # type -> [cycles, count]
    unmatched = defaultdict(int)
    for nm, cy in ev.items():
        base = nm.split(":OpId_")[0]
        t = ntype.get(base)
        if t is None:
            t = ntype.get(base.rstrip("_0123456789").rstrip("_"))
        if t is None:
            unmatched[base] += cy
            t = "<unmatched>"
        by_type[t][0] += cy
        by_type[t][1] += 1

    print("=" * 74)
    print("Per-op-type cycles, {}  (exec {} of {})".format(
        csv_path.split("\\")[-1].split("/")[-1], len(execs), len(execs)))
    print("  accelerator cycles this exec: {:,}".format(accel[-1] if accel else 0))
    print("  summed sub-event cycles:      {:,}  over {} ops".format(total, len(ev)))
    print("=" * 74)
    print("  {:>6}  {:>14} {:>7}   {}".format("ops", "cycles", "share", "op type"))
    print("  " + "-" * 66)
    for t, (cy, c) in sorted(by_type.items(), key=lambda kv: -kv[1][0]):
        print("  {:>6}  {:>14,} {:>6.1f}%   {}".format(c, cy, 100.0 * cy / total, t))

    if unmatched:
        print("")
        print("  top unmatched profile names (op-name -> net.json join missed):")
        for nm, cy in sorted(unmatched.items(), key=lambda kv: -kv[1])[:10]:
            print("    {:>12,}  {}".format(cy, nm[:60]))


if __name__ == "__main__":
    main()
