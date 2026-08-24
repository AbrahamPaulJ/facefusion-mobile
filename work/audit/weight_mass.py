import sys, onnx, numpy as np
from onnx import numpy_helper
for path in sys.argv[1:]:
    m = onnx.load(path); g = m.graph
    tot = 0; rows = []
    for init in g.initializer:
        n = 1
        for d in init.dims: n *= d
        tot += n
        rows.append((n, init.name, list(init.dims)))
    rows.sort(reverse=True)
    print("="*72); print(path.split('/')[-1], f"  total params {tot/1e6:.1f}M")
    for n, name, dims in rows[:12]:
        print(f"   {n/1e6:8.2f}M  {str(dims):24s} {name}")
