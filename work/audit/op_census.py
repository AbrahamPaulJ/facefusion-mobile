import sys, collections, onnx
from onnx import shape_inference

def desc(t):
    tt = t.type.tensor_type
    dims = []
    for d in tt.shape.dim:
        dims.append(d.dim_param if d.dim_param else (d.dim_value if d.HasField('dim_value') else '?'))
    return f"{t.name}: {onnx.TensorProto.DataType.Name(tt.elem_type)}{dims}"

for path in sys.argv[1:]:
    m = onnx.load(path)
    g = m.graph
    print("="*72)
    print(path.split('/')[-1], " opset:", [(o.domain or 'ai.onnx', o.version) for o in m.opset_import], " ir:", m.ir_version)
    print("  inputs :", "; ".join(desc(t) for t in g.input))
    print("  outputs:", "; ".join(desc(t) for t in g.output))
    ops = collections.Counter(n.op_type for n in g.node)
    print(f"  nodes: {len(g.node)}  initializers: {len(g.initializer)}  distinct ops: {len(ops)}")
    for op, c in ops.most_common():
        print(f"     {c:5d}  {op}")
