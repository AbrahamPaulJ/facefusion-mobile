import sys, onnx, collections
from onnx import shape_inference

def run(path):
    m = onnx.load(path)
    for i in m.graph.input:
        d = i.type.tensor_type.shape.dim[0]
        if d.dim_param or d.dim_value == 0:
            d.ClearField('dim_param'); d.dim_value = 1
    m = shape_inference.infer_shapes(m, strict_mode=False)
    sh = {}
    for vi in list(m.graph.value_info)+list(m.graph.input)+list(m.graph.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        # a resolved-to-0 batch is an artefact of Reshape(-1); treat as 1
        if dims and dims[0] == 0: dims[0] = 1
        if all(isinstance(x,int) and x>0 for x in dims): sh[vi.name] = dims
    init = {i.name: list(i.dims) for i in m.graph.initializer}
    per = collections.Counter(); rows=[]; unresolved=[]
    for n in m.graph.node:
        mac = 0
        if n.op_type in ('Conv','ConvTranspose'):
            w = init.get(n.input[1]); o = sh.get(n.output[0]); i0 = sh.get(n.input[0])
            if not (w and o and i0): unresolved.append((n.op_type, n.name)); continue
            k = 1
            for x in w[2:]: k *= x
            # Spatial extent is EVERY dim after N,C -- not a fixed o[2]*o[3].  Hardcoding two
            # of them silently drops the last axis of a 3D conv, so a 5D graph comes out short
            # by that axis: live_portrait_generator read 457 GMAC against a true 617, a 26%
            # under-count that looked entirely plausible.  Nothing in the shipped set is 5D,
            # which is why it went unnoticed.
            sp, ref = 1, (o if n.op_type == 'Conv' else i0)
            for x in ref[2:]: sp *= x
            mac = ref[1]*sp*w[1]*k
        elif n.op_type in ('Gemm','MatMul'):
            # input[1] is an INITIALIZER for a weight matmul and an ACTIVATION for the two
            # matmuls inside attention (q@k^T, attn@v).  Looking only in `init` silently
            # dropped both of those from the total: nsfw_2 came out 3.15 GMAC against a
            # true 4.68, a 33% under-count that showed up only as `[unresolved: 24]`.
            # 24 == 12 transformer blocks x 2.  Any attention graph hit this.
            w = init.get(n.input[1]) or sh.get(n.input[1]); o = sh.get(n.output[0])
            if not (w and o): unresolved.append((n.op_type, n.name)); continue
            mac = 1
            for x in o: mac *= x
            mac *= w[1] if n.op_type=='Gemm' else w[-2]
        if mac: per[n.op_type] += mac; rows.append((mac, n.op_type, n.name or n.output[0], sh.get(n.output[0])))
    tot = sum(per.values())
    print(f"{path.split('/')[-1]:26s} {tot/1e9:8.2f} GMAC   " +
          "  ".join(f"{op} {v/1e9:.2f}" for op,v in per.most_common()) +
          (f"   [unresolved: {len(unresolved)}]" if unresolved else ""))
    return tot

for p in sys.argv[1:]: run(p)
