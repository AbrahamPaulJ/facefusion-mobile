"""Make an ONNX graph pnnx-friendly, WITHOUT touching the graph QAIRT ships.

pnnx maps `Gemm` to ncnn's `InnerProduct` only in the `transB=1` form, where the weight is
stored [out, in].  A `transB=0` Gemm -- weight [in, out], the plain `x @ W` -- is left as
`F.linear`, which ncnn has no layer for, so the model loads and computes nothing.

gpen hits this nine times: GPEN's style mapping network is one 4096x512 followed by eight
512x512, all exported transB=0.  Transposing the stored weight and setting transB=1 is the
same arithmetic -- x @ W == x @ (W^T)^T -- so this is exact, not an approximation.

⚠ It is a SEPARATE step, deliberately.  Folding it into `prepare_onnx.py:do_gpen` would
change the graph the shipping QNN context binaries are built from, for a benefit only the
ncnn path collects; the hosted binaries would then differ from a fresh rebuild for no
reason. QNN already maps all 42 Gemms to FullyConnected either way.

    py -3.10 work/ncnn/pnnx_prep.py work/onnx/gpen_bfr_256_sim.onnx work/onnx/gpen_ncnn.onnx
"""
import sys

import numpy
import onnx
from onnx import checker, helper, numpy_helper, shape_inference


def normalise_gemms(m):
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    n_fixed = n_skipped = 0
    for node in g.node:
        if node.op_type != 'Gemm':
            continue
        attrs = {a.name: a for a in node.attribute}
        transB = attrs['transB'].i if 'transB' in attrs else 0
        if transB:
            continue
        w = node.input[1]
        if w not in inits:
            # A computed weight is a different problem and is not silently "fixed".
            n_skipped += 1
            continue
        arr = numpy_helper.to_array(inits[w])
        if arr.ndim != 2:
            n_skipped += 1
            continue
        new = numpy_helper.from_array(numpy.ascontiguousarray(arr.T), w)
        inits[w].CopyFrom(new)
        if 'transB' in attrs:
            attrs['transB'].i = 1
        else:
            node.attribute.append(helper.make_attribute('transB', 1))
        n_fixed += 1
    return n_fixed, n_skipped


def detile_style(m):
    """Delete `Unsqueeze -> Tile -> 14x Gather` and wire the consumers to the source.

    GPEN broadcasts one 512-vector into 14 identical copies and then Gathers one copy per
    generator layer. Every Gather therefore returns the SAME tensor the Unsqueeze was given,
    so the whole fan-out is an identity: 16 nodes that compute nothing.

    pnnx has no ncnn mapping for the Tile (it emits `torch.tile`, which ncnn refuses to
    load), and this removes the need for one rather than working around it.
    """
    g = m.graph
    alive = list(g.node)          # see gemm_to_conv: id() recycling on temporary wrappers
    prod = {o: n for n in alive for o in n.output}
    cons = {}
    for n in alive:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    drop, rewire = set(), {}
    for tile in alive:
        if tile.op_type != 'Tile':
            continue
        un = prod.get(tile.input[0])
        if un is None or un.op_type != 'Unsqueeze':
            continue
        gathers = cons.get(tile.output[0], [])
        if not gathers or any(x.op_type != 'Gather' for x in gathers):
            continue
        # Only safe when the Tile's output feeds nothing BUT those Gathers.
        if len(gathers) != len(cons.get(tile.output[0], [])):
            continue
        src = un.input[0]
        for gth in gathers:
            rewire[gth.output[0]] = src
            drop.add(id(gth))
        drop.add(id(tile)); drop.add(id(un))

    if not rewire:
        return 0
    kept = []
    for n in alive:
        if id(n) in drop:
            continue
        for k, i in enumerate(n.input):
            while i in rewire:
                i = rewire[i]
            n.input[k] = i
        kept.append(n)
    del g.node[:]
    g.node.extend(kept)
    return len(drop)


def gemm_to_conv(m):
    """Every Gemm as a 1x1 Convolution: [N,I] -> [N,I,1,1] -> Conv -> [N,O].

    pnnx folds only SOME Gemms into ncnn's InnerProduct -- GPEN's nine style-mapping
    layers come out as `F.linear`, which ncnn has no layer for, and one of those in a graph
    is a hard `load_param failed`. Which nine is not predictable from the ONNX: they have
    constant weights and the same attributes as the thirty-three that map fine, and
    normalising transB (below) does not move them.

    A 1x1 convolution over a 1x1 spatial map is the same arithmetic and pnnx maps
    Convolution unconditionally, so this sidesteps the question instead of guessing at it.
    """
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    made, new_nodes, new_inits, drop = 0, {}, [], set()
    # `list(g.node)` and NOT `g.node`: iterating the repeated field yields temporary wrapper
    # objects, and once one is collected Python reuses its id() -- so a dict keyed on id()
    # starts matching the wrong nodes. That produced a graph with a duplicated output name
    # and an SSA error from the checker.
    alive = list(g.node)
    for node in alive:
        if node.op_type != 'Gemm' or node.input[1] not in inits:
            continue
        attrs = {a.name: (a.i if a.type == 2 else a.f) for a in node.attribute}
        if attrs.get('transA', 0) or attrs.get('alpha', 1.0) != 1.0 or attrs.get('beta', 1.0) != 1.0:
            continue
        w = numpy_helper.to_array(inits[node.input[1]])
        if w.ndim != 2:
            continue
        # after normalise_gemms every Gemm is transB=1, i.e. weight [out, in]
        if not attrs.get('transB', 0):
            continue
        o, i = w.shape
        b = node.name or ('gemm%d' % made)
        wn = b + '/ff_c1_W'
        new_inits.append(numpy_helper.from_array(
            numpy.ascontiguousarray(w.reshape(o, i, 1, 1)), wn))
        pre, mid = b + '/ff_c1_in', b + '/ff_c1_out'
        sh_in, sh_out = b + '/ff_c1_shin', b + '/ff_c1_shout'
        new_inits.append(numpy_helper.from_array(numpy.array([-1, i, 1, 1], numpy.int64), sh_in))
        new_inits.append(numpy_helper.from_array(numpy.array([-1, o], numpy.int64), sh_out))
        conv_in = [pre, wn] + ([node.input[2]] if len(node.input) > 2 else [])
        new_nodes[id(node)] = [
            helper.make_node('Reshape', [node.input[0], sh_in], [pre], name=b + '/ff_c1_r0'),
            helper.make_node('Conv', conv_in, [mid], name=b + '/ff_c1',
                             kernel_shape=[1, 1], strides=[1, 1], dilations=[1, 1],
                             pads=[0, 0, 0, 0], group=1),
            helper.make_node('Reshape', [mid, sh_out], [node.output[0]], name=b + '/ff_c1_r1'),
        ]
        drop.add(id(node))
        made += 1

    if made:
        kept = []
        for n in alive:
            if id(n) in new_nodes:
                kept.extend(new_nodes[id(n)])
            if id(n) in drop:
                continue
            kept.append(n)
        del g.node[:]
        g.node.extend(kept)
        g.initializer.extend(new_inits)
    return made


def main():
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src)
    fixed, skipped = normalise_gemms(m)
    detiled = detile_style(m)
    conved = gemm_to_conv(m)
    print('  removed %d nodes of identity Unsqueeze/Tile/Gather fan-out' % detiled)
    print('  rewrote %d Gemms as 1x1 Conv' % conved)
    del m.graph.value_info[:]
    m = shape_inference.infer_shapes(m, strict_mode=False)
    checker.check_model(m)
    onnx.save(m, dst)
    print('  transB=0 -> transB=1 on %d Gemms (%d left alone)' % (fixed, skipped))
    print('  wrote %s' % dst)


if __name__ == '__main__':
    main()
