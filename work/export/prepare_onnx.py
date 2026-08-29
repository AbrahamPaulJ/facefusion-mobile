"""ONNX surgery for the five MVP graphs, before QNN conversion.

Each fix is here because something measured said it was needed -- see docs/model-audit.md.
Run:  py -3.10 work/export/prepare_onnx.py <model> [...]   (or `all`)
Outputs land in work/onnx/.
"""
import argparse
import os
import sys

import numpy
import onnx
from onnx import checker, helper, numpy_helper, shape_inference

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS = os.path.join(ROOT, 'models')
OUT = os.path.join(ROOT, 'onnx')


def load(name):
	return onnx.load(os.path.join(MODELS, name + '.onnx'))


def save(model, name):
	os.makedirs(OUT, exist_ok=True)
	path = os.path.join(OUT, name + '.onnx')
	# trap #21: one external-data file per tensor is a disaster over the 9p mount.
	# Keep every graph single-file; none of these is near the 2 GB protobuf limit.
	onnx.save(model, path)
	size = os.path.getsize(path) / 1e6
	print('  wrote %-28s %8.1f MB' % (name + '.onnx', size))
	return path


def pin_batch(model, value=1):
	changed = []
	for t in list(model.graph.input) + list(model.graph.output):
		dim = t.type.tensor_type.shape.dim
		if not len(dim):
			continue
		d = dim[0]
		if d.dim_param or d.dim_value == 0:
			d.ClearField('dim_param')
			d.dim_value = value
			changed.append(t.name)
	return changed


def strip_value_info(model):
	"""Stale value_info fights re-inference after a graph edit."""
	del model.graph.value_info[:]
	return model


# ------------------------------------------------------------------ arcface

def do_arcface():
	"""Only fix needed: the batch dim is 'None'. 6 op types, 130 nodes, nothing else to do."""
	m = load('arcface_w600k_r50')
	changed = pin_batch(m)
	print('  pinned batch on:', changed)
	m = shape_inference.infer_shapes(strip_value_info(m), strict_mode=False)
	checker.check_model(m)
	return save(m, 'arcface_w600k_r50_b1')


# ------------------------------------------------------------------ 2dfan4

def do_2dfan4():
	"""Cut at `heatmaps`; the twelve decode nodes run on the host (docs/model-audit.md).

	Removes ArgMax/Mod/Greater/Not/Expand/Tile/ReduceSum/Sqrt/Clip/Div from the NPU graph.
	The host decode is asserted equivalent in run_reference.verify_decode() -- 9.1e-06
	heatmap px on a real crop.
	"""
	m = load('2dfan4')
	keep = 'heatmaps'
	vi = {o.name: o for o in m.graph.output}
	if keep not in vi:
		sys.exit('2dfan4 has no `%s` output' % keep)

	del m.graph.output[:]
	m.graph.output.extend([vi[keep]])

	m = onnx.utils.Extractor(strip_value_info(m)).extract_model(
		[i.name for i in m.graph.input], [keep])
	m = shape_inference.infer_shapes(m, strict_mode=False)
	checker.check_model(m)
	ops = {}
	for n in m.graph.node:
		ops[n.op_type] = ops.get(n.op_type, 0) + 1
	print('  kept %d nodes, op types: %s' % (len(m.graph.node), sorted(ops)))
	return save(m, '2dfan4_heatmaps')


# ------------------------------------------------------------------ yoloface

def do_yoloface(cut_head=False):
	"""Convert whole by default. The decode head is ~0 FLOPs and every op is supported;
	only cut it if analyze_net.py reports bad transpose traffic (trap #23)."""
	m = load('yoloface_8n')
	pin_batch(m)
	m = shape_inference.infer_shapes(strip_value_info(m), strict_mode=False)
	checker.check_model(m)
	return save(m, 'yoloface_8n_b1')


# ------------------------------------------------------------------ hyperswap

def drop_broadcast_expands(model):
	"""Delete `Expand` nodes that only broadcast [1,C,1,1] -> [1,C,H,W] into a Mul/Add.

	QNN rejects Expand outright:
	    RuntimeError: validateQnnOpConfig: Failed QNN validation for
	    /model/generator/generator/layers.0/primary_layers.0/Expand
	and it is redundant anyway -- ElementWise ops broadcast natively.  hyperswap has 42 of
	them, all modulation gamma/beta tiled across space, and the largest materialises a
	[1,64,256,256] tensor (16 MB fp32) purely to be multiplied.  Deleting them removes the
	validation failure AND real bandwidth, which matters on a bandwidth-bound part
	(trap #41).  Same family as trap #19: remove structure that exists only to satisfy a
	shape.
	"""
	g = model.graph
	shaped = shape_inference.infer_shapes(model, strict_mode=False)
	sh = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
		  for vi in list(shaped.graph.value_info) + list(shaped.graph.input) +
					list(shaped.graph.output)}
	consumers = {}
	for n in g.node:
		for i in n.input:
			consumers.setdefault(i, []).append(n)

	BROADCASTERS = {'Mul', 'Add', 'Sub', 'Div'}
	removed = 0
	for node in list(g.node):
		if node.op_type != 'Expand':
			continue
		src, dst = sh.get(node.input[0]), sh.get(node.output[0])
		users = consumers.get(node.output[0], [])
		if not src or not dst or len(src) != len(dst):
			continue
		# only safe when every expanded axis starts at 1, and every consumer broadcasts
		if any(s != d and s != 1 for s, d in zip(src, dst)):
			continue
		if not users or any(u.op_type not in BROADCASTERS for u in users):
			continue
		for u in users:
			for k, i in enumerate(u.input):
				if i == node.output[0]:
					u.input[k] = node.input[0]
		g.node.remove(node)
		removed += 1
	return removed


def do_hyperswap():
	"""Drop the unused `mask` output, then delete the broadcast-only Expands.

	`mask` is free to remove -- face_swapper/core.py:657 reads output 0 only, and the
	masker feeds `output` too, so GMAC is unchanged at 31.93.
	"""
	m = load('hyperswap_1a_256')
	names = [o.name for o in m.graph.output]
	print('  outputs were:', names)
	m = onnx.utils.Extractor(strip_value_info(m)).extract_model(
		[i.name for i in m.graph.input], [names[0]])

	removed = drop_broadcast_expands(strip_value_info(m))
	print('  deleted %d broadcast-only Expand nodes (trap #19 / QNN validation)' % removed)

	m = shape_inference.infer_shapes(strip_value_info(m), strict_mode=False)
	checker.check_model(m)
	ops = {}
	for n in m.graph.node:
		ops[n.op_type] = ops.get(n.op_type, 0) + 1
	print('  now: %d nodes; Shape=%d Expand=%d Reshape=%d' %
		  (len(m.graph.node), ops.get('Shape', 0), ops.get('Expand', 0), ops.get('Reshape', 0)))
	return save(m, 'hyperswap_1a_256_nomask')


# ------------------------------------------------------------------ inswapper

def do_inswapper():
	"""trap #31: split each modulation Gemm so gamma and beta get SEPARATE encodings.

	source[1,512] -> Gemm -> [1,2048] -> Unsqueeze x2 -> Slice[0:1024] / Slice[1024:2048]

	One tensor, one encoding, two semantically different halves multiplying straight into
	the activation stream.  Twelve occurrences.  Splitting the Gemm into two Gemms is
	identical arithmetic and gives each half its own range.
	"""
	m = load('inswapper_128')
	g = m.graph
	init = {i.name: i for i in g.initializer}
	prod = {o: n for n in g.node for o in n.output}
	consumers = {}
	for n in g.node:
		for i in n.input:
			consumers.setdefault(i, []).append(n)

	def const_val(name):
		if name in init:
			return numpy_helper.to_array(init[name])
		n = prod.get(name)
		if n is not None and n.op_type == 'Constant':
			return numpy_helper.to_array(n.attribute[0].t)
		return None

	split_count = 0
	new_nodes = []
	drop = set()
	new_inits = []

	for node in g.node:
		if node.op_type != 'Gemm':
			continue
		# Gemm -> Unsqueeze -> Unsqueeze -> {Slice, Slice}
		u1 = [c for c in consumers.get(node.output[0], []) if c.op_type == 'Unsqueeze']
		if len(u1) != 1:
			continue
		u2 = [c for c in consumers.get(u1[0].output[0], []) if c.op_type == 'Unsqueeze']
		if len(u2) != 1:
			continue
		slices = [c for c in consumers.get(u2[0].output[0], []) if c.op_type == 'Slice']
		if len(slices) != 2:
			continue

		W = const_val(node.input[1])
		B = const_val(node.input[2]) if len(node.input) > 2 else None
		if W is None:
			continue
		transB = any(a.name == 'transB' and a.i for a in node.attribute)
		# rows of the output = W.shape[0] if transB else W.shape[1]
		out_dim = W.shape[0] if transB else W.shape[1]

		bounds = []
		for s in slices:
			st = const_val(s.input[1])
			en = const_val(s.input[2])
			bounds.append((int(st.ravel()[0]), int(min(en.ravel()[0], out_dim))))
		if sorted(bounds) != [(0, out_dim // 2), (out_dim // 2, out_dim)]:
			continue

		for k, (s, (lo, hi)) in enumerate(zip(slices, bounds)):
			Wk = W[lo:hi] if transB else W[:, lo:hi]
			Bk = B[lo:hi] if B is not None else None
			wn = node.name + '_split%d_W' % k
			new_inits.append(numpy_helper.from_array(numpy.ascontiguousarray(Wk), wn))
			ins = [node.input[0], wn]
			if Bk is not None:
				bn = node.name + '_split%d_B' % k
				new_inits.append(numpy_helper.from_array(numpy.ascontiguousarray(Bk), bn))
				ins.append(bn)
			gemm_out = node.name + '_split%d_out' % k
			new_nodes.append(helper.make_node(
				'Gemm', ins, [gemm_out], name=node.name + '_split%d' % k,
				**{a.name: helper.get_attribute_value(a) for a in node.attribute}))
			# rebuild the two Unsqueezes so the consumer still sees [1, C, 1, 1]
			a1 = node.name + '_split%d_u1' % k
			new_nodes.append(helper.make_node('Unsqueeze', [gemm_out] + list(u1[0].input[1:]),
											  [a1], name=node.name + '_split%du1' % k,
											  **{a.name: helper.get_attribute_value(a)
												 for a in u1[0].attribute}))
			new_nodes.append(helper.make_node('Unsqueeze', [a1] + list(u2[0].input[1:]),
											  [s.output[0]], name=node.name + '_split%du2' % k,
											  **{a.name: helper.get_attribute_value(a)
												 for a in u2[0].attribute}))
			drop.add(id(s))
		drop.update({id(node), id(u1[0]), id(u2[0])})
		split_count += 1

	if not split_count:
		sys.exit('  inswapper: found no modulation Gemms to split -- graph shape changed?')

	kept = [n for n in g.node if id(n) not in drop]
	del g.node[:]
	g.node.extend(kept + new_nodes)
	g.initializer.extend(new_inits)

	# Appending the replacement nodes leaves the graph out of topological order, so it
	# must be sorted BEFORE any checker call -- checking first just reports that.
	m = _toposort(strip_value_info(m))
	m = shape_inference.infer_shapes(m, strict_mode=False)
	checker.check_model(m)
	print('  split %d modulation Gemms into gamma/beta pairs (trap #31)' % split_count)
	return save(m, 'inswapper_128_split')


def _toposort(model):
	g = model.graph
	ready = {i.name for i in g.initializer} | {i.name for i in g.input}
	pending = list(g.node)
	ordered = []
	while pending:
		progressed = False
		still = []
		for n in pending:
			if all((not i) or i in ready for i in n.input):
				ordered.append(n)
				ready.update(n.output)
				progressed = True
			else:
				still.append(n)
		pending = still
		if not progressed:
			sys.exit('  toposort stuck on %d nodes' % len(pending))
	del g.node[:]
	g.node.extend(ordered)
	return model


def demote_fp16_to_fp32(model):
	"""Rewrite an fp16 graph as fp32: initialisers, Constant nodes, value_info, and the
	Cast nodes that wrap the fp16 body in an fp32 I/O shell.

	`hyperswap_1a_256` is natively an **fp16 model** -- 306 fp16 initialisers, 482 fp16
	tensors, `Cast(source)->fp16`, `Cast(target)->fp16`, body in fp16, `Cast(...)->fp32` at
	the end.  Two consequences, and neither was obvious until the datatype codes were read:

	1. **QAIRT 2.28 cannot convert it.**  Its FULLY_CONNECTED validator rejects the first
	   Gemm with `Tensor 5 and 134 have mismatching datatypes. 0x216 != 0x232`, which is
	   QNN_DATATYPE_FLOAT_16 against QNN_DATATYPE_FLOAT_32.  A five-variant sweep of
	   `--preserve_io` and per-channel options changed nothing, because none of them touches
	   the graph's own precision (docs/roadmap.md 1.7).
	2. **The fp16 context stamp may not be purely a 2.49 artifact for this graph.**  The
	   model really does ask for fp16.  Worth testing whether a 2.49 build of the demoted
	   graph still declares the requirement.

	We quantise to W8A16 anyway, so every one of these tensors becomes fixed-point in the
	end -- the fp16 storage buys nothing downstream and costs precision on the way in.
	"""
	g = model.graph
	FP16, FP32 = 10, 1
	stats = {'init': 0, 'const': 0, 'value_info': 0, 'cast': 0}

	for t in g.initializer:
		if t.data_type == FP16:
			arr = numpy_helper.to_array(t).astype(numpy.float32)
			t.CopyFrom(numpy_helper.from_array(arr, t.name))
			stats['init'] += 1

	for node in g.node:
		for attr in node.attribute:
			if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == FP16:
				arr = numpy_helper.to_array(attr.t).astype(numpy.float32)
				attr.t.CopyFrom(numpy_helper.from_array(arr, attr.t.name))
				stats['const'] += 1

	for vi in list(g.value_info) + list(g.input) + list(g.output):
		if vi.type.tensor_type.elem_type == FP16:
			vi.type.tensor_type.elem_type = FP32
			stats['value_info'] += 1

	# The Cast shell is now identity on both sides -- rewire consumers past it and delete.
	# Casts to anything else (int32 indices, bool masks) are left strictly alone.
	produced = {n.output[0]: n for n in g.node if n.output}
	graph_outputs = {o.name for o in g.output}
	for node in list(g.node):
		if node.op_type != 'Cast':
			continue
		to = next((a.i for a in node.attribute if a.name == 'to'), None)
		if to not in (FP16, FP32):
			continue
		src, dst = node.input[0], node.output[0]
		# a Cast feeding a graph output must keep that output's name, so rename instead
		if dst in graph_outputs:
			if src in produced:
				producer = produced[src]
				producer.output[0] = dst
				for other in g.node:
					for k, i in enumerate(other.input):
						if i == src:
							other.input[k] = dst
			else:
				continue
		else:
			for other in g.node:
				for k, i in enumerate(other.input):
					if i == dst:
						other.input[k] = src
		g.node.remove(node)
		stats['cast'] += 1

	return stats


def do_hyperswap_fp32():
	"""`do_hyperswap`, plus the fp16 -> fp32 demotion.  THIS IS THE SHIPPING GRAPH.

	`hyperswap` is the only natively-fp16 model here (306 fp16 initialisers).  Demoting the
	weights to fp32 before quantisation was promoted on 2026-08-24: deploy SNR and
	end-to-end PSNR are unchanged to the decimal (30.87 dB / 47.87 / 41.68 dB), and it is
	the only form QAIRT 2.28 will convert at all.  It is NOT faster -- session 3 measured
	-3.5 % and a counterbalanced re-measurement found 0.02 ms (docs/traps.md #16).

	Still written as `hyperswap_1a_256_fp32.onnx` beside `..._sim.onnx` rather than over
	it, so the pre-promotion control stays buildable for A/Bs.
	"""
	m = load('hyperswap_1a_256')
	names = [o.name for o in m.graph.output]
	m = onnx.utils.Extractor(strip_value_info(m)).extract_model(
		[i.name for i in m.graph.input], [names[0]])

	removed = drop_broadcast_expands(strip_value_info(m))
	print('  deleted %d broadcast-only Expand nodes' % removed)

	stats = demote_fp16_to_fp32(strip_value_info(m))
	print('  fp16 -> fp32: %d initialisers, %d constant attrs, %d value_info, %d Cast removed'
		  % (stats['init'], stats['const'], stats['value_info'], stats['cast']))

	m = shape_inference.infer_shapes(strip_value_info(m), strict_mode=False)
	checker.check_model(m)
	left = sum(1 for t in m.graph.initializer if t.data_type == 10)
	print('  fp16 initialisers remaining: %d (want 0)' % left)
	return save(m, 'hyperswap_1a_256_fp32')


def add_explicit_slice_steps(model):
	"""Give every `Slice` an explicit all-ones `steps` input.

	ONNX defaults `steps` to 1 when the input is omitted.  **QAIRT 2.28 defaults it to 0**,
	and then rejects its own output:

	    Failed QNN validation for /model.22/Slice
	    Validating tensor 661 and 659 have the same Datatype.
	    Stride is zero at index 0.

	on `QNN_OP_STRIDED_SLICE`.  2.49 fills the same gap with 1 and converts happily.
	`yoloface_8n` has four Slices in the DFL decode head; the two that carry a `steps`
	input already convert, and the two that omit it are exactly the two that fail --
	which is what identifies the default, rather than the op, as the problem.

	Writing the default out explicitly is semantics-preserving under the ONNX spec, so this
	is safe for 2.49 too.  It is also far cheaper than the alternative on the table
	(cutting the whole decode head to CPU, docs/roadmap.md 6).
	"""
	g = model.graph
	# Before simplification these operands are `Constant` NODES, not initialisers, so a
	# lookup that only searches g.initializer finds nothing and silently skips every Slice.
	const = {t.name: numpy_helper.to_array(t) for t in g.initializer}
	for cn in g.node:
		if cn.op_type == 'Constant' and cn.output:
			t = next((a.t for a in cn.attribute if a.name == 'value'), None)
			if t is not None:
				const[cn.output[0]] = numpy_helper.to_array(t)

	added = 0
	for node in g.node:
		if node.op_type != 'Slice' or len(node.input) >= 5:
			continue
		if len(node.input) < 3:
			continue  # opset-9 Slice keeps starts/ends in attributes; not our case
		# starts, ends and axes are all the same length -- take whichever is resolvable
		sizes = [const[i].size for i in node.input[1:4] if i in const]
		if not sizes:
			print('    ! %s: cannot size steps, left alone' % node.name)
			continue
		n = int(sizes[0])

		while len(node.input) < 4:
			node.input.append('')  # axes is optional and must be filled before steps
		name = node.name.replace('/', '_').lstrip('_') + '_steps'
		g.initializer.append(
			numpy_helper.from_array(numpy.ones(n, dtype=numpy.int64), name))
		node.input.append(name)
		added += 1
	return added


def expand_slices_to_full_rank(model):
	"""Rewrite every `Slice` so starts/ends/axes/steps name EVERY axis explicitly.

	`add_explicit_slice_steps` was not enough: supplying `steps=[1]` for the sliced axis
	still leaves 2.28 reporting `Stride is zero at index 0` on `QNN_OP_STRIDED_SLICE`.
	Index 0 is not the sliced axis here -- `/model.22/Slice` cuts axis 1 -- which says 2.28
	fills the stride for the axes the Slice does NOT name with 0 rather than 1.  ONNX says
	unnamed axes are untouched; 2.28's STRIDED_SLICE ranges tuple wants a real stride for
	all of them.

	So leave nothing to infer: emit `[0, dim, 1]` for every axis, overwritten by the real
	slice on the axes the node actually names.  Semantics-preserving, and 2.49 is
	unaffected (verified: it converts both forms).
	"""
	g = model.graph
	shaped = shape_inference.infer_shapes(model, strict_mode=False)
	shapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
			  for vi in list(shaped.graph.value_info) + list(shaped.graph.input) +
						list(shaped.graph.output)}
	const = {t.name: numpy_helper.to_array(t) for t in g.initializer}
	for cn in g.node:
		if cn.op_type == 'Constant' and cn.output:
			t = next((a.t for a in cn.attribute if a.name == 'value'), None)
			if t is not None:
				const[cn.output[0]] = numpy_helper.to_array(t)

	rewritten = 0
	for node in g.node:
		if node.op_type != 'Slice' or len(node.input) < 3:
			continue
		shape = shapes.get(node.input[0])
		if not shape:
			print('    ! %s: input shape unknown, left alone' % node.name)
			continue
		rank = len(shape)

		def get(idx):
			return const.get(node.input[idx]) if len(node.input) > idx else None

		starts, ends, axes, steps = get(1), get(2), get(3), get(4)
		if starts is None or ends is None:
			print('    ! %s: starts/ends not constant, left alone' % node.name)
			continue
		if axes is None:
			axes = numpy.arange(len(starts))
		if steps is None:
			steps = numpy.ones(len(starts), dtype=numpy.int64)

		full_s = numpy.zeros(rank, dtype=numpy.int64)
		full_e = numpy.array(shape, dtype=numpy.int64)
		full_t = numpy.ones(rank, dtype=numpy.int64)
		for k, ax in enumerate(numpy.asarray(axes).tolist()):
			ax = ax if ax >= 0 else ax + rank
			full_s[ax] = int(starts[k])
			full_e[ax] = min(int(ends[k]), shape[ax])   # ONNX clamps; INT64_MAX is common
			full_t[ax] = int(steps[k])

		base = node.name.replace('/', '_').lstrip('_')
		names = []
		for suffix, arr in (('starts', full_s), ('ends', full_e),
							('axes', numpy.arange(rank, dtype=numpy.int64)), ('steps', full_t)):
			nm = '%s_full_%s' % (base, suffix)
			g.initializer.append(numpy_helper.from_array(arr, nm))
			names.append(nm)

		del node.input[1:]
		node.input.extend(names)
		rewritten += 1
	return rewritten


def do_yoloface_slicefix():
	"""`do_yoloface`, plus explicit Slice steps.  A SEPARATE artifact on purpose.

	The shipping 2.49 detector is verified at 43.58 dB and its geometry is the most
	trap-prone thing in the project (trap #9 -- the best SNR here hid the worst geometry).
	This writes `yoloface_8n_slicefix.onnx` beside it so the change can be A/B'd against a
	known-good control, and judged on DETECTION AGREEMENT rather than tensor SNR.
	"""
	# Operate on the SIMPLIFIED graph, not the raw one.  The two Slices that matter carry
	# computed starts/ends before simplification and constant ones after, so the raw model
	# leaves exactly the two nodes we need untouched.  This is also the graph convert.sh
	# consumes, which makes it a true A/B against the shipping build.
	src = os.path.join(OUT, 'yoloface_8n_b1_sim.onnx')
	if not os.path.exists(src):
		sys.exit('  need %s -- run `prepare_onnx.py yoloface` and simplify first' % src)
	m = onnx.load(src)
	rewritten = expand_slices_to_full_rank(m)
	print('  expanded %d Slice nodes to full rank' % rewritten)
	m = shape_inference.infer_shapes(strip_value_info(m), strict_mode=False)
	checker.check_model(m)
	return save(m, 'yoloface_8n_slicefix')


# ---------------------------------------------------------------------- nsfw

def do_nsfw():
	"""`nsfw_2`, the content gate.  The whole surgery is constant folding.

	It is a ViT-Small: 12 blocks, one patch-embedding Conv, one Gemm head, output [1, 2].
	The raw graph carries the usual ViT cls-token scaffolding -- Shape -> Slice -> Equal ->
	Where -> ConstantOfShape -> Expand -- which reads the batch dimension at runtime.  The
	input is already static at [1, 3, 384, 384], so all of it folds:

	    636 nodes / 24 op types  ->  380 nodes / 15 op types

	and with it goes the `Expand` that trap #4 says QNN rejects, plus every `Slice`, which
	is the op that blocks the detector under 2.28.  Nothing has to be rewritten by hand.

	⚠ Unlike the other five, this task runs onnxsim ITSELF and writes `_sim` directly.  For
	the others the simplify pass is a separate manual step; folding is the entire surgery
	here, so splitting it in two would leave a `nsfw_2.onnx` in work/onnx/ that nothing
	consumes and that no verification covers.

	All 152 initialisers are already fp32 -- checked, because this is exactly what
	`hyperswap` hid (docs/roadmap.md 1.7).
	"""
	import onnxsim

	m = load('nsfw_2')
	fp16 = sum(1 for t in m.graph.initializer if t.data_type == 10)
	print('  fp16 initialisers: %d (want 0 -- see hyperswap)' % fp16)

	before = len(m.graph.node)
	s, ok = onnxsim.simplify(m, overwrite_input_shapes={'input': [1, 3, 384, 384]})
	if not ok:
		sys.exit('  onnxsim reported failure')
	print('  simplified %d -> %d nodes' % (before, len(s.graph.node)))

	for op in ('Expand', 'Slice', 'Shape', 'Where', 'ConstantOfShape', 'Cast'):
		left = sum(1 for n in s.graph.node if n.op_type == op)
		if left:
			print('  WARNING: %d %s survived the fold' % (left, op))

	s = shape_inference.infer_shapes(strip_value_info(s), strict_mode=False)
	checker.check_model(s)
	return save(s, 'nsfw_2_sim')


# ---------------------------------------------------------------------- gpen

def do_gpen():
	"""`gpen_bfr_256`, the face enhancer: make every modulated conv kernel STATIC.

	GPEN is a StyleGAN2 generator, and StyleGAN2 modulates its convolution weights with the
	style vector at RUNTIME.  As exported, 20 of the 45 convs take a computed kernel:

	    style -> Mul(W) -> Pow/ReduceSum/Sqrt/Div (demodulate) -> Reshape -> Conv

	HTP maps Conv only when the kernel is a static parameter, and those 20 carry 7.44 of
	the graph's 8.57 GMAC -- 87% of the model would not lower.  The chain also builds 97
	rank>=5 activations (12 at rank 6, e.g. [1,256,256,3,3]), where the converter's
	elementwise broadcast path is `max_rank <= 5` and the arch linter's MAX_RANK is 4.

	The refactor is exact algebra, not an approximation.  Modulation scales per INPUT
	channel, demodulation per OUTPUT channel, and batch is 1 (hence group 1), so

	    Conv(x, W[o,i,..] * s[i])[o] * d[o]  ==  Conv(x * s[i], W)[o] * d[o]

	and the demodulator itself drops out of rank 5, because

	    d[o] = 1/sqrt( sum_{i,kh,kw} (W[o,i,kh,kw] * s[i])^2 + eps )
	         = 1/sqrt( sum_i s[i]^2 * A[o,i] + eps ),  A[o,i] = sum_{kh,kw} W[o,i,kh,kw]^2

	A is CONSTANT, so the whole demodulator becomes one [1,I]x[I,O] MatMul.  Every kernel
	ends up static, every rank-5/6 tensor disappears, and the Pow/ReduceSum over a 5-D
	weight goes with it.

	The 20 split cleanly and the code asserts it: 13 demodulated convs (7 Conv 3x3 + 6
	ConvTranspose 3x3 s2) that carry NO bias, and 7 `to_rgb` 1x1 convs that carry a bias
	and are modulated but NOT demodulated.  That matters -- a bias must be added AFTER the
	demodulation scale, never folded through it, so a demodulated conv that also had a bias
	would need a code path this does not have.  It bails instead of guessing.

	NOT bit-exact: the two sums reassociate.  verify_surgery.py measures the cost.
	"""
	import onnxsim

	m = load('gpen_bfr_256')

	# Fold FIRST.  The export declares 128 initialisers as graph inputs as well (the old
	# PyTorch pattern onnxsim warns about), and every blur kernel reaches its Conv through a
	# Reshape, which makes 18 more convs look dynamic than are.  Folding takes dynamic
	# kernels 38 -> 20 and leaves exactly the modulated ones, so the matcher sees a clean
	# pattern instead of having to tell the two apart.
	s, ok = onnxsim.simplify(m, overwrite_input_shapes={'input': [1, 3, 256, 256]})
	if not ok:
		sys.exit('  onnxsim reported failure (pre-pass)')
	m = s
	g = m.graph
	init = {t.name: t for t in g.initializer}
	prod = {o: n for n in g.node for o in n.output}

	def const_val(name):
		if name in init:
			return numpy_helper.to_array(init[name])
		n = prod.get(name)
		if n is not None and n.op_type == 'Constant':
			return numpy_helper.to_array(n.attribute[0].t)
		return None

	def expect(cond, node, what):
		if not cond:
			sys.exit('  %s: %s' % (node.name, what))

	new_inits = []
	out_nodes = []
	n_demod = n_plain = 0

	def shape_init(dims, tag):
		name = tag
		new_inits.append(numpy_helper.from_array(
			numpy.asarray(dims, dtype=numpy.int64), name))
		return name

	for node in g.node:
		if node.op_type not in ('Conv', 'ConvTranspose') or node.input[1] in init:
			out_nodes.append(node)
			continue

		b = node.name
		# ---- walk the kernel chain: Reshape [ <- Transpose ] <- Mul
		rs = prod.get(node.input[1])
		expect(rs is not None and rs.op_type == 'Reshape', node, 'kernel is not a Reshape')
		src = prod.get(rs.input[0])
		transposed = False
		if src is not None and src.op_type == 'Transpose':
			perm = next((list(a.ints) for a in src.attribute if a.name == 'perm'), None)
			# [1,O,I,kh,kw] -> [1,I,O,kh,kw], i.e. ONNX's ConvTranspose kernel layout
			expect(perm == [0, 2, 1, 3, 4], node, 'unexpected kernel Transpose perm %s' % perm)
			transposed = True
			src = prod.get(src.input[0])
		expect(src is not None and src.op_type == 'Mul', node, 'kernel chain does not end in Mul')

		def modulate_of(mul):
			"""(base weight, style tensor) if `mul` is the modulating Mul, else (None, None)."""
			for a, other in ((mul.input[0], mul.input[1]), (mul.input[1], mul.input[0])):
				w = const_val(a)
				if w is not None and w.ndim == 5:
					return w, other
			return None, None

		W5, style_r = modulate_of(src)
		demod_t = None
		if W5 is None:
			# `src` is the DEMODULATING Mul: one operand is the modulating Mul, the other
			# the reshaped 1/sqrt(...) vector.
			mod = None
			for a, other in ((src.input[0], src.input[1]), (src.input[1], src.input[0])):
				n = prod.get(a)
				if n is not None and n.op_type == 'Mul':
					mod, demod_t = n, other
					break
			expect(mod is not None, node, 'no modulating Mul beneath the demodulating one')
			W5, style_r = modulate_of(mod)
			expect(W5 is not None, node, 'modulating Mul carries no 5-D weight')

		sr = prod.get(style_r)
		expect(sr is not None and sr.op_type == 'Reshape', node, 'style is not reshaped')
		style = sr.input[0]                       # [1, I], straight off the modulation Gemm

		_, O, I, kh, kw = W5.shape
		attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
		expect(attrs.get('group', 1) == 1, node, 'group != 1 breaks the per-channel identity')

		# ---- static kernel
		W4 = W5[0]                                            # [O, I, kh, kw]
		if transposed:
			W4 = W4.transpose(1, 0, 2, 3)                     # [I, O, kh, kw]
		wn = b + '/ff_W'
		new_inits.append(numpy_helper.from_array(numpy.ascontiguousarray(W4), wn))

		# ---- fold the style onto the ACTIVATION instead of the kernel
		s4 = b + '/ff_style4'
		out_nodes.append(helper.make_node(
			'Reshape', [style, shape_init([1, I, 1, 1], b + '/ff_style_shape')], [s4],
			name=b + '/ff_style_reshape'))
		xs = b + '/ff_x'
		out_nodes.append(helper.make_node('Mul', [node.input[0], s4], [xs], name=b + '/ff_xscale'))

		has_bias = len(node.input) > 2
		conv_ins = [xs, wn]
		if demod_t is None:
			# No demodulation: the bias rides along inside the conv, exactly as before.
			if has_bias:
				conv_ins.append(node.input[2])
			out_nodes.append(helper.make_node(
				node.op_type, conv_ins, [node.output[0]], name=b, **attrs))
			n_plain += 1
			continue

		# A demodulated conv must not carry a bias: the scale below would multiply it.
		expect(not has_bias, node, 'demodulated conv has a bias; would need folding after the scale')

		# ---- rebuild the demodulator at rank 2
		dr = prod.get(demod_t)
		expect(dr is not None and dr.op_type == 'Reshape', node, 'demod is not reshaped')
		dv = prod.get(dr.input[0])
		expect(dv is not None and dv.op_type == 'Div', node, 'demod does not come from a Div')
		one = const_val(dv.input[0])
		sq = prod.get(dv.input[1])
		expect(sq is not None and sq.op_type == 'Sqrt', node, 'demod divisor is not a Sqrt')
		ad = prod.get(sq.input[0])
		expect(ad is not None and ad.op_type == 'Add', node, 'no epsilon Add under the Sqrt')
		eps = const_val(ad.input[1])
		expect(one is not None and eps is not None, node, 'demod eps/numerator are not constant')

		# A[o,i] = sum over the kernel window of W^2, accumulated in float64 -- this is the
		# one place the refactor could lose precision that the original did not, and it is
		# free to do it wide.
		A = (W5[0].astype(numpy.float64) ** 2).sum(axis=(2, 3))          # [O, I]
		an = b + '/ff_A'
		new_inits.append(numpy_helper.from_array(
			numpy.ascontiguousarray(A.T.astype(numpy.float32)), an))     # [I, O]
		# eps as the Gemm's BIAS vector, not a separate Add.
		#
		# `MatMul -> ElementwiseSum` is the exact pattern the 2.49 converter's squash_sum
		# folds into a FullyConnected's bias, and it reads input_names[2] unconditionally
		# -- but ONNX MatMul has no third input, so the pass dies with IndexError
		# (op_graph_optimizations.py:4201). Gemm computes alpha*A@B + beta*C in one node
		# with a real bias, which is the same arithmetic and leaves nothing to squash.
		en = b + '/ff_eps'
		new_inits.append(numpy_helper.from_array(
			numpy.full((O,), float(numpy.asarray(eps).ravel()[0]), numpy.float32), en))
		on = b + '/ff_one'
		new_inits.append(numpy_helper.from_array(numpy.asarray(one, numpy.float32), on))

		s2, t, t2, r, d, d4 = (b + '/ff_' + k for k in ('s2', 't', 't2', 'r', 'd', 'd4'))
		conv_out = b + '/ff_conv'
		out_nodes += [
			helper.make_node('Mul', [style, style], [s2], name=b + '/ff_s2'),
			helper.make_node('Gemm', [s2, an, en], [t2], name=b + '/ff_demod',
							 alpha=1.0, beta=1.0, transA=0, transB=0),
			helper.make_node('Sqrt', [t2], [r], name=b + '/ff_demod_sqrt'),
			helper.make_node('Div', [on, r], [d], name=b + '/ff_demod_div'),
			helper.make_node('Reshape', [d, shape_init([1, O, 1, 1], b + '/ff_d_shape')], [d4],
							 name=b + '/ff_demod_reshape'),
			helper.make_node(node.op_type, conv_ins, [conv_out], name=b, **attrs),
			helper.make_node('Mul', [conv_out, d4], [node.output[0]], name=b + '/ff_demod_scale'),
		]
		n_demod += 1

	print('  rewrote %d demodulated + %d modulated-only convs' % (n_demod, n_plain))
	if n_demod + n_plain != 20:
		print('  WARNING: expected 20 modulated convs, matched %d' % (n_demod + n_plain))

	del g.node[:]
	g.node.extend(out_nodes)
	g.initializer.extend(new_inits)

	# ---- second pass: the six `to_rgbs.N/upsample` blocks, the last rank-6 tensors
	#
	# StyleGAN's upfirdn2d upsample inserts a zero after every sample by reshaping NHWC to
	# SIX dimensions and padding the two unit axes:
	#
	#   [1,C,H,W] -Transpose-> [1,H,W,C] -Reshape-> [1,H,1,W,1,C] -Pad-> [1,H,2,W,2,C]
	#                                              -Reshape-> [1,2H,2W,C]
	#
	# Rank 6 is past every limit that matters (the converter's elementwise broadcast path
	# is `max_rank <= 5`, the arch linter's MAX_RANK is 4), and it exists only to express
	# "scatter each pixel onto an even grid position" -- which IS a stride-2 transposed
	# convolution with an identity kernel.  Same values, rank 4, one op:
	#
	#   ConvTranspose(x, I[C,C,1,1], strides 2, output_padding 1) -> [1,C,2H,2W]
	#
	# output_padding 1 is what makes it 2H rather than 2*(H-1)+1: the zero-insert leaves a
	# trailing zero after the LAST sample too, and without it that column is dropped.
	m = shape_inference.infer_shapes(strip_value_info(m), strict_mode=False)
	g = m.graph
	rank = {vi.name: len(vi.type.tensor_type.shape.dim) for vi in g.value_info}
	dims = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in g.value_info}
	init = {t.name: t for t in g.initializer}
	prod = {o: n for n in g.node for o in n.output}
	cons = {}
	for n in g.node:
		for i in n.input:
			cons.setdefault(i, []).append(n)

	def only(t):
		c = cons.get(t, [])
		return c[0] if len(c) == 1 else None

	rewired = {}          # old tensor name -> new tensor name
	extra_nodes = {}      # id(node to splice before) -> replacement nodes
	extra_inits = []
	drop = set()
	n_up = 0

	for rs1 in list(g.node):
		if rs1.op_type != 'Reshape' or rank.get(rs1.output[0]) != 6:
			continue
		tp = prod.get(rs1.input[0])
		if tp is None or tp.op_type != 'Transpose':
			continue
		perm = next((list(a.ints) for a in tp.attribute if a.name == 'perm'), None)
		if perm != [0, 2, 3, 1]:
			continue
		pad = only(rs1.output[0])
		if pad is None or pad.op_type != 'Pad':
			continue
		rs2 = only(pad.output[0])
		if rs2 is None or rs2.op_type != 'Reshape' or rank.get(rs2.output[0]) != 4:
			continue
		pv = numpy_helper.to_array(init[pad.input[1]]).tolist() if pad.input[1] in init else None
		# begins[6] + ends[6]: one trailing element on the two unit axes, nothing else.
		if pv != [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]:
			print('  to_rgbs upsample: unexpected pad %s, left alone' % pv)
			continue

		x = tp.input[0]
		C = dims[x][1]
		wn = rs1.name + '/ff_up_I'
		eye = numpy.eye(C, dtype=numpy.float32).reshape(C, C, 1, 1)
		extra_inits.append(numpy_helper.from_array(eye, wn))
		up = rs1.name + '/ff_up'
		ct = helper.make_node(
			'ConvTranspose', [x, wn], [up], name=rs1.name + '/ff_upsample',
			kernel_shape=[1, 1], strides=[2, 2], pads=[0, 0, 0, 0], output_padding=[1, 1])
		# back to NHWC, because Pad_1 and the blur conv downstream still work there
		nhwc = rs1.name + '/ff_up_nhwc'
		tr = helper.make_node(
			'Transpose', [up], [nhwc], name=rs1.name + '/ff_up_t', perm=[0, 2, 3, 1])
		rewired[rs2.output[0]] = nhwc
		# The replacement is spliced in AT the Transpose it replaces, not prepended: its
		# input is the previous block's conv output, which is produced partway down the
		# graph, and ONNX requires the node list to stay topologically sorted.
		extra_nodes[id(tp)] = [ct, tr]
		drop.update(id(n) for n in (tp, rs1, pad, rs2))
		n_up += 1

	if rewired:
		kept = []
		for n in g.node:
			if id(n) in extra_nodes:
				kept.extend(extra_nodes[id(n)])
			if id(n) in drop:
				continue
			for k, i in enumerate(n.input):
				if i in rewired:
					n.input[k] = rewired[i]
			kept.append(n)
		del g.node[:]
		g.node.extend(kept)
		g.initializer.extend(extra_inits)
	print('  replaced %d upfirdn2d upsamples with a stride-2 ConvTranspose' % n_up)

	# ---- third pass: the blur blocks, as one grouped conv each
	#
	# Every StyleGAN blur -- 6 in the encoder, 6 before a generator conv, 6 in the
	# to_rgbs upsamples -- is exported as upfirdn2d's "depthwise via the batch axis":
	#
	#   [1,C,H,W] -T(0,2,3,1)-> [1,H,W,C] -Pad-> -Slice-> -T(0,3,1,2)-> [1,C,H',W']
	#             -Reshape-> [C,1,H',W'] -Conv(K[1,1,4,4])-> [C,1,Ho,Wo]
	#             -Reshape-> [1,C,Ho,Wo]
	#
	# One kernel over C batches IS a depthwise convolution over C channels, and the Pad
	# is what the conv's own `pads` express.  The converter already sees the DepthWiseConv2d
	# -- what it cannot see through is the layout churn around it, and that churn is where
	# gpen's time goes: the shipped net moves 86.51 MB through Transpose against 33.36 MB
	# of compute output (ratio 2.59), and its ten largest Transposes are all named for these
	# blocks.  So the win is not the node count, it is the traffic.
	#
	# The Slice is a full-range no-op that onnxsim keeps (starts 0, steps 1, ends == the
	# padded extent); it is asserted to be one rather than assumed, and skipped.
	m = shape_inference.infer_shapes(strip_value_info(m), strict_mode=False)
	g = m.graph
	dims = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
			for vi in list(g.value_info) + list(g.input) + list(g.output)}
	inits = {t.name: t for t in g.initializer}
	prod = {o: n for n in g.node for o in n.output}
	cons = {}
	for n in g.node:
		for i in n.input:
			cons.setdefault(i, []).append(n)

	def arr(name):
		return numpy_helper.to_array(inits[name]) if name in inits else None

	def attrs_of(n):
		return {a.name: (list(a.ints) if len(a.ints) else a.i) for a in n.attribute}

	def sole(n):
		"""The producer of n.input[0], but only if n is that tensor's only consumer.

		Anything with a second consumer has to stay, so the chain is not foldable.
		"""
		if n is None or len(cons.get(n.input[0], [])) != 1:
			return None
		return prod.get(n.input[0])

	blur_nodes = {}      # id(node to splice at) -> [replacement nodes]
	blur_drop = set()
	blur_inits = []
	n_blur = 0

	for conv in list(g.node):
		if conv.op_type != 'Conv' or len(conv.input) != 2:
			continue
		w = arr(conv.input[1])
		# The depthwise-via-batch kernel is [1,1,kh,kw]: one input channel, one output.
		if w is None or w.ndim != 4 or w.shape[0] != 1 or w.shape[1] != 1:
			continue
		kh, kw = int(w.shape[2]), int(w.shape[3])
		ca = attrs_of(conv)
		# All the spatial framing lives in the Pad; a conv that also strides or pads on its
		# own is a different block and is left alone.
		if (ca.get('strides', [1, 1]) != [1, 1] or ca.get('dilations', [1, 1]) != [1, 1]
				or ca.get('pads', [0] * 4) != [0] * 4 or ca.get('group', 1) != 1):
			continue

		rs = sole(conv)
		if rs is None or rs.op_type != 'Reshape':
			continue
		tp1 = sole(rs)
		if tp1 is None or tp1.op_type != 'Transpose' or attrs_of(tp1).get('perm') != [0, 3, 1, 2]:
			continue
		up = sole(tp1)
		# The full-range Slice, if onnxsim kept one.  Identity == same shape, starts 0,
		# steps 1; that needs no reasoning about how the ends were spelled.
		sl = None
		if up is not None and up.op_type == 'Slice':
			st, sp = arr(up.input[1]), arr(up.input[4]) if len(up.input) > 4 else None
			if (st is None or any(int(v) for v in st)
					or (sp is not None and any(int(v) != 1 for v in sp))
					or dims.get(up.input[0]) is None
					or dims.get(up.input[0]) != dims.get(up.output[0])):
				continue
			sl, up = up, sole(up)
		pad = up
		if pad is None or pad.op_type != 'Pad':
			continue
		if next((a.s for a in pad.attribute if a.name == 'mode'), b'constant') != b'constant':
			continue
		cv = arr(pad.input[2]) if len(pad.input) > 2 else None
		if cv is not None and float(numpy.asarray(cv).ravel()[0]) != 0.0:
			continue
		tp0 = sole(pad)
		if tp0 is None or tp0.op_type != 'Transpose' or attrs_of(tp0).get('perm') != [0, 2, 3, 1]:
			continue

		x = tp0.input[0]
		shape = dims.get(x)
		if not shape or len(shape) != 4 or shape[0] != 1:
			continue
		C = shape[1]

		# The pad is expressed in NHWC: begins[4] + ends[4] over N,H,W,C.  Conv wants
		# [top, left, bottom, right].  Anything padding N or C is not a spatial blur and is
		# left alone rather than guessed at.
		pv = arr(pad.input[1])
		if pv is None or len(pv) != 8:
			continue
		pv = [int(v) for v in pv]
		if pv[0] or pv[3] or pv[4] or pv[7]:
			continue
		pads = [pv[1], pv[2], pv[5], pv[6]]

		# The batch trick only holds if the two Reshapes are exactly the [1,C,..]<->[C,1,..]
		# pair; assert the shapes rather than trust the names.
		rs2 = cons.get(conv.output[0], [])
		if len(rs2) != 1 or rs2[0].op_type != 'Reshape':
			continue
		rs2 = rs2[0]
		h2, w2 = shape[2] + pads[0] + pads[2], shape[3] + pads[1] + pads[3]
		ho, wo = h2 - kh + 1, w2 - kw + 1
		if (dims.get(rs.output[0]) != [C, 1, h2, w2]
				or dims.get(conv.output[0]) != [C, 1, ho, wo]
				or dims.get(rs2.output[0]) != [1, C, ho, wo]):
			continue

		wn = conv.name + '/ff_blur_W'
		# One kernel, repeated per channel: that IS what the batch trick computed.
		tiled = numpy.repeat(w.reshape(1, 1, kh, kw).astype(numpy.float32), C, axis=0)
		blur_inits.append(numpy_helper.from_array(numpy.ascontiguousarray(tiled), wn))

		blur_nodes[id(tp0)] = [helper.make_node(
			'Conv', [x, wn], [rs2.output[0]], name=conv.name + '/ff_blur',
			kernel_shape=[kh, kw], strides=[1, 1], dilations=[1, 1],
			pads=pads, group=C)]
		blur_drop.update(id(n) for n in (tp0, pad, tp1, rs, conv, rs2))
		if sl is not None:
			blur_drop.add(id(sl))
		n_blur += 1

	if blur_nodes:
		kept = []
		for n in g.node:
			# Spliced in AT the first node it replaces, so the list stays topologically
			# sorted: x is produced partway down the graph, not at the top.
			if id(n) in blur_nodes:
				kept.extend(blur_nodes[id(n)])
			if id(n) in blur_drop:
				continue
			kept.append(n)
		del g.node[:]
		g.node.extend(kept)
		g.initializer.extend(blur_inits)
	print('  collapsed %d blur blocks into a grouped conv' % n_blur)
	if n_blur != 18:
		print('  WARNING: expected 18 blur blocks, matched %d' % n_blur)

	# ---- fourth pass: an explicit zero bias on every bias-less conv
	#
	# The 2.49 converter's `squash_sum` folds an ElementwiseSum into a preceding NN node by
	# writing into that node's bias, and reads `input_names[2]` without checking the node
	# HAS three inputs (op_graph_optimizations.py:4201 -> IndexError). StyleGAN convolutions
	# carry no bias -- it is a separate Add after the noise -- so the graph is full of
	# two-input convs and the pass dies on the first one it matches.
	#
	# A zero bias is numerically inert, and inert under the demodulation scale too, since
	# 0 * d == 0. Caught by the --layout screen before any device time (trap #12).
	init_names = {t.name for t in g.initializer}
	shape_of = {t.name: list(t.dims) for t in g.initializer}
	n_bias = 0
	for n in g.node:
		if n.op_type not in ('Conv', 'ConvTranspose') or len(n.input) >= 3:
			continue
		w = shape_of.get(n.input[1])
		if w is None:
			continue
		group = 1
		for a in n.attribute:
			if a.name == 'group':
				group = a.i
		# Conv kernels are [O, I/group, kh, kw]; ConvTranspose kernels are [I, O/group, ...],
		# so the output channel count is NOT in the same axis for the two.
		out_ch = w[0] if n.op_type == 'Conv' else w[1] * group
		bn = (n.name or ('conv%d' % n_bias)) + '/ff_zero_bias'
		g.initializer.append(numpy_helper.from_array(
			numpy.zeros((out_ch,), dtype=numpy.float32), bn))
		n.input.append(bn)
		n_bias += 1
	print('  added a zero bias to %d bias-less convs (converter squash_sum)' % n_bias)

	# The old kernel chains are now unreachable; onnxsim drops them along with the
	# initialisers they held.
	m = shape_inference.infer_shapes(strip_value_info(m), strict_mode=False)
	s, ok = onnxsim.simplify(m, overwrite_input_shapes={'input': [1, 3, 256, 256]})
	if not ok:
		sys.exit('  onnxsim reported failure (post-pass)')
	m = shape_inference.infer_shapes(strip_value_info(s), strict_mode=False)
	checker.check_model(m)

	# The two properties the whole surgery exists for, asserted rather than assumed.
	statics = {t.name for t in m.graph.initializer}
	dyn = [n.name for n in m.graph.node
		   if n.op_type in ('Conv', 'ConvTranspose') and n.input[1] not in statics]
	ranks = {}
	for vi in m.graph.value_info:
		ranks[len(vi.type.tensor_type.shape.dim)] = \
			ranks.get(len(vi.type.tensor_type.shape.dim), 0) + 1
	print('  dynamic kernels left: %d %s' % (len(dyn), dyn[:4]))
	print('  activation ranks    :', dict(sorted(ranks.items())))
	if dyn:
		sys.exit('  a computed kernel survived -- HTP will not map it')
	if any(k >= 5 for k in ranks):
		print('  WARNING: rank>=5 activations survived')
	return save(m, 'gpen_bfr_256_sim')


TASKS = {
	'arcface': do_arcface,
	'2dfan4': do_2dfan4,
	'yoloface': do_yoloface,
	'hyperswap': do_hyperswap,
	# The fp32 demotion IS the shipping hyperswap graph as of 2026-08-24 -- convert.sh
	# reads `hyperswap_1a_256_fp32.onnx`, so `all` has to produce it or a clean rebuild
	# stops at a missing file.
	'hyperswap_fp32': do_hyperswap_fp32,
	'nsfw': do_nsfw,
	'gpen': do_gpen,
	'yoloface_slicefix': do_yoloface_slicefix,
	'inswapper': do_inswapper,
}

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('models', nargs='+', choices=list(TASKS) + ['all'])
	args = ap.parse_args()
	# Experiments, not the shipping set -- ask for these by name.  `hyperswap_fp32` was
	# here until it was promoted (2026-08-24); it is now what convert.sh reads.
	EXPERIMENTAL = ('yoloface_slicefix',)
	names = [n for n in TASKS if n not in EXPERIMENTAL] if 'all' in args.models else args.models
	for n in names:
		print(n + ':')
		TASKS[n]()
