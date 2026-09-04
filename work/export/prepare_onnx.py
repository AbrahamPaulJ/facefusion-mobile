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


# ------------------------------------------------------------------ wav2lip

def do_wav2lip(name='wav2lip_gan_96'):
	"""Lip syncer. Batch is dynamic on both inputs and the output; nothing else to do.

	143 nodes, 7 op types (Conv, ConvTranspose, Relu, Add, Concat, BatchNormalization,
	Sigmoid), every one of which already ships in this project. `wav2lip_96` and
	`wav2lip_gan_96` are the same architecture to the node -- byte-identical ONNX size --
	so this handles either; upstream 3.8.2 defaults to the GAN one and so do we.
	"""
	m = load(name)
	changed = pin_batch(m)
	print('  pinned batch on:', changed)
	m = shape_inference.infer_shapes(strip_value_info(m), strict_mode=False)
	checker.check_model(m)
	return save(m, name + '_b1')


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


SHAPE_OPERANDS = {
	# op          -> the input positions that carry shape, not data
	'Reshape':    (1,),
	'Tile':       (1,),
	'Expand':     (1,),
	'Slice':      (1, 2, 3, 4),
	'Pad':        (1,),
	'Unsqueeze':  (1,),
	'Squeeze':    (1,),
	'ConstantOfShape': (0,),
}


def freeze_shape_plumbing(model, feeds):
	"""Turn every COMPUTED shape operand into a constant, by running the graph once.

	A fixed-shape deployment has exactly one value for each of these, and QNN needs the
	shapes static anyway -- but onnxsim cannot prove it here, so it leaves the machinery
	that computes them at runtime.  edtalk's decoder is the case: seven `Tile` nodes take a
	COMPUTED `repeats`, so every dim after them is `unk__NNN`, and the `Shape -> Slice ->
	Concat` chains that feed them stay in the graph.  Those chains are most of the 41
	`Shape` and 40 `StridedSlice` nodes, and -- because a `Shape` reading a tensor counts
	as a second consumer of it -- they also BLOCK `collapse_blur_blocks` on all sixteen
	decoder blur blocks, which is how this was found.

	Freezing them and re-simplifying makes the whole decoder static, and onnxsim then
	deletes the dead chains itself.

	⚠ This is only sound because the inputs are pinned.  It is exactly the assumption
	`--input_dim` already makes for the converter; a graph that was meant to take a second
	resolution must not go through here.  `do_edtalk`'s bit-identical check against the raw
	graph is what actually holds it honest.
	"""
	import tempfile
	import onnxruntime

	g = model.graph
	inits = {t.name for t in g.initializer}
	prod = {o: n for n in g.node for o in n.output}

	want = []
	for n in g.node:
		for pos in SHAPE_OPERANDS.get(n.op_type, ()):
			if pos >= len(n.input):
				continue
			t = n.input[pos]
			if t and t not in inits and t in prod and t not in want:
				want.append(t)
	if not want:
		return 0

	probe = onnx.ModelProto()
	probe.CopyFrom(model)
	strip_value_info(probe)
	for name in want:
		probe.graph.output.add().name = name
	with tempfile.TemporaryDirectory() as td:
		path = os.path.join(td, 'freeze.onnx')
		onnx.save(probe, path)
		sess = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
		values = sess.run(want, feeds)

	frozen = 0
	for name, v in zip(want, values):
		v = numpy.asarray(v)
		# Shape operands are small int64 vectors.  Anything else reaching this list would
		# be a data tensor and must not be frozen, so it is skipped rather than trusted.
		if v.dtype != numpy.int64 or v.ndim > 1 or v.size > 16:
			continue
		g.initializer.append(numpy_helper.from_array(v, name))
		frozen += 1
	# The producers are now shadowed by an initializer of the same name; drop them so the
	# graph has one definition per tensor, then let the dead-code sweep in onnxsim take
	# the Shape chains behind them.
	names = {t.name for t in g.initializer}
	kept = [n for n in g.node if not (n.output and all(o in names for o in n.output))]
	del g.node[:]
	g.node.extend(kept)
	print('  froze %d computed shape operands' % frozen)
	return frozen


def blur_dims_from_ort(model, feeds, radius=6):
	"""Real shapes for the tensors around every `[1,1,kh,kw]` Conv, read by RUNNING it.

	`collapse_blur_blocks` asserts the batch trick on shapes, and static shape inference
	cannot supply them for a decoder like edtalk's: six `Tile` nodes take a COMPUTED
	`repeats`, so every dim after them is `unk__NNN` and every assertion fails on a graph
	that is actually static.  Sixteen of edtalk's twenty-eight blur blocks were skipped for
	that reason alone -- not because the pattern was absent.

	So this does what the EyeLike fold does: asks onnxruntime instead of inferring.  Only
	the tensors within `radius` producers of a candidate Conv are probed, because probing
	all 1110 would materialise several GB to read a hundred shapes.
	"""
	import tempfile
	import onnxruntime

	g = model.graph
	inits = {t.name for t in g.initializer}
	prod = {o: n for n in g.node for o in n.output}
	cons = {}
	for n in g.node:
		for i in n.input:
			cons.setdefault(i, []).append(n)

	want = set()
	for conv in g.node:
		if conv.op_type != 'Conv' or len(conv.input) != 2 or conv.input[1] not in inits:
			continue
		w = next(t for t in g.initializer if t.name == conv.input[1])
		if len(w.dims) != 4 or w.dims[0] != 1 or w.dims[1] != 1:
			continue
		want.update(conv.output)
		for c in cons.get(conv.output[0], []):
			want.update(c.output)
		cur = conv.input[0]
		for _ in range(radius):
			want.add(cur)
			n = prod.get(cur)
			if n is None:
				break
			cur = n.input[0]

	existing = {o.name for o in g.output}
	want = sorted(n for n in want if n not in existing and n not in inits)
	if not want:
		return {}

	probe = onnx.ModelProto()
	probe.CopyFrom(model)
	# The graph has been edited since onnxsim wrote its value_info, and ORT REFUSES to load
	# a model whose declared shapes disagree with what it re-infers -- it is the check that
	# caught the mis-lowered Squeeze in the first place. Drop it and let ORT infer.
	strip_value_info(probe)
	for name in want:
		probe.graph.output.add().name = name
	# ORT wants a path for a model this size; the scratch file is deleted with the dir.
	with tempfile.TemporaryDirectory() as td:
		path = os.path.join(td, 'probe.onnx')
		onnx.save(probe, path)
		sess = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
		values = sess.run(want, feeds)
	out = {n: list(numpy.asarray(v).shape) for n, v in zip(want, values)}
	print('  read %d real shapes out of onnxruntime' % len(out))
	return out


def collapse_blur_blocks(m, expect=None, dims_hint=None):
	"""Collapse every StyleGAN `upfirdn2d` blur into ONE grouped conv.  Returns the count.

	Both StyleGAN2 graphs this project converts -- `gpen_bfr_256` and `edtalk_256` -- export
	their blur as "depthwise via the batch axis": the kernel is [1,1,kh,kw] and the C
	channels are folded into N so that one kernel covers all of them.

	    [1,C,H,W] -(Transpose 0,2,3,1)-> -Pad-> -Slice-> -(Transpose 0,3,1,2)->
	              -Reshape-> [C,1,H',W'] -Conv(K[1,1,kh,kw])-> [C,1,Ho,Wo]
	              -Reshape-> [1,C,Ho,Wo]

	One kernel over C batches IS a depthwise convolution over C channels, and the Pad is
	what the conv's own `pads` express.  The converter already sees a DepthWiseConv2d --
	what it cannot see through is the layout churn around it, and a Reshape that mixes N
	with C is exactly what forces it to leave its preferred layout and transpose back.
	That churn is the cost: gpen shipped 86.51 MB through Transpose against 33.36 MB of
	compute output (ratio 2.59) and the ten largest were all named for these blocks;
	collapsing them cut the enhancer 9.5x for no accuracy change (trap #9).  The win is
	the traffic, not the node count.

	⚠ **The two transposes are OPTIONAL and which layout the Pad speaks depends on them.**
	gpen carries them, so its pad is NHWC.  edtalk's onnxsim run leaves the block in NCHW
	with no transposes at all, so its pad is NCHW -- the same eight numbers meaning
	different axes.  Reading the pad in the wrong layout silently pads N and C instead of
	H and W, so the layout is decided by what was matched, never assumed.

	The Slice is a full-range no-op that onnxsim keeps; it is asserted to be one rather
	than assumed, and skipped.  It may name only the spatial axes, so the identity test is
	"starts are 0, steps are 1, and the shape did not change" -- which needs no reasoning
	about how the ends were spelled or which axes were listed.
	"""
	m2 = shape_inference.infer_shapes(strip_value_info(m), strict_mode=False)
	del m.graph.value_info[:]
	m.graph.value_info.extend(m2.graph.value_info)
	g = m.graph
	dims = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
			for vi in list(g.value_info) + list(g.input) + list(g.output)}
	# Measured shapes win over inferred ones: a graph whose decoder is symbolic still has
	# one real shape per tensor, and every assertion below is about that one.
	dims.update(dims_hint or {})
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
		up = sole(rs)
		# gpen transposes back to NCHW here; edtalk never left it.  Whether this node is
		# present is what decides how the Pad below is read.
		tp1 = None
		if up is not None and up.op_type == 'Transpose':
			if attrs_of(up).get('perm') != [0, 3, 1, 2]:
				continue
			tp1, up = up, sole(up)
		# The full-range Slice, if onnxsim kept one.  Identity == same shape, starts 0,
		# steps 1; that needs no reasoning about how the ends were spelled.
		sl = None
		if up is not None and up.op_type == 'Slice':
			st = arr(up.input[1])
			sp = arr(up.input[4]) if len(up.input) > 4 else None
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
		if tp0 is not None and tp0.op_type == 'Transpose':
			if attrs_of(tp0).get('perm') != [0, 2, 3, 1]:
				continue
		else:
			tp0 = None
		# Both transposes travel together: one without the other is not this block.
		if (tp0 is None) != (tp1 is None):
			continue

		head = tp0 if tp0 is not None else pad
		x = head.input[0]
		shape = dims.get(x)
		if not shape or len(shape) != 4 or shape[0] != 1:
			continue
		C = shape[1]

		# begins[4] + ends[4], over NHWC when the transposes are there and NCHW when they
		# are not.  Conv wants [top, left, bottom, right].  Anything padding N or C is not
		# a spatial blur and is left alone rather than guessed at.
		pv = arr(pad.input[1])
		if pv is None or len(pv) != 8:
			continue
		pv = [int(v) for v in pv]
		if tp0 is not None:
			if pv[0] or pv[3] or pv[4] or pv[7]:      # N, C in NHWC
				continue
			pads = [pv[1], pv[2], pv[5], pv[6]]       # H, W
		else:
			if pv[0] or pv[1] or pv[4] or pv[5]:      # N, C in NCHW
				continue
			pads = [pv[2], pv[3], pv[6], pv[7]]       # H, W

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

		blur_nodes[id(head)] = [helper.make_node(
			'Conv', [x, wn], [rs2.output[0]], name=conv.name + '/ff_blur',
			kernel_shape=[kh, kw], strides=[1, 1], dilations=[1, 1],
			pads=pads, group=C)]
		blur_drop.update(id(n) for n in (pad, rs, conv, rs2))
		for n in (tp0, tp1, sl):
			if n is not None:
				blur_drop.add(id(n))
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
	if expect is not None and n_blur != expect:
		print('  WARNING: expected %d blur blocks, matched %d' % (expect, n_blur))
	return n_blur


# ---------------------------------------------------------------- modulated convs

def rewrite_modulated_convs(m, expect_total=None):
	"""Make every StyleGAN2 modulated conv kernel STATIC, in place on `m`.

	Shared by `do_gpen` and `do_edtalk` -- see `do_gpen`'s docstring for the algebra:

	    Conv(x, W[o,i,..] * s[i])[o] * d[o]  ==  Conv(x * s[i], W)[o] * d[o]
	    d[o] = 1/sqrt( sum_i s[i]^2 * A[o,i] + eps ),  A[o,i] = sum_kh,kw W[o,i,kh,kw]^2

	gpen's 13 demodulated convs happen to carry no bias, so the original version of this
	bailed rather than fold one. Here a demodulated conv's bias, when it has one, is added
	AFTER the demodulation scale (a bias rides on the true output, never on the scale
	itself) instead of assumed absent -- edtalk is not expected to split the same way gpen
	does.

	Returns (n_demod, n_plain, n_bias_folded).
	"""
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
	n_demod = n_plain = n_bias_folded = 0

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
		bias_name = node.input[2] if has_bias else None
		conv_ins = [xs, wn]
		if demod_t is None:
			# No demodulation: the bias rides along inside the conv, exactly as before.
			if has_bias:
				conv_ins.append(bias_name)
			out_nodes.append(helper.make_node(
				node.op_type, conv_ins, [node.output[0]], name=b, **attrs))
			n_plain += 1
			continue

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

		s2, t2, r, d, d4 = (b + '/ff_' + k for k in ('s2', 't2', 'r', 'd', 'd4'))
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
		]
		if has_bias:
			# The bias rides on the true output, added AFTER the demod scale -- gpen's 13
			# demodulated convs carry none, so this path is untested by gpen.
			scaled = b + '/ff_demod_scaled'
			out_nodes.append(helper.make_node(
				'Mul', [conv_out, d4], [scaled], name=b + '/ff_demod_scale'))
			b4 = b + '/ff_bias4'
			out_nodes.append(helper.make_node(
				'Reshape', [bias_name, shape_init([1, O, 1, 1], b + '/ff_bias_shape')], [b4],
				name=b + '/ff_bias_reshape'))
			out_nodes.append(helper.make_node(
				'Add', [scaled, b4], [node.output[0]], name=b + '/ff_bias_add'))
			n_bias_folded += 1
		else:
			out_nodes.append(helper.make_node(
				'Mul', [conv_out, d4], [node.output[0]], name=b + '/ff_demod_scale'))
		n_demod += 1

	print('  rewrote %d demodulated (%d with a folded bias) + %d modulated-only convs'
		  % (n_demod, n_bias_folded, n_plain))
	if expect_total is not None and n_demod + n_plain != expect_total:
		print('  WARNING: expected %d modulated convs, matched %d' % (expect_total, n_demod + n_plain))

	del g.node[:]
	g.node.extend(out_nodes)
	g.initializer.extend(new_inits)
	return n_demod, n_plain, n_bias_folded


def rewrite_demod_factor(m, expect_total=None):
	"""The SAME identity as `rewrite_modulated_convs`, for an export that already applies
	it halfway.

	Assumed, per HANDOFF, that edtalk's raw graph matches gpen's -- Conv fed a kernel
	computed at runtime from `Reshape(Mul(W5, style))`. It measures 63.32 ms/frame and the
	same profiling that found the shape (`work/device/prof_by_optype.py`) named the culprit,
	so this was checked before believing it: **it does not match.** edtalk's export already
	modulates the ACTIVATION and feeds Conv a STATIC weight --

	    Conv(x * s[i] * eq_lr, W)[o] * d[o]

	-- `rewrite_modulated_convs` finds zero dynamic kernels here because there are none: the
	Conv's own weight input folds to a plain initializer under onnxsim once shapes are
	pinned. The 90.4%-of-cycles tensor HANDOFF found is not feeding a Conv at all -- it is
	built ONLY to compute `d[o]`, in a chain that ends in a Mul applied AFTER the conv:

	    Div(1, Sqrt(Add(ReduceSum(Pow(Mul(W5, style_reshaped), 2)), eps)))

	Same algebra, same fix -- `d[o] = 1/sqrt(sum_i s[i]^2 A[o,i] + eps)`, A constant -- but
	the REWIRE differs: the Conv and its activation-side Mul are already correct and are left
	completely alone. Only the Div node's producer chain is replaced, and the replacement's
	last node reuses the ORIGINAL Div's output name, so the Reshape and Mul that consume
	`d[o]` downstream (unchanged, applying it after the conv) need no rewiring at all. The
	orphaned Mul/Pow/ReduceSum/Add/Sqrt chain and the big weight initializer it held are left
	in place for the next onnxsim pass to drop, exactly as `do_gpen`'s old kernel chains are.

	Returns n_rewritten.
	"""
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
	n_rewritten = 0

	for node in g.node:
		if node.op_type != 'Div':
			out_nodes.append(node)
			continue

		one = const_val(node.input[0])
		sq = prod.get(node.input[1])
		if sq is None or sq.op_type != 'Sqrt':
			out_nodes.append(node)
			continue
		ad = prod.get(sq.input[0])
		if ad is None or ad.op_type != 'Add':
			out_nodes.append(node)
			continue
		eps, rs = const_val(ad.input[1]), prod.get(ad.input[0])
		if eps is None:
			eps, rs = const_val(ad.input[0]), prod.get(ad.input[1])
		if rs is None or rs.op_type != 'ReduceSum':
			out_nodes.append(node)
			continue
		pw = prod.get(rs.input[0])
		if pw is None or pw.op_type != 'Pow':
			out_nodes.append(node)
			continue
		mul = prod.get(pw.input[0])
		if mul is None or mul.op_type != 'Mul':
			out_nodes.append(node)
			continue
		W5, style_r = const_val(mul.input[0]), mul.input[1]
		if W5 is None or W5.ndim != 5:
			W5, style_r = const_val(mul.input[1]), mul.input[0]
		if W5 is None or W5.ndim != 5:
			out_nodes.append(node)
			continue

		# Past this point the shape is confirmed: bail loudly rather than skip, exactly as
		# rewrite_modulated_convs does once IT has committed to a match.
		expect(one is not None and eps is not None, node, 'demod eps/numerator not constant')
		sr = prod.get(style_r)
		expect(sr is not None and sr.op_type == 'Reshape', node, 'style is not reshaped')
		style = sr.input[0]

		_, O, I, kh, kw = W5.shape
		b = node.name

		# A[o,i] = sum over the kernel window of W^2 -- W5 here already carries whatever
		# export-time constant (edtalk's equalized-LR scale) the raw Mul used, so this
		# reproduces the ORIGINAL demod factor exactly, not a rescaled one.
		A = (W5[0].astype(numpy.float64) ** 2).sum(axis=(2, 3))          # [O, I]
		an = b + '/ff_A'
		new_inits.append(numpy_helper.from_array(
			numpy.ascontiguousarray(A.T.astype(numpy.float32)), an))     # [I, O]
		en = b + '/ff_eps'
		new_inits.append(numpy_helper.from_array(
			numpy.full((O,), float(numpy.asarray(eps).ravel()[0]), numpy.float32), en))
		on = b + '/ff_one'
		new_inits.append(numpy_helper.from_array(numpy.asarray(one, numpy.float32), on))

		s2, t2, r = (b + '/ff_' + k for k in ('s2', 't2', 'r'))
		out_nodes += [
			helper.make_node('Mul', [style, style], [s2], name=b + '/ff_s2'),
			helper.make_node('Gemm', [s2, an, en], [t2], name=b + '/ff_demod',
							 alpha=1.0, beta=1.0, transA=0, transB=0),
			helper.make_node('Sqrt', [t2], [r], name=b + '/ff_demod_sqrt'),
			# same output name as the node being replaced: nothing downstream changes.
			helper.make_node('Div', [on, r], [node.output[0]], name=b + '/ff_demod_div'),
		]
		n_rewritten += 1

	print('  rewrote %d demodulation factors to a static-A Gemm' % n_rewritten)
	if expect_total is not None and n_rewritten != expect_total:
		print('  WARNING: expected %d demod factors, matched %d' % (expect_total, n_rewritten))

	del g.node[:]
	g.node.extend(out_nodes)
	g.initializer.extend(new_inits)
	return n_rewritten


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

	# ---- first pass: every modulated conv kernel becomes static (shared with do_edtalk)
	n_demod, n_plain, n_bias_folded = rewrite_modulated_convs(m, expect_total=20)
	if n_bias_folded:
		sys.exit('  gpen: %d demodulated conv(s) carried a bias -- previously unseen, check '
				 'the fold' % n_bias_folded)
	g = m.graph

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

	# ---- third pass: the blur blocks, as one grouped conv each.  The matcher is
	# shared with do_edtalk, which is the same StyleGAN2 block in a different layout.
	collapse_blur_blocks(m, expect=18)
	m = shape_inference.infer_shapes(strip_value_info(m), strict_mode=False)
	g = m.graph

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


# -------------------------------------------------------------------- edtalk

def do_edtalk():
	"""`edtalk_256`, the 256x256 lip syncer.  Two BUG fixes, then two reassociating surgeries.

	Screened and rejected once on an op census of the RAW graph -- 4772 nodes, 2 `If`,
	65 `ConstantOfShape` -- which was the wrong screen.  onnxsim with the three input
	shapes pinned folds it to 1170 nodes and BOTH `If` nodes away, which is the same step
	that was the entire surgery for `nsfw_2`.  What is left needs two correctness fixes:

	1. ⚠ **onnxsim 0.4.36 mis-lowers one node, and the graph it writes is WRONG.**  Inside
	   the `If` branch it folds is a `Reshape(conv_out, [-1, 0])` -- torch's `view(-1, 0)`
	   over the [1,512,1,1] output of `enc.net_app.convs.7`.  onnxsim rewrites it as a
	   `Squeeze` and REUSES the shape tensor as the axes tensor, so `[-1, 0]` changes
	   meaning underneath it:

	       Reshape [1,512,1,1] shape=[-1, 0]  ->  [1, 512]   (0 = copy that input dim)
	       Squeeze [1,512,1,1] axes =[-1, 0]  ->  [512, 1]   (drop axes 3 and 0)

	   Same two numbers, transposed result, and the next `Gemm` then contracts over K=1
	   instead of K=512.  It is invisible to a node count and to every op census: the
	   graph still has one `Squeeze` and `Squeeze` is an op we ship.  It shows up only if
	   something tries to RUN the thing -- onnxruntime refuses to load it, because the
	   value_info onnxsim wrote alongside says [1, 512] and disagrees with its own node.
	   Fixed by putting the axes back as [2, 3], which is what the reshape meant.

	2. `EyeLike` is the one op the converter has no translation for, and it is constant:
	   its input is a `ConstantOfShape` sized from `Shape(pose+lip latents)`, which is
	   [1, 26] once the inputs are static.  So the whole
	   `Shape -> Slice -> Squeeze -> Unsqueeze -> Concat -> ConstantOfShape -> EyeLike`
	   chain is a 26x26 identity matrix, read out of onnxruntime rather than inferred --
	   shape inference cannot size it (`unk__471`).  Folding it to an initializer is the
	   same move `do_nsfw` makes wholesale, and it takes the last unconvertible op out.

	Both are exact -- the code checks it, `worst_snr`/`worst` at the bottom read 0.0 / 347 dB
	with the two surgeries below disabled. Then two reassociating surgeries that are NOT
	exact, each measured and each sharing the ALGEBRA with `do_gpen` even where the graph
	shape does not:

	3. **The demodulation-factor surgery** (`rewrite_demod_factor`). `do_gpen`'s defect --
	   90.4% of cycles were `Eltwise_Binary` ops rebuilding a `[1,512,512,3,3]` modulated
	   weight tensor per conv per frame, which is why edtalk measured 63.32 ms/frame on
	   device where GMAC alone predicted ~14.7 -- but NOT `do_gpen`'s graph shape, checked
	   before assumed: `rewrite_modulated_convs` matches zero nodes on this export, because
	   this export already modulates the ACTIVATION and hands Conv a plain static weight
	   (onnxsim folds that reshape once shapes are pinned -- there is no dynamic Conv kernel
	   here to rewrite). The `[1,512,512,3,3]` tensor exists only to be squared and reduced
	   into the demod factor `d[o]`, in a chain applied AFTER the conv rather than baked into
	   its kernel. Same identity, `d[o] = 1/sqrt(sum_i s[i]^2 A[o,i] + eps)` with A constant,
	   collapsing to one static `[1,I]x[I,O]` MatMul -- but the rewire only replaces the Div
	   node's producer chain, in place, under the SAME output name, and leaves the Conv and
	   its activation-side Mul untouched because they were already correct.
	4. `collapse_blur_blocks`, gpen's 28-block surgery re-run here. edtalk's blocks carry NO
	   transposes, so its `Pad` speaks NCHW where gpen's speaks NHWC -- the layout is decided
	   by what was matched, not assumed.

	Both cost real, measured dB (float32 reassociation -- the same products summed in a
	different order), and the check at the bottom is exactly the one the original screen
	never made: run the surgery's OUTPUT against the raw graph in onnxruntime, not just count
	nodes.
	"""
	import onnxsim

	EYE = 26          # the pose (6) + lip (20) latent, measured, not inferred

	m = load('edtalk_256')
	before = len(m.graph.node)
	s, ok = onnxsim.simplify(m, overwrite_input_shapes={
		'source': [1, 1, 80, 16], 'target': [1, 3, 256, 256], 'weight': [1]})
	if not ok:
		sys.exit('  onnxsim reported failure')
	print('  simplified %d -> %d nodes' % (before, len(s.graph.node)))
	for op in ('If', 'Loop', 'Scan'):
		left = sum(1 for n in s.graph.node if n.op_type == op)
		if left:
			sys.exit('  %d %s survived -- control flow does not convert' % (left, op))

	g = s.graph
	inits = {t.name: t for t in g.initializer}

	# 1. the mis-lowered Squeeze.  Matched on the axes VALUE, not the node name, which
	#    onnxsim leaves empty; [-1, 0] is never a legal Squeeze anyway -- axis 0 of a
	#    tensor whose last axis is also being dropped is exactly the reshape idiom.
	fixed = 0
	for n in g.node:
		if n.op_type != 'Squeeze' or len(n.input) < 2:
			continue
		axes = inits.get(n.input[1])
		if axes is None or list(numpy_helper.to_array(axes)) != [-1, 0]:
			continue
		g.initializer.append(
			numpy_helper.from_array(numpy.array([2, 3], numpy.int64), 'edtalk_squeeze_axes'))
		n.input[1] = 'edtalk_squeeze_axes'
		fixed += 1
	print('  Squeeze axes [-1,0] -> [2,3]: %d node(s)' % fixed)
	if fixed != 1:
		sys.exit('  expected exactly one mis-lowered Squeeze; onnxsim changed under us')

	# 2. fold the EyeLike chain to a constant identity, then sweep what it fed on.
	eye = [n for n in g.node if n.op_type == 'EyeLike']
	if len(eye) != 1:
		sys.exit('  expected exactly one EyeLike, found %d' % len(eye))
	g.initializer.append(
		numpy_helper.from_array(numpy.eye(EYE, dtype=numpy.float32), 'edtalk_eye26'))
	for n in g.node:
		for k, i in enumerate(n.input):
			if i == eye[0].output[0]:
				n.input[k] = 'edtalk_eye26'
	g.node.remove(eye[0])
	while True:
		used = set(o.name for o in g.output)
		for n in g.node:
			used.update(n.input)
		dead = [n for n in g.node if n.output and not any(o in used for o in n.output)]
		if not dead:
			break
		for n in dead:
			g.node.remove(n)
	print('  after the fold: %d nodes' % len(g.node))
	for op in ('EyeLike', 'ConstantOfShape'):
		left = sum(1 for n in g.node if n.op_type == op)
		if left:
			sys.exit('  %d %s survived the fold' % (left, op))

	# 3. the demodulation factor becomes a static-A Gemm instead of a per-frame
	#    [1,512,512,3,3] tensor -- measured at 63.32 ms/frame on device without it, 90.4% of
	#    cycles on Eltwise_Binary ops building and squaring that tensor.
	#    ⚠ `rewrite_modulated_convs` (do_gpen's fix, verbatim) matches ZERO nodes here --
	#    checked, not assumed. edtalk's export already modulates the ACTIVATION and feeds
	#    Conv a plain static weight (onnxsim folds that reshape once shapes are pinned), so
	#    there is no dynamic Conv kernel to rewrite. The expensive tensor exists only to feed
	#    the demod factor; see rewrite_demod_factor's docstring for the graph shape and why
	#    it needed its own matcher rather than reusing gpen's.
	n_rewritten = rewrite_demod_factor(s, expect_total=13)
	if n_rewritten == 0:
		sys.exit('  edtalk: matched no demod factors -- did the export change shape?')
	g = s.graph

	# 4. the StyleGAN2 blur blocks, the same surgery gpen needed and for the same reason.
	#    The FLOAT build without this measured transpose/compute 1.92 against the 0.25
	#    defect line (trap #9) with spill_bytes 2.7 GB on an 8 MB VTCM, and its ten largest
	#    transposes were all named `..._conv_blur_Reshape_6` -- gpen's symptom exactly.
	#    ⚠ edtalk's blocks carry NO transposes, so the Pad speaks NCHW where gpen's speaks
	#    NHWC; collapse_blur_blocks decides that from what it matched.
	feeds = {'source': numpy.zeros((1, 1, 80, 16), numpy.float32),
			 'target': numpy.zeros((1, 3, 256, 256), numpy.float32),
			 'weight': numpy.ones((1,), numpy.float32)}
	freeze_shape_plumbing(s, feeds)
	s, ok = onnxsim.simplify(s, overwrite_input_shapes={
		'source': [1, 1, 80, 16], 'target': [1, 3, 256, 256], 'weight': [1]})
	if not ok:
		sys.exit('  onnxsim reported failure on the frozen graph')
	print('  re-simplified to %d nodes' % len(s.graph.node))
	collapse_blur_blocks(s, expect=28, dims_hint=blur_dims_from_ort(s, feeds))

	s = shape_inference.infer_shapes(strip_value_info(s), strict_mode=False)
	checker.check_model(s)
	path = save(s, 'edtalk_256_sim')

	# The check the original screen never made: does it still compute the same thing?
	import onnxruntime
	a = onnxruntime.InferenceSession(os.path.join(MODELS, 'edtalk_256.onnx'),
									 providers=['CPUExecutionProvider'])
	b = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
	rng = numpy.random.default_rng(0)
	worst, worst_snr = 0.0, 1e9
	for _ in range(3):
		feeds = {'source': rng.standard_normal((1, 1, 80, 16)).astype(numpy.float32),
				 'target': rng.random((1, 3, 256, 256)).astype(numpy.float32),
				 'weight': numpy.array([1.0], numpy.float32)}
		ra, rb = a.run(None, feeds)[0], b.run(None, feeds)[0]
		d = ra - rb
		worst = max(worst, float(numpy.abs(d).max()))
		worst_snr = min(worst_snr, float(
			10 * numpy.log10((ra ** 2).sum() / max((d ** 2).sum(), 1e-30))))
	print('  vs the raw graph: %.2f dB, max abs %.3e over 3 inputs' % (worst_snr, worst))
	# The two bug fixes are EXACT: with both reassociating surgeries disabled this same
	# check reads 0.0 / 347 dB. The modulated-conv rewrite and the blur collapse EACH
	# reassociate a float32 sum, gpen's alone costing 107.9 dB and 121 dB respectively, and
	# stacked here they do not simply add -- the combined floor is set by whichever error is
	# larger, so ~100+ dB is still the expectation, not half of it. Anything materially below
	# that is a bug, not rounding: the deploy targets here are 30-45 dB and would hide one
	# completely.
	if worst_snr < 100.0:
		print('  WARNING: %.2f dB is too low for reassociation -- a surgery is WRONG'
			  % worst_snr)
	return path


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
	'wav2lip': do_wav2lip,
	'wav2lip_nogan': lambda: do_wav2lip('wav2lip_96'),
	'edtalk': do_edtalk,
}

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('models', nargs='+', choices=list(TASKS) + ['all'])
	args = ap.parse_args()
	# Experiments, not the shipping set -- ask for these by name.  `hyperswap_fp32` was
	# here until it was promoted (2026-08-24); it is now what convert.sh reads.
	EXPERIMENTAL = ('yoloface_slicefix', 'wav2lip_nogan', 'edtalk')
	names = [n for n in TASKS if n not in EXPERIMENTAL] if 'all' in args.models else args.models
	for n in names:
		print(n + ':')
		TASKS[n]()
