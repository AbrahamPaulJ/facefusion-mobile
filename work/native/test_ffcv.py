"""Verify every native geometry op against the OpenCV call it replaces.

Run in WSL (that is where libffcv.so is built and where cv2 lives):
    bash work/native/build_and_test.sh

The port must reproduce FaceFusion's geometry, not merely be "close" -- a 1 px landmark
shift visibly moves the swap.  Each check prints the measured deviation so a regression is
a number, not a pass/fail opinion.
"""
import ctypes
import os
import sys

import cv2
import numpy

LIB = os.path.join(os.path.dirname(__file__), 'libffcv.so')
ff = ctypes.CDLL(LIB)

f32 = numpy.float32
u8 = numpy.uint8
P = ctypes.POINTER

ff.ff_warp_affine_u8.argtypes = [P(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
                                 P(ctypes.c_double), ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, P(ctypes.c_uint8)]
ff.ff_resize_linear_u8.argtypes = [P(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, P(ctypes.c_uint8)]
ff.ff_invert_affine.argtypes = [P(ctypes.c_double), P(ctypes.c_double)]
ff.ff_umeyama.argtypes = [P(ctypes.c_float), P(ctypes.c_float), ctypes.c_int, P(ctypes.c_double)]
ff.ff_rot_matrix.argtypes = [ctypes.c_double] * 4 + [P(ctypes.c_double)]
ff.ff_nms.argtypes = [P(ctypes.c_float), P(ctypes.c_float), ctypes.c_int,
                      ctypes.c_float, ctypes.c_float, P(ctypes.c_int)]
ff.ff_nms.restype = ctypes.c_int
ff.ff_decode_heatmaps.argtypes = [P(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, P(ctypes.c_float), P(ctypes.c_float)]
ff.ff_face_angle.argtypes = [P(ctypes.c_float)]
ff.ff_face_angle.restype = ctypes.c_int
ff.ff_to_landmark5.argtypes = [P(ctypes.c_float), P(ctypes.c_float)]
ff.ff_gaussian_blur.argtypes = [P(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                                ctypes.c_double, P(ctypes.c_float)]
ff.ff_box_mask.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float,
                           P(ctypes.c_int), P(ctypes.c_float)]
ff.ff_paste_back.argtypes = [P(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
                             P(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                             P(ctypes.c_float), P(ctypes.c_double)]
ff.ff_warp_template.argtypes = [ctypes.c_int]
ff.ff_warp_template.restype = P(ctypes.c_float)   # without this ctypes truncates to int
ff.ff_get_affine_transform.argtypes = [P(ctypes.c_float), P(ctypes.c_float), P(ctypes.c_double)]
ff.ff_create_bounding_box.argtypes = [P(ctypes.c_float), P(ctypes.c_float)]
ff.ff_warp_face_by_bbox.argtypes = [P(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
                                    P(ctypes.c_float), ctypes.c_int, P(ctypes.c_uint8),
                                    P(ctypes.c_double)]
ff.ff_create_area_mask.argtypes = [ctypes.c_int, ctypes.c_int, P(ctypes.c_float),
                                   ctypes.c_int, P(ctypes.c_float)]

# Keep every buffer handed to C alive for the whole test: binding the array to `_` lets
# it be freed on the next assignment while the pointer is still in flight.
_LIVE = []

def cp(a, t):
    a = numpy.ascontiguousarray(a, t)
    _LIVE.append(a)
    return a, a.ctypes.data_as(P({numpy.uint8: ctypes.c_uint8, numpy.float32: ctypes.c_float,
                                  numpy.float64: ctypes.c_double, numpy.int32: ctypes.c_int}[t]))

FAILURES = []

def report(name, measured, limit, unit=''):
    ok = measured <= limit
    if not ok:
        FAILURES.append(name)
    print('  %-34s %10.5f %-6s %s (limit %g)' % (name, measured, unit,
                                                 'OK ' if ok else 'FAIL', limit))

rng = numpy.random.default_rng(7)


def test_affine_helpers():
    print('\n[affine helpers]')
    M = numpy.array([[0.63, -0.006, -44.3], [0.006, 0.63, -39.0]])
    m, mp = cp(M.ravel(), numpy.float64)
    out, op = cp(numpy.zeros(6), numpy.float64)
    ff.ff_invert_affine(mp, op)
    ref = cv2.invertAffineTransform(M)
    report('invertAffineTransform', numpy.abs(out.reshape(2, 3) - ref).max(), 1e-9)

    for ang in (0, 90, 180, 270):
        out, op = cp(numpy.zeros(6), numpy.float64)
        ff.ff_rot_matrix(128.0, 128.0, float(ang), 1.0, op)
        ref = cv2.getRotationMatrix2D((128.0, 128.0), ang, 1)
        report('getRotationMatrix2D(%d)' % ang, numpy.abs(out.reshape(2, 3) - ref).max(), 1e-9)

    # The 5-point face fit, against the call it replaces, over many random faces.
    # Scored in PIXELS after applying the matrix, not on raw matrix entries: the
    # translation terms are ~40, so an absolute tolerance on them is the wrong metric.
    worst = 0.0
    for _ in range(200):
        base = numpy.array([[131, 142], [188, 141], [160, 176], [137, 205], [184, 204]], f32)
        src = base + rng.normal(0, 6, base.shape).astype(f32)
        for which, tmplsize in ((0, 112), (1, 256)):
            tp = ff.ff_warp_template(which)
            tmpl = numpy.ctypeslib.as_array(ctypes.cast(tp, P(ctypes.c_float)),
                                            (10,)).reshape(5, 2) * tmplsize
            ref = cv2.estimateAffinePartial2D(src, tmpl.astype(f32), method=cv2.RANSAC,
                                              ransacReprojThreshold=100)[0]
            _, sp = cp(src.ravel(), numpy.float32)
            _, dp = cp(tmpl.ravel(), numpy.float32)
            out, op = cp(numpy.zeros(6), numpy.float64)
            ff.ff_umeyama(sp, dp, 5, op)
            mine = out.reshape(2, 3)
            pm = cv2.transform(src.reshape(1, -1, 2), mine).reshape(-1, 2)
            pr = cv2.transform(src.reshape(1, -1, 2), ref).reshape(-1, 2)
            worst = max(worst, float(numpy.abs(pm - pr).max()))
    report('umeyama vs estimateAffine2D', worst, 0.01, 'px')


def test_warp_and_resize():
    print('\n[warpAffine / resize]  cv2 uses 5-bit fixed point, this uses float')
    img = (rng.random((360, 640, 3)) * 255).astype(u8)
    for name, M, border, cvborder in (
            ('warp 256 replicate', numpy.array([[0.63, -0.006, -44.3], [0.006, 0.63, -39.0]]), 1, cv2.BORDER_REPLICATE),
            ('warp 256 constant', numpy.array([[0.63, -0.006, -44.3], [0.006, 0.63, -39.0]]), 0, cv2.BORDER_CONSTANT),
            ('warp 112 replicate', numpy.array([[0.28, 0.01, -20.0], [-0.01, 0.28, -12.0]]), 1, cv2.BORDER_REPLICATE)):
        dw = dh = 256 if '256' in name else 112
        _, ip = cp(img, numpy.uint8)
        _, mp = cp(M.ravel(), numpy.float64)
        dst, dp = cp(numpy.zeros((dh, dw, 3)), numpy.uint8)
        ff.ff_warp_affine_u8(ip, img.shape[1], img.shape[0], mp, dw, dh, border, dp)
        ref = cv2.warpAffine(img, M, (dw, dh), borderMode=cvborder, flags=cv2.INTER_LINEAR)
        d = numpy.abs(dst.astype(int) - ref.astype(int))
        report(name + ' maxdiff', d.max(), 2, 'LSB')
        report(name + ' meandiff', d.mean(), 0.35, 'LSB')

    for dw, dh in ((640, 337), (404, 640)):
        _, ip = cp(img, numpy.uint8)
        dst, dp = cp(numpy.zeros((dh, dw, 3)), numpy.uint8)
        ff.ff_resize_linear_u8(ip, img.shape[1], img.shape[0], dw, dh, dp)
        ref = cv2.resize(img, (dw, dh))
        d = numpy.abs(dst.astype(int) - ref.astype(int))
        report('resize %dx%d maxdiff' % (dw, dh), d.max(), 2, 'LSB')
        report('resize %dx%d meandiff' % (dw, dh), d.mean(), 0.35, 'LSB')


def test_nms():
    print('\n[NMS]')
    worst_missing = 0
    for _ in range(50):
        n = int(rng.integers(3, 40))
        xy = rng.random((n, 2)) * 500
        wh = rng.random((n, 2)) * 200 + 40
        boxes = numpy.hstack([xy, xy + wh]).astype(f32)
        scores = (rng.random(n) * 0.5 + 0.5).astype(f32)
        keep = numpy.zeros(n, numpy.int32)
        _, bp = cp(boxes.ravel(), numpy.float32)
        _, sp = cp(scores, numpy.float32)
        _, kp = cp(keep, numpy.int32)
        cnt = ff.ff_nms(bp, sp, n, 0.5, 0.4, kp)
        mine = sorted(keep[:cnt].tolist())
        norm = [(float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])) for b in boxes]
        ref = sorted(numpy.array(cv2.dnn.NMSBoxes(norm, scores.tolist(),
                                                  score_threshold=0.5,
                                                  nms_threshold=0.4)).ravel().tolist())
        worst_missing = max(worst_missing, len(set(ref) ^ set(mine)))
    report('NMS index-set mismatches', worst_missing, 0, 'boxes')


def test_heatmaps():
    print('\n[heatmap decode]')
    hm = numpy.zeros((68, 64, 64), f32)
    for k in range(68):
        cx, cy = rng.random(2) * 60 + 2
        yy, xx = numpy.mgrid[0:64, 0:64]
        hm[k] = numpy.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 6.0).astype(f32)
    _, hp = cp(hm.ravel(), numpy.float32)
    xy, xp = cp(numpy.zeros(68 * 2), numpy.float32)
    pk, pp = cp(numpy.zeros(68), numpy.float32)
    ff.ff_decode_heatmaps(hp, 68, 64, 64, xp, pp)

    # the same computation as run_reference.decode_heatmaps
    flat = hm.reshape(68, -1)
    peak = numpy.argmax(flat, 1)
    py, px = numpy.divmod(peak, 64)
    ys_i, xs_i = numpy.mgrid[0:64, 0:64]
    dist = numpy.sqrt((xs_i[None] - px[:, None, None]) ** 2 + (ys_i[None] - py[:, None, None]) ** 2)
    win = numpy.clip(hm * (dist <= 6.4), 0.0, None)
    m00 = numpy.clip(win.sum((1, 2)), 1.1920929e-07, None)
    rx = (win * (xs_i[None] + 0.5)).sum((1, 2)) / m00
    ry = (win * (ys_i[None] + 0.5)).sum((1, 2)) / m00
    report('soft-argmax x', numpy.abs(xy.reshape(-1, 2)[:, 0] - rx).max(), 2e-4, 'px')
    report('soft-argmax y', numpy.abs(xy.reshape(-1, 2)[:, 1] - ry).max(), 2e-4, 'px')
    report('peak value', numpy.abs(pk - flat.max(1)).max(), 1e-6)


def test_landmark_helpers():
    print('\n[landmark helpers]')
    worst_ang, worst_l5 = 0, 0.0
    for _ in range(100):
        lm = (rng.random((68, 2)) * 400).astype(f32)
        _, lp = cp(lm.ravel(), numpy.float32)
        got = ff.ff_face_angle(lp)
        x1, y1 = lm[0]; x2, y2 = lm[16]
        th = numpy.degrees(numpy.arctan2(y2 - y1, x2 - x1)) % 360
        angles = numpy.linspace(0, 360, 5)
        ref = int(angles[numpy.argmin(numpy.abs(angles - th))] % 360)
        worst_ang = max(worst_ang, abs(got - ref))
        out5, op = cp(numpy.zeros(10), numpy.float32)
        ff.ff_to_landmark5(lp, op)
        r5 = numpy.array([lm[36:42].mean(0), lm[42:48].mean(0), lm[30], lm[48], lm[54]])
        worst_l5 = max(worst_l5, numpy.abs(out5.reshape(5, 2) - r5).max())
    report('estimate_face_angle', worst_ang, 0, 'deg')
    report('convert_to_landmark_5', worst_l5, 1e-4, 'px')


def test_mask_and_paste():
    print('\n[box mask / blur / paste_back]')
    for sigma in (3.0, 9.6, 19.2):
        src = rng.random((128, 128)).astype(f32)
        _, sp = cp(src, numpy.float32)
        dst, dp = cp(numpy.zeros_like(src), numpy.float32)
        ff.ff_gaussian_blur(sp, 128, 128, sigma, dp)
        ref = cv2.GaussianBlur(src, (0, 0), sigma)
        report('GaussianBlur sigma=%.1f' % sigma, numpy.abs(dst - ref).max(), 2e-3)

    pad = numpy.zeros(4, numpy.int32)
    _, pp = cp(pad, numpy.int32)
    mask, mp = cp(numpy.zeros(256 * 256), numpy.float32)
    ff.ff_box_mask(256, 256, 0.3, pp, mp)
    # run_reference.create_box_mask
    blur_amount = int(256 * 0.5 * 0.3); blur_area = max(blur_amount // 2, 1)
    ref = numpy.ones((256, 256), f32)
    ref[:blur_area, :] = 0; ref[-blur_area:, :] = 0
    ref[:, :blur_area] = 0; ref[:, -blur_area:] = 0
    ref = cv2.GaussianBlur(ref, (0, 0), blur_amount * 0.25)
    report('create_box_mask', numpy.abs(mask.reshape(256, 256) - ref).max(), 2e-3)

    frame = (rng.random((360, 640, 3)) * 255).astype(u8)
    crop = (rng.random((256, 256, 3)) * 255).astype(f32)
    M = numpy.array([[0.63, -0.006, -44.3], [0.006, 0.63, -39.0]])
    mine, fp = cp(frame.copy(), numpy.uint8)
    _, cpp = cp(crop.ravel(), numpy.float32)
    _, mkp = cp(mask, numpy.float32)
    _, ap = cp(M.ravel(), numpy.float64)
    ff.ff_paste_back(fp, 640, 360, cpp, 256, 256, mkp, ap)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipeline'))
    import run_reference as R
    ref = R.paste_back(frame, crop, mask.reshape(256, 256), M)
    d = numpy.abs(mine.astype(int) - ref.astype(int))
    report('paste_back maxdiff', d.max(), 2, 'LSB')
    report('paste_back meandiff', d.mean(), 0.1, 'LSB')




def test_lipsync_crop():
    """The lip syncer's own crop chain, which is NOT the swap crop.

    Upstream: warp to ffhq_512 by landmark-5, transform the 68 points by that same
    affine, take create_area_mask(lower-face) and create_bounding_box IN CROP SPACE,
    warp that box to 96x96. Only the last three steps are new; the ffhq_512 warp is
    already covered above.
    """
    print('\n[lip syncer crop]')
    LOWER_FACE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 35, 34, 33, 32, 31]

    # A plausible 68-point set in a 512 crop: an ellipse for the jaw and a scatter for
    # the rest. The hull only ever sees the subset, so the rest just has to exist.
    t = numpy.linspace(numpy.pi * 0.15, numpy.pi * 0.85, 17)
    lm = numpy.zeros((68, 2), numpy.float32)
    lm[:17, 0] = 256 - 190 * numpy.cos(t)
    lm[:17, 1] = 150 + 210 * numpy.sin(t)
    lm[17:, 0] = rng.uniform(120, 400, 51)
    lm[17:, 1] = rng.uniform(100, 380, 51)
    lm[31:36, 0] = numpy.linspace(210, 300, 5)
    lm[31:36, 1] = 290.0

    # getAffineTransform against cv2's
    src = numpy.array([[10.0, 20.0], [300.0, 25.0], [15.0, 260.0]], numpy.float32)
    dst = numpy.array([[0.0, 0.0], [96.0, 0.0], [0.0, 96.0]], numpy.float32)
    _, sp = cp(src.ravel(), numpy.float32)
    _, dp = cp(dst.ravel(), numpy.float32)
    got = numpy.zeros(6, numpy.float64)
    ff.ff_get_affine_transform(sp, dp, got.ctypes.data_as(P(ctypes.c_double)))
    ref = cv2.getAffineTransform(src, dst)
    report('getAffineTransform', numpy.abs(got.reshape(2, 3) - ref).max(), 1e-9)

    # create_bounding_box against upstream's min/max
    _, lp = cp(lm.ravel(), numpy.float32)
    box = numpy.zeros(4, numpy.float32)
    ff.ff_create_bounding_box(lp, box.ctypes.data_as(P(ctypes.c_float)))
    x1, y1 = lm.min(axis=0)
    x2, y2 = lm.max(axis=0)
    report('create_bounding_box', numpy.abs(box - numpy.array([x1, y1, x2, y2])).max(), 0.0)

    # warp_face_by_bounding_box against cv2
    frame = (rng.random((512, 512, 3)) * 255).astype(u8)
    _, fp = cp(frame, numpy.uint8)
    _, bp = cp(box, numpy.float32)
    mine = numpy.zeros(96 * 96 * 3, numpy.uint8)
    aff = numpy.zeros(6, numpy.float64)
    _LIVE.append(mine)
    ff.ff_warp_face_by_bbox(fp, 512, 512, bp, 96, mine.ctypes.data_as(P(ctypes.c_uint8)),
                            aff.ctypes.data_as(P(ctypes.c_double)))
    srcp = numpy.array([[box[0], box[1]], [box[2], box[1]], [box[0], box[3]]], numpy.float32)
    dstp = numpy.array([[0, 0], [96, 0], [0, 96]], numpy.float32)
    M = cv2.getAffineTransform(srcp, dstp)
    ref = cv2.warpAffine(frame, M, (96, 96), flags=cv2.INTER_AREA)
    d = numpy.abs(mine.reshape(96, 96, 3).astype(int) - ref.astype(int))
    report('warp_by_bbox maxdiff', d.max(), 2, 'LSB')
    report('warp_by_bbox meandiff', d.mean(), 0.35, 'LSB')

    # create_area_mask against cv2's convexHull + fillConvexPoly + blur
    got = numpy.zeros(512 * 512, numpy.float32)
    _LIVE.append(got)
    ff.ff_create_area_mask(512, 512, lp, 1, got.ctypes.data_as(P(ctypes.c_float)))
    hull = cv2.convexHull(lm[LOWER_FACE].astype(numpy.int32))
    ref = numpy.zeros((512, 512), numpy.float32)
    cv2.fillConvexPoly(ref, hull, 1.0)
    ref = (cv2.GaussianBlur(ref.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
    got = got.reshape(512, 512)
    report('create_area_mask maxdiff', numpy.abs(got - ref).max(), 1e-6)
    report('create_area_mask meandiff', numpy.abs(got - ref).mean(), 1e-8)
    # The mask is a gate on where the mouth is written, so its AREA is the number that
    # matters more than any single pixel.
    # The FILL is bit-identical: (mine > 0) equals (cv2 > 0) everywhere, so the hull and
    # every scanline span agree exactly, and the area limit is a true 0. What is left is
    # 5.96e-07 max on 5374 pixels, which is one float32 ULP at magnitude 1 out of the
    # Gaussian blur. Measured, not chosen.
    report('create_area_mask area %', abs(got.sum() - ref.sum()) / max(ref.sum(), 1) * 100,
           0.0, '%')


for t in (test_affine_helpers, test_warp_and_resize, test_nms, test_heatmaps,
          test_landmark_helpers, test_mask_and_paste, test_lipsync_crop):
    t()

print('\n' + ('ALL CHECKS PASSED' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)))
sys.exit(1 if FAILURES else 0)
