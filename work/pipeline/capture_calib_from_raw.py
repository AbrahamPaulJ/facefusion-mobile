"""Calibration capture from raw BGR dumps, for every graph in the chain.

Why this exists rather than `run_reference.py --dump-calib`:

  * The material available is `work/device/raw/*.bgr` -- packed BGR frames pulled back off
    the phone, not a video file. run_reference's video path goes through cv2's `mp4v`
    writer/reader, and that codec is exactly what `compare_e2e_raw.py` was written to avoid:
    an MP4 in the path costs ~8 dB. Calibration tensors must be the REAL ones (trap #4), so
    a lossy round trip in the capture is worse than pointless -- it quantises the graph
    against a distribution the device never produces.
  * run_reference's single-image path IS lossless, but one invocation captures one frame and
    each run rewrites `<set>_0000.raw` from zero. Twelve runs would leave one frame's worth.
  * Loading arcface (174 MB) and hyperswap (403 MB) twelve times to capture twelve frames is
    minutes of nothing.

So: load once, walk the frames, accumulate, write. Everything numeric is imported from
run_reference -- this file must never reimplement a stage, or the calibration set stops
matching the reference it is supposed to represent.

    py -3.10 work/pipeline/capture_calib_from_raw.py \\
        --source work/device/raw/source.bgr --sw 1024 --sh 1024 \\
        --target work/device/raw/target12.bgr --tw 1366 --th 720 --frames 12 \\
        --out work/calib
"""
import argparse
import os
import sys

import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_reference as ref


def read_raw(path, w, h, frames=1):
    """Packed BGR, `frames` back to back. The CLI's own on-device format."""
    want = w * h * 3 * frames
    buf = numpy.fromfile(path, dtype=numpy.uint8)
    if buf.size != want:
        sys.exit('%s is %d bytes, expected %d for %dx%d x%d'
                 % (path, buf.size, want, w, h, frames))
    return buf.reshape(frames, h, w, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--sw', type=int, required=True)
    ap.add_argument('--sh', type=int, required=True)
    ap.add_argument('--target', required=True)
    ap.add_argument('--tw', type=int, required=True)
    ap.add_argument('--th', type=int, required=True)
    ap.add_argument('--frames', type=int, default=1)
    ap.add_argument('--out', required=True)
    ap.add_argument('--swapper', default='hyperswap_1a_256', choices=list(ref.SWAPPER_SPEC))
    args = ap.parse_args()

    models = ref.Models(args.swapper)

    source_frame = read_raw(args.source, args.sw, args.sh)[0]
    source_faces = ref.get_many_faces(models, source_frame)
    if not source_faces:
        sys.exit('no face in source')
    source_face = max(source_faces,
                      key=lambda f: (f.bounding_box[2] - f.bounding_box[0]) *
                                    (f.bounding_box[3] - f.bounding_box[1]))
    print('source: %d face(s), embedding norm %.4f'
          % (len(source_faces), numpy.linalg.norm(source_face.embedding)))

    frames = read_raw(args.target, args.tw, args.th, args.frames)
    calib = {}
    faces_total = 0
    for i in range(args.frames):
        # .copy() because paste_back writes in place and the array is a view into one
        # big buffer -- without it, frame i+1 would carry frame i's swapped face.
        frame = frames[i].copy()
        n_before = len(calib.get('arcface', []))
        ref.process_frame(models, source_face, frame, calib)
        n = len(calib.get('arcface', [])) - n_before
        faces_total += n
        print('  frame %2d/%d  %d face(s)' % (i + 1, args.frames, n))

    if not calib:
        sys.exit('captured nothing -- no faces found in any frame')

    os.makedirs(args.out, exist_ok=True)
    for name, tensors in calib.items():
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        for k, t in enumerate(tensors):
            numpy.ascontiguousarray(t, dtype=numpy.float32).tofile(
                os.path.join(d, '%s_%04d.raw' % (name, k)))
        print('calib %-14s %4d tensors  shape %s' % (name, len(tensors), tensors[0].shape))
    print('\n%d frames, %d faces -> %s' % (args.frames, faces_total, args.out))


if __name__ == '__main__':
    main()
