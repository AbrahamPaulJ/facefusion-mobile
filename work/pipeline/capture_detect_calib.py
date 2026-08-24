"""Capture real 640x640 detector inputs for yoloface calibration (trap #4).

Cheap: no landmarker, no recognizer, no swapper -- just the letterboxed detect frame that
detect_faces() feeds the network.
"""
import os, sys
import cv2, numpy
sys.path.insert(0, os.path.dirname(__file__))
import run_reference as R

video, outdir, stride, limit = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(outdir, exist_ok=True)
cap = cv2.VideoCapture(video)
i = kept = 0
while kept < limit:
    ok, frame = cap.read()
    if not ok:
        break
    if i % stride == 0:
        temp = R.restrict_frame(frame, R.FACE_DETECTOR_SIZE)
        det = numpy.zeros((R.FACE_DETECTOR_SIZE[1], R.FACE_DETECTOR_SIZE[0], 3))
        det[:temp.shape[0], :temp.shape[1], :] = temp
        det = det.transpose(2, 0, 1).astype(numpy.float32) / 255.0
        numpy.ascontiguousarray(det).tofile(os.path.join(outdir, 'yoloface_%04d.raw' % kept))
        kept += 1
    i += 1
cap.release()
print('wrote %d detect tensors of shape (3,640,640) to %s' % (kept, outdir))
