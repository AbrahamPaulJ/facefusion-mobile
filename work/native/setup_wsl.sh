#!/usr/bin/env bash
# Host-side verification environment for the native geometry port.
#
# The C++ geometry ops are compared against OpenCV in ONE process via ctypes, so the
# comparison is exact rather than mediated by files.  That needs cv2 inside WSL, where
# the .so is built.
set -uo pipefail
source "$HOME/npuconvert/qnn_env.sh"
pip install --quiet opencv-python-headless 2>&1 | tail -2
python - <<'PY'
import cv2, numpy
print("wsl cv2", cv2.__version__, "numpy", numpy.__version__)
PY
