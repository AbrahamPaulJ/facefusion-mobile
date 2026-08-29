#!/usr/bin/env python3
"""
Drive the phone's NPU from this machine.

The app serves the swap over HTTP (see ApiServer.kt). This is the client: it sets the
source face once, then swaps stills, or a whole video frame by frame.

    # phone on USB, server started from Settings or with:
    #   adb shell am start -n com.facefusion.mobile/.MainActivity --es api start
    adb forward tcp:8760 tcp:8760

    py -3.10 ffclient.py health
    py -3.10 ffclient.py source face.jpg
    py -3.10 ffclient.py swap photo.jpg out.png
    py -3.10 ffclient.py video clip.mp4 out.mp4              # frame by frame, needs ffmpeg
    py -3.10 ffclient.py upload clip.mp4 out.mp4             # hand the whole file to the phone

There is no auth. The server binds loopback unless the switch in Settings says otherwise,
so the only way in is from the phone itself or through `adb forward` -- which needs USB or
an authorised adb connection, i.e. a machine that could install an APK anyway.

WHY TWO VIDEO MODES. `upload` is one request and the phone does everything, which is the
simplest thing that works -- but the clip has to fit in the request body (64 MB) and in the
phone's heap, and you get no progress until it finishes. `video` keeps the demux and the
mux here, where ffmpeg already is, and sends only frames; it handles any length, shows
progress, and survives a failure halfway with the frames it already has. Neither is more
correct: the swap is identical, because both end up in the same processFrame.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_HOST = "http://127.0.0.1:8760"


def request(host, path, body=None, method=None, timeout=900):
    """One request. Returns (status, content_type, bytes)."""
    url = host.rstrip("/") + path
    req = urllib.request.Request(url, data=body, method=method or ("POST" if body else "GET"))
    if body is not None:
        req.add_header("Content-Type", "application/octet-stream")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()
    except urllib.error.URLError as e:
        raise SystemExit(
            "cannot reach %s (%s).\n"
            "  - is the server on?  Settings > Remote API, or\n"
            "      adb shell am start -n com.facefusion.mobile/.MainActivity --es api start\n"
            "  - forwarded?         adb forward tcp:8760 tcp:8760" % (url, e.reason))


def die(status, ctype, body):
    """The server's errors are JSON with a sentence in them; show the sentence."""
    msg = body[:400].decode("utf-8", "replace")
    if "json" in ctype:
        try:
            msg = json.loads(body).get("error", msg)
        except Exception:
            pass
    raise SystemExit("HTTP %d: %s" % (status, msg))


def cmd_health(a):
    st, ct, body = request(a.host, "/health")
    if st != 200:
        die(st, ct, body)
    print(json.dumps(json.loads(body), indent=2))


def cmd_source(a):
    st, ct, body = request(a.host, "/source", open(a.image, "rb").read())
    if st != 200:
        die(st, ct, body)
    print("source set:", json.loads(body))


def cmd_swap(a):
    t0 = time.time()
    st, ct, body = request(a.host, "/swap" + qs(a), open(a.image, "rb").read())
    if st != 200:
        die(st, ct, body)
    open(a.out, "wb").write(body)
    print("%s  %d KB  %.2f s" % (a.out, len(body) // 1024, time.time() - t0))


def cmd_upload(a):
    data = open(a.video, "rb").read()
    mb = len(data) / 1048576.0
    if mb > 64:
        raise SystemExit("%.1f MB is over the server's 64 MB body limit -- use `video`" % mb)
    print("uploading %.1f MB, then waiting (a minute of 30 fps video is ~1800 frames)..." % mb)
    t0 = time.time()
    st, ct, body = request(a.host, "/swap_video" + qs(a), data)
    if st != 200:
        die(st, ct, body)
    open(a.out, "wb").write(body)
    print("%s  %.1f MB  %.1f s" % (a.out, len(body) / 1048576.0, time.time() - t0))


def cmd_video(a):
    """Demux here, swap there, mux here. ffmpeg does both ends."""
    ff = shutil.which("ffmpeg")
    if not ff:
        raise SystemExit("ffmpeg is not on PATH; `upload` needs no ffmpeg but caps at 64 MB")
    work = os.path.abspath(a.workdir)
    raw, done = os.path.join(work, "in"), os.path.join(work, "out")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(done, exist_ok=True)

    fps = a.fps or probe_fps(ff, a.video)
    print("splitting at %g fps..." % fps)
    run([ff, "-y", "-loglevel", "error", "-i", a.video,
         os.path.join(raw, "%06d.png")])
    frames = sorted(os.listdir(raw))
    if not frames:
        raise SystemExit("ffmpeg produced no frames")

    t0 = time.time()
    skipped = 0
    for i, name in enumerate(frames, 1):
        st, ct, body = request(a.host, "/swap" + qs(a),
                               open(os.path.join(raw, name), "rb").read())
        if st == 422:
            # No face in this frame. Keep the original: dropping it would shorten the clip
            # and shift everything after it out of sync with the audio.
            shutil.copyfile(os.path.join(raw, name), os.path.join(done, name))
            skipped += 1
        elif st != 200:
            die(st, ct, body)
        else:
            open(os.path.join(done, name), "wb").write(body)
        if i % 5 == 0 or i == len(frames):
            el = time.time() - t0
            print("\r  %d/%d  %.1f fps  eta %ds  (%d without a face)"
                  % (i, len(frames), i / el, int((len(frames) - i) / (i / el)), skipped),
                  end="", flush=True)
    print()

    print("muxing...")
    cmd = [ff, "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", os.path.join(done, "%06d.png")]
    # Carry the original audio when there is any; -shortest keeps a rounding difference in
    # frame count from stretching the file.
    if has_audio(ff, a.video):
        cmd += ["-i", a.video, "-map", "0:v", "-map", "1:a", "-c:a", "copy", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", a.out]
    run(cmd)
    print("%s  %.1f s total" % (a.out, time.time() - t0))
    if not a.keep:
        shutil.rmtree(raw, ignore_errors=True)
        shutil.rmtree(done, ignore_errors=True)


def qs(a):
    """Per-request option overrides, the same names ApiServer accepts."""
    parts = []
    for k in ("weight", "blur", "boost", "enhancer", "largest", "swapper", "detector"):
        v = getattr(a, k, None)
        if v is not None:
            parts.append("%s=%s" % (k, v))
    return ("?" + "&".join(parts)) if parts else ""


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("%s failed:\n%s" % (cmd[0], p.stderr[-2000:]))


def probe_fps(ff, path):
    probe = shutil.which("ffprobe")
    if not probe:
        return 30.0
    p = subprocess.run([probe, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        num, den = p.stdout.strip().split("/")
        return round(float(num) / float(den), 3)
    except Exception:
        return 30.0


def has_audio(ff, path):
    probe = shutil.which("ffprobe")
    if not probe:
        return False
    p = subprocess.run([probe, "-v", "error", "-select_streams", "a", "-show_entries",
                        "stream=index", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return bool(p.stdout.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("FF_HOST", DEFAULT_HOST))
    for name in ("weight", "blur", "detector"):
        ap.add_argument("--" + name, type=float)
    ap.add_argument("--boost", type=int, choices=(1, 2, 3, 4))
    ap.add_argument("--enhancer", type=int, choices=(0, 1))
    ap.add_argument("--largest", type=int, choices=(0, 1))
    ap.add_argument("--swapper", choices=("hyperswap", "inswapper"))

    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("health").set_defaults(fn=cmd_health)

    p = sub.add_parser("source", help="set the face to swap FROM")
    p.add_argument("image")
    p.set_defaults(fn=cmd_source)

    p = sub.add_parser("swap", help="swap one still")
    p.add_argument("image")
    p.add_argument("out", nargs="?", default="swapped.png")
    p.set_defaults(fn=cmd_swap)

    p = sub.add_parser("video", help="frame by frame, via ffmpeg here")
    p.add_argument("video")
    p.add_argument("out", nargs="?", default="swapped.mp4")
    p.add_argument("--fps", type=float)
    p.add_argument("--workdir", default="ffwork")
    p.add_argument("--keep", action="store_true", help="leave the frame directories")
    p.set_defaults(fn=cmd_video)

    p = sub.add_parser("upload", help="hand the whole clip to the phone (<= 64 MB)")
    p.add_argument("video")
    p.add_argument("out", nargs="?", default="swapped.mp4")
    p.set_defaults(fn=cmd_upload)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
