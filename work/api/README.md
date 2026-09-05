# The remote API — swap from a PC

The app serves its own swap over HTTP so another machine can drive the phone's NPU. Off by
default; **Settings → Remote API** turns it on.

## Why it is in the app and not a native binary

Local Dream puts all of its inference in a standalone native server and reduces its app to
a launcher, so any client that can exec the binary replaces the UI. That is the right shape
*there*, and the wrong one here, for three reasons that are specific to this port:

| | |
|---|---|
| **The content gate's policy is Kotlin** | `NativePipe.contentScore` is native, but the thresholds, the video sampling and the refusal live in `ContentGate.kt`. A native server would reimplement them, and the first thing it got wrong would be a processing path with no gate on it, reachable over the network. |
| **The video path is Android's** | `VideoSwapper` is MediaExtractor + MediaCodec + MediaMuxer. There is no encoder in this tree to reimplement it against. |
| **The models are the app's** | They live in the app's external files dir, downloaded and SHA256-verified by the app, and a shell-owned copy of them is unreadable to it (see the `modelDir()` comment). |

So `ApiServer.kt` runs inside the app and calls exactly what the screen calls. One job at a
time, enforced by `PipeGuard`, because `g_pipe` is a single global on the C++ side.

## Connecting

**Over USB** — the default, and the safe one. The server binds loopback, so nothing on the
network can reach it; `adb forward` is the tunnel.

```powershell
adb forward tcp:8760 tcp:8760
adb shell am start -n com.facefusion.mobile/.MainActivity --es api start   # or the switch
```

**Over the network** — Settings → *Reachable from the network*. Binds `0.0.0.0`; the
address to use is shown under it. The `--es api start` intent can **not** turn this on: the
Activity is exported, so an extra that opened the port to the network would be a remote
exposure switch any app on the phone could flip without the user seeing it. With no token,
this switch is the whole of the access control.

Tailscale is the good version of this: the phone gets a stable address reachable from your
own machines and nothing else. ⚠ Transfers over it drop above ~40 MB (trap in `CLAUDE.md`),
which is a reason to prefer `video` over `upload` for anything long.

## No auth

There is no token. The protection is the bind address:

- **Loopback (default)** — reachable only from the phone itself, or through `adb forward`,
  which needs USB or an authorised adb connection. A machine that can do that could install
  an APK on the phone anyway.
- **Network** — the switch in Settings binds `0.0.0.0`, and then anyone who can reach the
  phone on port 8760 can swap faces with it. Leave it off on networks you do not own;
  Tailscale, where the phone is reachable from your own machines and nothing else, is the
  version of this worth using.

The `--es api start` intent can only ever start the loopback server, so no app on the phone
can quietly open the port to the network.

## Endpoints

| | | |
|---|---|---|
| `GET /health` | — | JSON: tier, missing models, whether a source is set, who holds the NPU |
| `POST /source` | image bytes | Sets the face to swap **from**. Gated, then embedded. Kept until replaced. |
| `POST /swap` | image bytes | `image/png` of that image with the face swapped in |
| `POST /swap_video` | mp4 bytes | `video/mp4`, the whole clip. Blocks; capped at 512 MB. |

Bodies are raw bytes, not multipart — the client is `curl --data-binary @file`.

Options default to whatever the app's processor settings are set to, per-request overrides in the
query string: `weight`, `blur`, `detector`, `boost` (1-4), `largest`, `enhancer`,
`enhance_blend`, `fps`, `swapper`, `lip_sync`.

`lip_sync=1` applies only to `/swap_video`, and only when the clip HAS AN AUDIO TRACK: the
lip syncer redraws the mouth to match speech, so a silent clip and a still image have
nothing to condition on. It is a separate model download like the enhancer; when it is
absent, or the clip is silent, the run logs a reason and continues as a plain swap rather
than failing. `GET /health` lists missing models.

```powershell
curl --data-binary "@face.jpg" http://127.0.0.1:8760/source
curl --data-binary "@photo.jpg" "http://127.0.0.1:8760/swap?boost=2&enhancer=1" -o out.png
```

### Statuses that mean something specific

| | |
|---|---|
| `403` | the content gate refused. The body carries the sentence and the verdict. |
| `409` | no source set yet — `POST /source` first |
| `422` | no face in that frame. The buffer is returned untouched by `processFrame`, so this is an explicit answer rather than an unswapped image with a 200 on it. |
| `500` | something threw. The body names the exception TYPE as well as its message, because MediaCodec's carries an empty one and arrived as `{"error":""}` |
| `503` | the screen is using the NPU, or the models are not on the device (`missing` lists them) |

## The client

`ffclient.py` — no dependencies, `urllib` only.

```
py -3.10 ffclient.py health
py -3.10 ffclient.py source face.jpg
py -3.10 ffclient.py swap photo.jpg out.png --boost 2
py -3.10 ffclient.py video clip.mp4 out.mp4     # ffmpeg here, frames over the wire
py -3.10 ffclient.py upload clip.mp4 out.mp4    # whole file to the phone, <= 64 MB
```

`video` vs `upload`: the swap is identical, both end in the same `processFrame`. `upload` is
one request and the phone does the demux and mux; it gives no progress until it finishes.
`video` keeps ffmpeg on this side, so it prints frames per second and an ETA, and a frame
with no face keeps the original rather than shortening the clip and desyncing the audio.

Measured: 432 frames at 478x850 through `upload` took **5.7 s** on an SM8750, 13 ms a frame.
The upload streams to disk as it arrives and the result streams back off disk -- an earlier
version held the request body, a copy of it and the finished file in the heap at once,
alongside ~300 MB of context binaries, and the process was killed mid-job. From the client
that is indistinguishable from the server hanging up.

## Notes

- The pipeline stays warm between requests. It reloads only when the options change or when
  the screen took the NPU in between — detected by `PipeGuard.sequence`, so no caller has to
  remember to announce an init.
- A `/swap` request during a preview refresh waits up to 15 s rather than failing; a preview
  during an API request waits 4 s. Neither queues behind a video run: that gets a 503, which
  is the honest answer for something that can take minutes.
- The source is re-embedded automatically if the screen borrowed the pipeline, so the client
  never has to resend the face.
- Video dimensions are the encoder's business, not ours: the size is checked against
  `MediaCodecInfo.VideoCapabilities` and cropped down to its alignment. Below its minimum
  (128x128 on this part) the clip is refused with those numbers in the sentence, rather than
  a `CodecException` with nothing in it.
