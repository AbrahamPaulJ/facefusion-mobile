# FaceFusion Mobile

Offline face swapping on Android. Pick a source face and a target photo or video, and the
swap runs entirely on your phone. It uses the Qualcomm Hexagon NPU where there is one, and
the GPU and CPU where there is not. No server, no account, nothing uploaded.

This is a mobile port of [FaceFusion](https://github.com/facefusion/facefusion) by Henry
Ruhs. The pipeline, the models, and the option names, defaults and ranges are FaceFusion's.

**Source face**

<img src="media/source-face.jpg" width="200" alt="Source face">

**Target**

<img src="media/swap-before.gif" width="260" alt="Target clip">

**Result**

<img src="media/swap-after.gif" width="260" alt="Swapped clip">

## What it does

- Swaps faces in photos and video, fully offline.
- Runs every neural network on the phone's Qualcomm Hexagon NPU. A 10 second 720p clip is
  processed in about 13 seconds on a Snapdragon 8 Elite, or 11 with **Fast video** on.
- Falls back to the GPU and CPU on phones without a Qualcomm NPU. Same result, about four
  times the time per frame.
- Shows the target frame and the swapped frame side by side before you commit to a run.
- **Play with live swap**: press it instead of Swap and watch the clip run through the
  pipeline as it plays, with its own sound, at whatever rate the phone manages. Nothing is
  written to disk — it is there to show you what a run would produce before you wait for
  one. Seek and pause as you would in any player.
- Load **several source faces** and switch between them from a row under the source pane.
  The row is the same on both screens, so a face picked on one is there on the other.
- Trims the clip, and under **Output settings** drops the frame rate or caps the size
  (1080p, 720p, 480p) for a smaller file and a faster run. Any run can be cancelled.
- **Fast video** (the gear on the `face_swapper` chip → Face detector): tracks the face
  between frames instead of finding it again every frame, for about 15% more speed. Off by
  default.
- **Live**: the front or back camera, swapped, in real time, and you can record what it
  shows. Enable **Microphone** before recording to include microphone audio in the MP4
  (Android asks for permission); leave it off for silent video. The switch is locked while
  recording and saving. Fast mode forces the settings a camera can keep up with; turning it
  off is experimental and asks first.
- **Assign per person**: give each person in the shot their own source face. On **Live**,
  enable the mode, choose a source, then tap a detected-person photo (`Person 1`, `Person 2`,
  and so on) or tap the person directly in the preview. The photos show the real camera faces,
  before swapping, and the row refreshes once when a new person enters the shot. On **Swap**,
  pick people from the row found in the frame you are looking at. Anyone you do not pick
  follows whichever source is selected.
- **Keep face**: pick it instead of a source and that person is left exactly as they were
  filmed — untouched by the swapper and by the enhancer. It is how you swap everyone in a
  shot except somebody.
- **Batch**: queue several clips and swap them all in one run, with one source face. Each
  finished clip gets a thumbnail you can tap, and can be saved to your gallery as it lands.
- **Pick the face**: tap a face on the target to swap only that person. The boxes appear on
  their own when there is more than one face in the frame.
- The Live preview is mirrored on the front camera, as a selfie camera is, and not on the
  back one. A switch overrides that per lens; when the switch is on, recordings use the same
  selfie-style horizontal flip.
- Takes its inputs from the camera and microphone as well as your files — shoot a source
  face, film a target, or record the voice that drives the lip syncer.
- Saves to your gallery, or hands a still straight out of the preview.
- Lip sync, optionally: redraws the mouth to match a voice you pick — a dub, a different
  take, any audio or video file — at about 1-2 ms per frame on the NPU. Play the voice back
  and trim it first, so only the part you want drives the mouth.
- Shows the swap as it is written: while a video runs, the result pane is the frame going
  into the file, not a re-render of it. The sound arrives with the finished clip.
- Speaks English, Русский, 简体中文 and 繁體中文.
- Follows your phone's light or dark theme, or pin one in Settings.
- Can be driven from a browser on your PC over the local network, if you turn that on.

Optional extras, each a separate download: a face enhancer (`gpen_bfr_256`, about 2.5 ms
more per face, 25 MB), a lip syncer (`edtalk_256`, 60 MB) and pixel boost, which renders the
swapped face at 512, 768 or 1024 instead of 256 at a proportional cost in time.

The photo row in **Assign per person** is a way to pick somebody without chasing a moving
face on the preview: the pictures are taken before anything is swapped, and the row is
refreshed once when another person enters the shot rather than rebuilt every frame. Whoever
you assigned keeps their source while the row changes around them.

The lip syncer needs a target video and a separate audio or video file to drive the mouth
from — a photo, a silent clip, or a target with no voice picked has nothing to sync to. It
takes a 200 ms window of that audio and redraws the mouth to match it. The model itself is
1-2 ms per frame; decoding the audio and turning it into a spectrogram costs roughly 28 ms
per second of audio, once per clip. It runs on the speech as recorded, with no attempt to
separate a voice from a music bed, so it is at its best on clean speech.

**The app running a swap**

<img src="media/app-run.gif" width="220" alt="The app running a swap">

## Requirements and supported Snapdragon chips

- Android 12 (API 31) or newer, 64 bit ARM.
- A Qualcomm Snapdragon phone for the NPU path. Other phones run the GPU and CPU path.

The app measures your chip at startup and downloads the matching build.

| Tier | Chips |
|---|---|
| `v68` | Snapdragon 888 and older, 8 Gen 1, or parts with under 8 MB VTCM |
| `v73` | 8 Gen 2, 8 Gen 3, and v79 parts other than the SM8750 |
| `v79` | Snapdragon 8 Elite (SM8750) |
| `v81` | Snapdragon 8 Elite Gen 5 |

The NPU runtime for every Hexagon generation from v68 to v81 ships inside the APK, so
nothing extra is downloaded to talk to your chip.

> Tested on a Snapdragon 8 Elite (Galaxy S25 Ultra). The other tiers run the same code and
> are verified for accuracy against the v79 build, but each has had little or no time on
> real hardware of its own generation. If you run one, **Settings → Share bug report** is
> the most useful thing you can send.

## Install the APK

1. Download the APK (about 66 MB) from
   [Releases](https://github.com/AbrahamPaulJ/facefusion-mobile/releases).
2. Install it and open the app.
3. Tap **Download models**. The app fetches the roughly 420 MB set for your chip from
   [Hugging Face](https://huggingface.co/AbrahamPJ/facefusion-mobile-models).

The download resumes if it is interrupted, and every file is checked against a SHA256 hash
before it is used. You are warned before it starts on a metered connection.

Models are not bundled in the APK. They are large, and their licences are not ours to
redistribute inside an application binary.

## Speed: NPU vs GPU and CPU

Measured on a Snapdragon 8 Elite, same clip through the same app.

| | Hexagon NPU | GPU (Vulkan) + CPU |
|---|---:|---:|
| Per frame, 720p | 44 ms, or 38 with Fast video | 325 ms |
| Model load, first run | 0.5 s | 15 s |
| Same output | | yes, to 42.7 dB |

The NPU figure is measured on the current release. The GPU one was measured before the CPU
geometry was rewritten, and that work is shared by both paths, so expect it to be lower than
this table says — it has not been re-measured since.

The GPU and CPU path produces the same swap rather than an approximation of it. Two stages
stay on the CPU whatever the phone has, because on the GPU they come out measurably wrong:
the content checker and the face enhancer.

Graphics drivers vary, and on some phones the rest of it comes out wrong too -- the clearest
sign is the app finding faces in walls and furniture. So before it uses the GPU at all, the
app runs the face detector on both the GPU and the CPU over the same frame and compares the
two; if they disagree it stays on the CPU and says so on the Settings **Device** panel. You
can also decide yourself, under **Graphics chip** on that panel: Automatic, GPU or CPU. CPU
is slower and always works.

## Remote API

Turn on **Settings → Remote API** and open the address it shows in a browser on your
computer. Drop in a face and a target, and the phone does the work. There is an HTTP API
behind that page if you want to script it. It is off by default and bound to loopback until
you say otherwise.

## Roadmap

- Show the faces detected in the target, so you can see what will be swapped.
- Live portrait, for stills only. The generator costs nine times the entire current frame,
  which turns a 10 second clip into about four minutes.

## Content policy

The app includes FaceFusion's content checker and it blocks. Flagged material is refused,
with no output file and nothing shown. Every path that processes an image is checked: the
source face and every other source face you have loaded, the target photo or video, the
preview, the live player, the Live camera, every clip in a batch and the remote API. If the checker cannot run, for instance if its model is missing, the app
refuses to process anything rather than continuing unchecked.

**Do not use this on real people without their consent.** That is the main way software of
this kind causes harm, and it is prohibited by the upstream licence.

## Privacy

No accounts, no analytics, no telemetry. Your photos and videos are never uploaded. Bug
reports are assembled only when you tap **Share bug report**, contain no media, and go
wherever you choose to send them.

## Licence and attribution

Built on [FaceFusion](https://github.com/facefusion/facefusion), licensed **OpenRAIL-AS**,
which carries use restrictions. Read
[the licence](https://github.com/facefusion/facefusion/blob/master/LICENSE.md) before
using, modifying or redistributing this.

The models are converted from FaceFusion's and are not uniformly permissive.
`yoloface_8n` is GPL-3.0, `arcface_w600k_r50` and `inswapper_128` are Non-Commercial, and
`hyperswap_1a_256` is ResearchRAIL. See the
[model repository](https://huggingface.co/AbrahamPJ/facefusion-mobile-models) for details.

The app icon is FaceFusion's, used with permission.

## Authors

Written and maintained by **[@AbrahamPaulJ](https://github.com/AbrahamPaulJ)** — the NPU
port and the model conversions, the hand-written geometry, the pipeline, the app, the batch
runner, the lip syncer, the Live tab, the live player and the remote API.

Outside contributions, with thanks:

- **[@doctormajid7-ux](https://github.com/doctormajid7-ux)** — microphone audio in Live
  recordings and the first version of **Assign per person** on Live ([#1]), then the ideas
  behind several source faces on the Swap screen, per-person assignment there, the option to
  leave a person's own face alone, and a Live mirror independent of the lens ([#2], [#4]).
- **[@aaazhouaa](https://github.com/aaazhouaa)** — the monochrome light/dark theme and the
  theme setting, the Russian and Chinese translations, batch thumbnails, the collapsible
  log, and a preview performance fix ([#3]).

[#1]: https://github.com/AbrahamPaulJ/facefusion-mobile/pull/1
[#2]: https://github.com/AbrahamPaulJ/facefusion-mobile/pull/2
[#3]: https://github.com/AbrahamPaulJ/facefusion-mobile/pull/3
[#4]: https://github.com/AbrahamPaulJ/facefusion-mobile/pull/4

## Support

This is built and maintained in spare time, on hardware bought for it, and it is free with
no accounts, no analytics and nothing to pay for. If it has been useful to you and you would
like to put something towards it:

**[buymeacoffee.com/abrahampaulj](https://buymeacoffee.com/abrahampaulj)**

Bug reports and pull requests are worth just as much. The fastest way to get something
fixed is **Share bug report** in Settings, which assembles the device details and the run
log and contains no media.
