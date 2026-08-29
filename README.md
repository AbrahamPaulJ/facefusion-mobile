# FaceFusion Mobile

Offline face swapping on Android, running entirely on the phone's NPU.

Pick a source face and a target photo or video, trim the clip, and swap — with no server,
no account, and no upload. Every frame is processed on the device's Qualcomm Hexagon NPU.

This is a mobile port of [FaceFusion](https://github.com/facefusion/facefusion) by
Henry Ruhs. The pipeline, the models, and the option names, defaults and ranges are all
FaceFusion's.

## Features

- **Fully offline.** Nothing you select ever leaves the phone. The only network request the
  app makes is the one-time model download.
- **Before / after preview.** Scrub the trim handle and see the frame the swap will start
  from, and that same frame swapped, side by side at full width.
- **Real-time on-device processing.** About 19 ms per frame on a Snapdragon 8 Elite — a
  10-second 720p clip takes roughly 6 seconds.
- **Photos as well as video.** Pick a photo and the preview *is* the result, at the
  photo's own resolution — there is nothing to run, so there is no button to press.
- **Higher-resolution output.** Pixel boost renders the swapped face at 512, 768 or 1024
  instead of 256, at a proportional cost in time.
- **Optional face enhancer.** `gpen_bfr_256` restores detail in the swapped face. It is a
  separate 28 MB download from **Settings → Models**, and the app works without it.
- **Use it from a computer.** Turn on **Settings → Remote API** and open the address it
  shows in a browser on your PC: drop in a face and a target, and the phone does the work.
  There is an HTTP API behind that page for scripting.
- **FaceFusion's own controls.** Swapper weight, mask blur and padding, detector and
  landmarker thresholds, and every-face or largest-face selection.
- **Trim, live progress, cancel a run in progress, save to gallery, share.**

## Requirements

- Android 12 (API 31) or newer
- A **Qualcomm Snapdragon** phone with a Hexagon NPU

The app measures your chip at startup and downloads the matching build:

| tier | chips |
|---|---|
| `v68` | Snapdragon 888 and older, 8 Gen 1 (v69), or parts with under 8 MB VTCM |
| `v73` | 8 Gen 2 (v73), 8 Gen 3 (v75), and v79 parts other than the SM8750 |
| `v79` | Snapdragon 8 Elite (SM8750) |
| `v81` | Snapdragon 8 Elite Gen 5 and newer |

Devices without a Qualcomm NPU are not supported — there is no CPU fallback.

The NPU runtime for every Hexagon generation from v68 to v81 is bundled, so the app does
not need to download anything to talk to your chip.

> **Tested on:** Snapdragon 8 Elite (Galaxy S25 Ultra). The `v68` and `v73` builds run on
> the same code path but have not yet been exercised on real hardware of those
> generations. If you try one, **Settings → Share bug report** is the fastest way to tell
> me what happened.

## Install

1. Download the APK (~46 MB) from
   [Releases](https://github.com/AbrahamPaulJ/facefusion-mobile/releases).
2. Install it and open the app.
3. Tap **Download models**. The app fetches the ~275 MB set for your chip from
   [Hugging Face](https://huggingface.co/AbrahamPJ/facefusion-mobile-models-0.1.0).

The download is resumable and every file is checksum-verified, so an interrupted transfer
continues where it stopped rather than starting again. You will be warned before it starts
on a metered connection.

Models are not bundled in the APK: they are large, and their licences are not ours to
redistribute inside an application binary.

## Content policy

The app includes FaceFusion's content checker and it **blocks**. Material it flags is
refused, and a refusal produces no output file and nothing on the preview. If the checker
cannot run — for instance if its model is missing — the app refuses to process anything at
all rather than proceeding unchecked.

**Do not use this to create images or video of real people without their consent.** That is
the principal way software of this kind causes harm, and it is prohibited by the upstream
licence.

## Privacy

- No accounts, no analytics, no telemetry.
- Your photos and videos are never uploaded.
- Bug reports are assembled only when you tap **Share bug report**, contain no media, and
  are sent by whichever app you choose.

## Licence and attribution

Built on [FaceFusion](https://github.com/facefusion/facefusion), licensed
**OpenRAIL-AS** — which carries use restrictions. Read
[the licence](https://github.com/facefusion/facefusion/blob/master/LICENSE.md) before using,
modifying or redistributing this.

The neural models are converted from FaceFusion's and are **not uniformly permissive**:
`yoloface_8n` is GPL-3.0, `arcface_w600k_r50` and `inswapper_128` are Non-Commercial, and
`hyperswap_1a_256` is ResearchRAIL. See the
[model repository](https://huggingface.co/AbrahamPJ/facefusion-mobile-models-0.1.0) for
details.

The app icon is FaceFusion's, used with permission.
