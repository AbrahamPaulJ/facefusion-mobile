# FaceFusion Mobile

Offline face swapping on Android, running entirely on the phone's NPU.

Pick a source face and a target photo or video, trim the clip, and swap — with no server,
no account, and no upload. Every frame is processed on the device's Qualcomm Hexagon NPU.

This is a mobile port of [FaceFusion](https://github.com/facefusion/facefusion) by
Henry Ruhs. The pipeline, the models, and the option names, defaults and ranges are all
FaceFusion's.

## Features

- **Runs without a Qualcomm NPU too, as of 0.4.0.** Phones with no Hexagon fall back to the
  GPU (Vulkan) with the CPU underneath — the same result, about four times the time per
  frame. A preview: see the roadmap.
- **English, 简体中文 and 繁體中文.** Follows your phone, or pick a language just for this
  app in Android's own app settings. Model names and chip tiers stay in English on purpose —
  they are what FaceFusion's documentation and every bug report use.
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
- **Optional face enhancer.** `gpen_bfr_256` restores detail in the swapped face, for about
  2.5 ms more per face. It is a ~25 MB download from **Settings → Models**, and the app
  works without it.
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
| `v81` | Snapdragon 8 Elite Gen 5 |

Devices without a Qualcomm NPU fall back to the **GPU + CPU** path added in 0.4.0. It is a
preview: it produces the same result, at roughly four times the time per frame, and it has
not yet been run on a non-Qualcomm phone. See the roadmap below.

The NPU runtime for every Hexagon generation from v68 to v81 is bundled, so the app does
not need to download anything to talk to your chip.

> **Tested on:** Snapdragon 8 Elite (Galaxy S25 Ultra). The `v68`, `v73` and `v81` builds
> run the same code path and are verified against the `v79` build for accuracy, but none of
> them has been exercised on real hardware of its own generation. The `v81` tier in
> particular is new in 0.2.1 and no 8 Elite Gen 5 has yet run it. If you try one,
> **Settings → Share bug report** is the fastest way to tell me what happened.

## Roadmap

**~~A native v81 build~~ — shipped in 0.2.1.** v81 phones used to fall back to the `v73`
build, two generations old on that silicon, which is why owners reported around 9 fps.
The `v81` tier is now published. It has not been run on an 8 Elite Gen 5 yet; reports very
welcome.

**GPU and CPU support, for phones without a Qualcomm NPU — now in the app, as a preview.**
The most requested thing here for a year. It is built on
[ncnn](https://github.com/Tencent/ncnn), running on the GPU through Vulkan with the CPU
underneath, and 0.4.0 is the first release that carries it.

Measured on a Snapdragon 8 Elite, running the same clip through the same app twice:

| | Hexagon NPU | GPU (Vulkan) + CPU |
|---|---:|---:|
| per frame, 720p | **~75 ms** | ~325 ms |
| same output? | — | **yes, to 42.7 dB** |

So expect **about four times longer** — a 10-second clip in a couple of minutes rather than
twenty seconds — and the result is the same swap, not an approximation of it. Two things
run on the CPU no matter what the phone has, because on the GPU they come out measurably
wrong rather than merely slower: the content checker and the face enhancer. Loading the
models takes about fifteen seconds the first time, against half a second on the NPU.

⚠ **It is a preview and it is honest about that.** It has been exercised on one phone — a
Qualcomm one, with its NPU deliberately switched off — because that is the hardware
available here. No Mali, Exynos or Tensor device has run it yet. The model set is a separate
~600 MB download. If you try it on a phone without a Snapdragon, **Settings → Share bug
report** is the most useful thing you can send.

**Lip sync.** Drive the mouth from an audio track, as FaceFusion's lip syncer does. The
model itself is the cheapest thing this project has looked at — 4 GMAC against the 67 the
swap already runs, about a millisecond a frame — and it needs nothing the NPU cannot
already do. The work is everywhere else: the app currently copies a video's audio through
untouched and never decodes a single sample, and lip sync needs the waveform, resampled and
turned into a spectrogram on the phone, in step with the trim you set. So: plausible, and
not small.

**Live portrait / expression restore — measured, and it does not fit yet.** Transferring
expression rather than identity. All six of FaceFusion's LivePortrait graphs were downloaded
and costed before this line was written, and the generator alone is **9x the entire current
frame** — around 170 ms per frame at best, turning a 10-second clip from 21 seconds into
about four minutes. Nothing about it is unsupported; it is simply that expensive. It could
be a *photo* feature at that price and it cannot be a video one, so if it appears at all it
will appear for stills first.

## Install

1. Download the APK (~66 MB) from
   [Releases](https://github.com/AbrahamPaulJ/facefusion-mobile/releases).
2. Install it and open the app.
3. Tap **Download models**. The app fetches the ~300 MB set for your chip from
   [Hugging Face](https://huggingface.co/AbrahamPJ/facefusion-mobile-models).

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
[model repository](https://huggingface.co/AbrahamPJ/facefusion-mobile-models) for
details.

The app icon is FaceFusion's, used with permission.
