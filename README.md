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
- Runs every neural network on the phone's Qualcomm Hexagon NPU, at about 19 ms per frame
  on a Snapdragon 8 Elite. A 10 second 720p clip takes roughly 6 seconds of NPU time.
- Falls back to the GPU and CPU on phones without a Qualcomm NPU. Same result, about four
  times the time per frame.
- Shows the target frame and the swapped frame side by side before you commit to a run.
- Trims the clip, drops the frame rate if you want it faster, and lets you cancel mid run.
- Saves to your gallery, or hands a still straight out of the preview.
- Lip sync, optionally: redraws the mouth to match the speech on the clip's own audio
  track, at about 1 ms per frame on the NPU.
- Speaks English, 简体中文 and 繁體中文.
- Can be driven from a browser on your PC over the local network, if you turn that on.

Optional extras, each a separate download: a face enhancer (`gpen_bfr_256`, about 2.5 ms
more per face, 25 MB), a lip syncer (`wav2lip_gan_96`, 44 MB) and pixel boost, which renders
the swapped face at 512, 768 or 1024 instead of 256 at a proportional cost in time.

The lip syncer needs a video with sound. It takes a 200 ms window of the audio and redraws
the mouth to match it, so a photo or a silent clip has nothing to sync to. The model itself
is about 1 ms per frame; decoding the audio and turning it into a spectrogram costs roughly
28 ms per second of audio, once per clip. It runs on the speech as recorded, with no attempt
to separate a voice from a music bed, so it is at its best on clean speech.

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
3. Tap **Download models**. The app fetches the roughly 360 MB set for your chip from
   [Hugging Face](https://huggingface.co/AbrahamPJ/facefusion-mobile-models).

The download resumes if it is interrupted, and every file is checked against a SHA256 hash
before it is used. You are warned before it starts on a metered connection.

Models are not bundled in the APK. They are large, and their licences are not ours to
redistribute inside an application binary.

## Speed: NPU vs GPU and CPU

Measured on a Snapdragon 8 Elite, same clip through the same app.

| | Hexagon NPU | GPU (Vulkan) + CPU |
|---|---:|---:|
| Per frame, 720p | 75 ms | 325 ms |
| Model load, first run | 0.5 s | 15 s |
| Same output | | yes, to 42.7 dB |

The GPU and CPU path produces the same swap rather than an approximation of it. Two stages
stay on the CPU whatever the phone has, because on the GPU they come out measurably wrong:
the content checker and the face enhancer.

## Remote API

Turn on **Settings → Remote API** and open the address it shows in a browser on your
computer. Drop in a face and a target, and the phone does the work. There is an HTTP API
behind that page if you want to script it. It is off by default and bound to loopback until
you say otherwise.

## Roadmap

- Get the GPU and CPU path onto a phone that actually needs it. It has only ever run on a
  Snapdragon with the NPU switched off.
- Show the faces detected in the target, so you can see what will be swapped.
- Live portrait, for stills only. The generator costs nine times the entire current frame,
  which turns a 10 second clip into about four minutes.

## Content policy

The app includes FaceFusion's content checker and it blocks. Flagged material is refused,
with no output file and nothing shown. If the checker cannot run, for instance if its model
is missing, the app refuses to process anything rather than continuing unchecked.

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
`yoloface_8n` is GPL-3.0, `arcface_w600k_r50`, `inswapper_128` and `wav2lip_gan_96` are
Non-Commercial, and `hyperswap_1a_256` is ResearchRAIL. See the
[model repository](https://huggingface.co/AbrahamPJ/facefusion-mobile-models) for details.

The app icon is FaceFusion's, used with permission.
