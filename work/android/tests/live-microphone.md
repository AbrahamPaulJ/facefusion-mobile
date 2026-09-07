# Live microphone device checks

Requires an Android device with Live available and the face swap models installed.
The Live tab is enabled in the current fork.

1. Start Live with Microphone off. Record 10 seconds and save. Playback must be silent;
   `ffprobe -v error -show_streams live.mp4` must list only the video stream.
2. Enable Microphone and deny Android permission. The switch must stay off. Recording
   silent video must still work. Grant permission on a subsequent attempt (via Android
   settings if permission was permanently denied); enabling must not start voice capture.
3. Enable Microphone, record 30 seconds, speak and clap visibly at the beginning and end.
   Playback must contain the swapped video and audible speech with no growing sync drift.
   ffprobe must list H.264 video and mono AAC audio. Repeat with a slow face swap preset.
4. The Microphone switch must be locked during recording and finalization. After saving,
   switch it off and record again: the new file must have no audio stream.
5. Stop immediately after starting, before the first camera frame, and after a long clip.
   There must be no crash, unusable published MP4 or remaining microphone privacy indicator.
6. During recording, stop Live, switch lenses, leave the tab, and background the app in
   separate runs. Each must release the microphone and finalize its recording. Start again.
7. Start a Voice recording first, then try Live recording with Microphone enabled:
   it must report that Voice must stop. Voice cannot start while Live is recording/saving.
8. Revoke microphone permission in Android settings and return. Live must not claim to
   record audio without permission. Test Android's global microphone privacy switch too.
9. After successful, failed and discarded recordings, check the app cache for leaked
   `live_mic_*.m4a` and the output directory for `live_mux_*.mp4`: neither should remain.
   A content-gate refusal must not publish the recording.

These are device acceptance checks, not results of an automated test run.
