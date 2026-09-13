package com.facefusion.mobile

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaPlayer
import android.os.SystemClock
import java.nio.ByteBuffer

/**
 * The TARGET clip, swapped while it plays, with its own sound -- roadmap-adjacent, dev only.
 *
 * The Live tab does this for the camera. This does it for a FILE, which is the same pump
 * with two differences that are the whole of the design:
 *
 *  1. A camera hands you frames at its own rate and you show whatever you finish. A file
 *     hands you frames as fast as you ask, so something has to decide WHEN each one is due
 *     -- and when the pipeline cannot keep up, which ones to skip. That is [clockMs] and
 *     the drop test in [pump].
 *  2. A file has audio. The sound is the thing you cannot slow down, so it is the CLOCK
 *     rather than a passenger: video chases audio, never the other way round.
 *
 * ⚠ FRAMES ARE DROPPED BEFORE THE SWAP, never after. Dropping after would spend 30-60 ms of
 * NPU on a frame nobody sees, which makes the next one late too -- the failure mode where a
 * player that is slightly behind gets further behind until it stops moving. The test is one
 * comparison against the clock and it happens while the frame is still a decoder buffer, so
 * a dropped frame costs a `releaseOutputBuffer` and nothing else. This is what "just
 * decrease the fps" means mechanically: nothing lowers a frame rate, the late frames simply
 * do not get processed, and what is left plays in time.
 *
 * ⚠ It OWNS the native pipeline while it runs, exactly as Live does -- see
 * `MainActivity.startPlayer`, which acquires [PipeGuard] and calls `NativePipe.init` in the
 * same order `startLive` does. One pump at a time is not a throughput choice: `g_pipe` is a
 * single global and two callers is a use-after-free.
 *
 * The audio is a bare [MediaPlayer] with NO surface, which plays the sound and renders no
 * video. Decoding the audio track by hand into an AudioTrack would remove a decoder we do
 * not otherwise need, and would also be ~150 lines to arrive at the clock MediaPlayer
 * already exposes through [MediaPlayer.getCurrentPosition]. The cheap version wins until
 * there is a reason it cannot.
 */
class LivePlayer {

    /**
     * One pump result.
     *
     * ⚠ A null [bitmap] means "no new picture", NOT "blank the screen" -- it is what a
     * dropped frame reports so the scrub bar still advances. The UI keeps the last frame it
     * was given. Blanking on a drop is a flicker at exactly the moment the player is
     * already struggling.
     */
    data class Shot(
        val bitmap: Bitmap?,
        val positionMs: Int,
        /** Frames actually SWAPPED per second, averaged over the last second. */
        val fps: Double,
        val faces: Int,
        /** Decoded and skipped because the swap could not have been on time. */
        val dropped: Int,
        val error: String? = null,
        /** The clip ran out. The UI parks the transport at the end rather than looping. */
        val ended: Boolean = false,
        /**
         * A seek is being resolved: the picture on screen is the OLD one and the sound is
         * deliberately silent. Worth saying out loud, because resolving one is not
         * instantaneous and a still picture with no sound is otherwise indistinguishable
         * from a hang.
         */
        val seeking: Boolean = false,
    )

    /** Clip length in ms, from the container. 0 until [start] has opened the file. */
    @Volatile var durationMs: Int = 0
        private set

    /** Whether the pump is running at all -- not whether it is [playing]. */
    @Volatile var active: Boolean = false
        private set

    @Volatile private var playing = false
    @Volatile private var stopping = false
    @Volatile private var seekTargetMs = -1
    private var thread: Thread? = null

    private var audio: MediaPlayer? = null
    private var audioOk = false

    /**
     * A seek has been asked of the audio and has not landed yet.
     *
     * ⚠ `MediaPlayer.seekTo` is ASYNCHRONOUS. It returns at once and `currentPosition`
     * goes on reporting the OLD position until the seek completes -- so for a few tens of
     * milliseconds after every scrub the clock names a part of the film the pump is no
     * longer decoding. Believing it there is what made seeking stutter and stall: seek
     * FORWARD and every freshly decoded frame looks far too early, so the pump sleeps on
     * each one up to MAX_WAIT_MS; seek BACKWARD and every one looks far too late, so the
     * pump drops the lot and the picture simply stops.
     */
    @Volatile private var audioSeeking = false

    /** The audio track ran out. It stops being a clock at that point; wall time takes over. */
    @Volatile private var audioEnded = false

    /** The VIDEO ran out. Play means REPLAY from here -- see [play]. */
    @Volatile private var endReached = false

    /**
     * Start the sound the moment its seek lands, because playback is meant to be running.
     *
     * The sound is NOT started by whoever asked for the seek: it is started by the seek
     * completing, at the frame the video actually settled on. See [settleAudioAt].
     */
    @Volatile private var startAudioWhenSeeked = false

    // The clock, when there is no audio track to be one. `baseMs` is the position at the
    // last resume and `baseWall` the wall time it happened at; between them they survive
    // pausing, which a plain start-time cannot.
    @Volatile private var baseMs = 0
    @Volatile private var baseWall = 0L

    /**
     * Where the film is NOW, in ms.
     *
     * The audio's own position when there is audio: a sound card runs at its own rate and
     * anything that thinks it knows better produces drift you can hear. Wall time otherwise.
     */
    private fun clockMs(): Int {
        audioClockMs()?.let { a ->
            // Re-base the wall clock off the sound on every reading it is good for, so the
            // moment it stops being good -- a seek in flight, a short audio track that has
            // already run out -- the fallback carries on from where the sound actually was
            // rather than from whenever it was last started.
            baseMs = a
            baseWall = SystemClock.elapsedRealtime()
            return a
        }
        return if (playing) (baseMs + (SystemClock.elapsedRealtime() - baseWall)).toInt()
               else baseMs
    }

    /**
     * The audio's position, or null when the audio is not currently a clock.
     *
     * The two cases it is not: a seek that has not landed (the position is still the one
     * before the scrub), and a track that has finished (the position stops advancing while
     * the video has minutes left). Both used to read as a confident wrong answer.
     */
    private fun audioClockMs(): Int? {
        if (!audioOk || audioSeeking || audioEnded) return null
        return runCatching { audio?.currentPosition }.getOrNull()
    }

    /** Where the transport should say it is, clamped to the clip. */
    val positionMs: Int get() = clockMs().coerceIn(0, durationMs.coerceAtLeast(0))

    val isPlaying: Boolean get() = playing

    /**
     * Open [path] and start pumping. [onShot] is called from the decode thread.
     *
     * Snapshot state is safe to write from any thread, which is how [LiveEngine] hands the
     * camera's results to Compose, so this follows it rather than posting to the main
     * looper per frame.
     */
    fun start(path: String, onShot: (Shot) -> Unit) {
        stop()
        stopping = false
        seekTargetMs = -1
        appliedSeek = -1
        audioSeeking = false
        audioEnded = false
        endReached = false
        startAudioWhenSeeked = false
        baseMs = 0
        baseWall = SystemClock.elapsedRealtime()
        durationMs = durationOf(path)
        audioOk = openAudio(path)
        active = true
        thread = Thread({ pump(path, onShot) }, "ffplayer").apply { start() }
    }

    fun play() {
        if (!active) return
        // ⚠ PLAY AT THE END MEANS REPLAY. Nothing else can be meant by it, and without this
        // the pump stays parked in the end-of-stream branch -- which only ever woke for a
        // SEEK -- while the audio starts again underneath it. Sound, frozen picture, and a
        // transport that says it is playing.
        if (endReached) seekTo(0)
        baseMs = clockMs()
        baseWall = SystemClock.elapsedRealtime()
        playing = true
        if (!audioOk) return
        // A seek still resolving owns the sound: starting it here would play the OLD
        // position for as long as the seek takes to land. The completion callback starts
        // it instead, at the frame the video settled on.
        if (audioSeeking) startAudioWhenSeeked = true
        else runCatching { audio?.start() }
    }

    fun pause() {
        if (!active) return
        baseMs = clockMs()
        playing = false
        startAudioWhenSeeked = false
        if (audioOk) runCatching { audio?.pause() }
    }

    /**
     * Jump to [ms].
     *
     * The audio moves immediately -- it is the clock, so it has to be right before any
     * frame is measured against it -- and the video catches up inside the pump, which
     * re-seeks the extractor and then fast-forwards to the target WITHOUT swapping the
     * frames it passes. Swapping them would make a scrub cost one full pipeline run per
     * intermediate frame, which is a seek bar that fights back.
     */
    fun seekTo(ms: Int) {
        if (!active) return
        val t = ms.coerceIn(0, durationMs.coerceAtLeast(0))
        baseMs = t
        baseWall = SystemClock.elapsedRealtime()
        endReached = false
        if (audioOk) {
            // ⚠ THE SOUND STOPS FIRST, and it does not move yet. Resolving a seek is not
            // instant: `SEEK_TO_PREVIOUS_SYNC` lands on the keyframe BEFORE the target,
            // which can be seconds earlier, and every frame from there to the target has
            // to be decoded to get one that can be shown. Letting the audio play through
            // that is what made an abrupt seek come back out of sync -- the sound ran on
            // at 1x for the few hundred ms the video spent catching up, then the video
            // arrived that far behind and dropped frames until it caught back up.
            //
            // So the sound waits for the picture, and [settleAudioAt] starts it again at
            // the frame that is actually on screen. A scrub is briefly silent, which is
            // what a scrub sounds like everywhere else.
            audioSeeking = true
            audioEnded = false
            startAudioWhenSeeked = false
            runCatching { audio?.pause() }
        }
        seekTargetMs = t
    }

    /**
     * The video has landed on [shownPtsMs]. Put the sound there and let it go.
     *
     * ⚠ The sound is moved to the frame that is ON SCREEN, not to the number the finger
     * named. The two differ: the decoder resumes at a keyframe and settles on the first
     * frame at or after the request, which can be a little past it. Seeking the audio to
     * the request instead would leave a fixed offset behind every scrub.
     *
     * The wall clock is re-based here as well, so the stretch between this and the sound
     * actually restarting is measured from the right place.
     */
    private fun settleAudioAt(shownPtsMs: Int) {
        baseMs = shownPtsMs
        baseWall = SystemClock.elapsedRealtime()
        if (!audioOk) { audioSeeking = false; return }
        startAudioWhenSeeked = playing
        runCatching { audio?.seekTo(shownPtsMs) }.onFailure {
            audioSeeking = false
            startAudioWhenSeeked = false
        }
    }

    /** Stop the pump and free the audio. Safe to call repeatedly, and from any thread. */
    fun stop() {
        stopping = true
        playing = false
        audioSeeking = false
        audioEnded = false
        endReached = false
        startAudioWhenSeeked = false
        thread?.let { t -> runCatching { t.join(1500) } }
        thread = null
        runCatching { audio?.stop() }
        runCatching { audio?.release() }
        audio = null
        audioOk = false
        active = false
    }

    // ------------------------------------------------------------------ the pump

    private fun pump(path: String, onShot: (Shot) -> Unit) {
        var extractor: MediaExtractor? = null
        var decoder: MediaCodec? = null
        try {
            extractor = MediaExtractor().apply { setDataSource(path) }
            var track = -1
            var fmt: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                if (f.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true) {
                    track = i; fmt = f; break
                }
            }
            val vf = fmt ?: run {
                onShot(Shot(null, 0, 0.0, 0, 0, "no video track"))
                return
            }
            extractor.selectTrack(track)

            val width = vf.getInteger(MediaFormat.KEY_WIDTH)
            val height = vf.getInteger(MediaFormat.KEY_HEIGHT)
            // KEY_ROTATION first, the retriever second -- the same order and the same
            // reasoning as FrameSeeker and VideoSwapper. MediaCodec does not apply it, so a
            // portrait clip decodes on its side and every one of these three has to undo it.
            val rotation = when {
                vf.containsKey(MediaFormat.KEY_ROTATION) -> vf.getInteger(MediaFormat.KEY_ROTATION)
                else -> runCatching {
                    val r = MediaMetadataRetriever().apply { setDataSource(path) }
                    val d = r.extractMetadata(
                        MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
                    r.release()
                    d
                }.getOrDefault(0)
            }
            val outW = if (rotation == 90 || rotation == 270) height else width
            val outH = if (rotation == 90 || rotation == 270) width else height

            // The DISPLAY size, which is not the swap size. The swap runs at the clip's own
            // resolution because that is what the output would be; only the ARGB handed to
            // Compose is capped, and `bgrToArgb` does that scale natively so a 1080x1920
            // clip does not allocate an 8 MB IntArray per frame just to be drawn smaller.
            val scale = maxOf(outW, outH).let { if (it > MAX_DRAW) MAX_DRAW.toFloat() / it else 1f }
            val drawW = (outW * scale).toInt().coerceAtLeast(2)
            val drawH = (outH * scale).toInt().coerceAtLeast(2)

            vf.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            decoder = MediaCodec.createDecoderByType(vf.getString(MediaFormat.KEY_MIME)!!)
            decoder.configure(vf, null, null, 0)
            decoder.start()

            // TWO bitmaps, alternated, for the reason LiveEngine alternates two: the one
            // Compose is drawing must not be the one being written. Reusing them at all is
            // what keeps a 30 fps pump off the collector.
            val bmps = arrayOf(
                Bitmap.createBitmap(drawW, drawH, Bitmap.Config.ARGB_8888),
                Bitmap.createBitmap(drawW, drawH, Bitmap.Config.ARGB_8888),
            )
            var bmpIx = 0

            val info = MediaCodec.BufferInfo()
            var sawInputEOS = false
            var sawOutputEOS = false
            var dropped = 0
            var faces = 0
            var fps = 0.0
            var swapCostMs = 0.0
            var fpsN = 0
            var fpsT0 = SystemClock.elapsedRealtime()

            while (!stopping) {
                // A seek supersedes whatever is in flight. Checked FIRST so a drag that
                // lands while the pump is parked still moves the picture.
                val want = seekTargetMs
                if (want >= 0 && !sawSeekApplied(want)) {
                    extractor.seekTo(want * 1000L, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
                    runCatching { decoder.flush() }
                    sawInputEOS = false
                    sawOutputEOS = false
                    appliedSeek = want
                    continue
                }

                // Parked: paused, with no seek waiting to be shown. Nothing is decoded, so
                // a paused player costs nothing at all.
                if (!playing && seekTargetMs < 0) {
                    Thread.sleep(20)
                    continue
                }

                if (sawOutputEOS) {
                    // ⚠ A SEEK THAT RAN OFF THE END IS FINISHED, not still pending. Leaving
                    // it pending made this branch and the seek branch at the top of the loop
                    // hand the iteration back and forth for ever without decoding anything:
                    // the seek branch skipped itself (already applied), the park below
                    // refused to park (a seek was outstanding), and the loop spun flat out
                    // firing `ended` at the UI. Scrubbing near the end of a clip is exactly
                    // how you land on it, which is what "does not respond" was.
                    if (seekTargetMs >= 0 && appliedSeek == seekTargetMs) {
                        seekTargetMs = -1
                        appliedSeek = -1
                    }
                    playing = false
                    endReached = true
                    // A seek that ran off the end never reaches settleAudioAt, so the
                    // sound would stay paused and permanently "still seeking".
                    audioSeeking = false
                    startAudioWhenSeeked = false
                    if (audioOk) runCatching { audio?.pause() }
                    onShot(Shot(null, durationMs, fps, faces, dropped, ended = true))
                    // Park at the end rather than tearing down: the transport is still
                    // live, and a seek backwards has to be able to start it again. [play]
                    // turns a resume from here into a seek to 0, which is what wakes this.
                    while (!stopping && seekTargetMs < 0) Thread.sleep(20)
                    continue
                }

                if (!sawInputEOS) {
                    val inIx = decoder.dequeueInputBuffer(TIMEOUT_US)
                    if (inIx >= 0) {
                        val buf = decoder.getInputBuffer(inIx)!!
                        val n = extractor.readSampleData(buf, 0)
                        if (n < 0) {
                            decoder.queueInputBuffer(inIx, 0, 0, 0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            sawInputEOS = true
                        } else {
                            decoder.queueInputBuffer(inIx, 0, n, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                val outIx = decoder.dequeueOutputBuffer(info, TIMEOUT_US)
                if (outIx < 0) continue

                val ptsMs = (info.presentationTimeUs / 1000L).toInt()
                var render = info.size > 0
                val target = seekTargetMs
                var fastForwarding = false

                if (render && target >= 0) {
                    // Fast-forwarding from the keyframe before the seek target. These
                    // frames are decoded because they have to be -- the codec cannot start
                    // mid-GOP -- but they are never swapped and never shown.
                    if (ptsMs + SEEK_SLACK_MS < target) {
                        render = false
                        fastForwarding = true
                    } else {
                        // The one the user asked for. Shown whatever the clock says,
                        // because while PAUSED there is no clock to be late against and a
                        // scrub that does not update the picture is not a scrub.
                        seekTargetMs = -1
                        appliedSeek = -1
                        // ⚠ The sound is released HERE, at the frame actually landed on,
                        // and not a moment earlier. Everything above this line happened in
                        // silence on purpose.
                        settleAudioAt(ptsMs)
                    }
                } else if (render && playing) {
                    val clock = clockMs()
                    // ⚠ LEAD TIME: a frame is due when the swap will FINISH on time, not
                    // when it would start. Swapping costs 30-60 ms, so scheduling on the
                    // raw timestamp put every picture on screen a whole swap late, which
                    // then made the next one late enough to drop -- three shown, one
                    // dropped, an uneven cadence that reads as stutter even though the
                    // average rate is fine. Leading by what a frame actually costs here
                    // lands them on time and makes the drops rare instead of rhythmic.
                    val lead = swapCostMs.toInt().coerceIn(0, MAX_LEAD_MS)
                    val due = ptsMs - lead
                    if (ptsMs < clock - LATE_MS) {
                        // ⚠ THE DROP, and it happens here rather than after the swap. See
                        // the class doc: this is the entire frame-rate adaptation.
                        render = false
                        dropped++
                    } else if (due > clock + EARLY_MS) {
                        // Ahead of the sound. Wait it out in small steps so a pause, a
                        // seek or a stop is still answered within a frame.
                        var waited = 0
                        while (!stopping && playing && seekTargetMs < 0 &&
                               due > clockMs() + EARLY_MS && waited < MAX_WAIT_MS) {
                            Thread.sleep(4)
                            waited += 4
                        }
                    }
                }

                if (render) {
                    val frameT0 = SystemClock.elapsedRealtime()
                    val img = decoder.getOutputImage(outIx)
                    if (img == null) {
                        render = false
                    } else {
                        val bgr = imageToBgr(img, width, height)
                        img.close()
                        val up = if (rotation == 0) bgr
                                 else NativePipe.rotateBgr(bgr, width, height, rotation)
                        // In place: processFrame writes the swap back into `up`, and
                        // leaves it BYTE-IDENTICAL when it found no face -- which is why
                        // `faces` is reported rather than inferred from the picture.
                        val n = NativePipe.processFrame(up, outW, outH)
                        if (n < 0) {
                            onShot(Shot(null, ptsMs, fps, 0, dropped, NativePipe.lastError()))
                        } else {
                            faces = n
                            val argb = NativePipe.bgrToArgb(up, outW, outH, drawW, drawH)
                            val bmp = bmps[bmpIx]
                            bmpIx = 1 - bmpIx
                            bmp.setPixels(argb, 0, drawW, 0, 0, drawW, drawH)
                            fpsN++
                            val now = SystemClock.elapsedRealtime()
                            if (now - fpsT0 >= 1000) {
                                fps = fpsN * 1000.0 / (now - fpsT0)
                                fpsN = 0
                                fpsT0 = now
                            }
                            onShot(Shot(bmp, ptsMs, fps, faces, dropped))
                        }
                        // What one frame really costs, end to end, smoothed. Measured
                        // rather than assumed: it moves with the swapper, pixel boost, the
                        // enhancer, lip sync and the clip's own resolution, and a constant
                        // guess would be wrong for every combination but one.
                        val cost = (SystemClock.elapsedRealtime() - frameT0).toDouble()
                        swapCostMs = if (swapCostMs <= 0.0) cost
                                     else swapCostMs * 0.8 + cost * 0.2
                    }
                }
                if (!render) {
                    // Position only. The bar has to keep moving through a run of drops or
                    // it reads as a freeze, which is the opposite of what is happening.
                    //
                    // ⚠ While fast-forwarding it reports the TARGET, not this frame. These
                    // timestamps run from the keyframe BEFORE the request up to it, so
                    // reporting them honestly would drag the thumb visibly backwards and
                    // then forwards again on every scrub -- describing the decoder, not the
                    // film. `seeking` is what the UI shows instead.
                    onShot(Shot(null, if (fastForwarding) target else ptsMs,
                                fps, faces, dropped, seeking = fastForwarding))
                }

                if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) sawOutputEOS = true
                decoder.releaseOutputBuffer(outIx, false)
            }
        } catch (t: Throwable) {
            if (!stopping) onShot(Shot(null, 0, 0.0, 0, 0, t.message ?: t.javaClass.simpleName))
            android.util.Log.w("ffplayer", "pump failed for " + path, t)
        } finally {
            runCatching { decoder?.stop() }
            runCatching { decoder?.release() }
            runCatching { extractor?.release() }
        }
    }

    /** Which seek the pump has already re-positioned the extractor for. */
    @Volatile private var appliedSeek = -1
    private fun sawSeekApplied(want: Int) = appliedSeek == want

    /** Identical to FrameSeeker's and VideoSwapper's -- same planes, same converter. */
    private fun imageToBgr(image: android.media.Image, w: Int, h: Int): ByteArray {
        val p = image.planes
        fun bytes(b: ByteBuffer): ByteArray {
            val a = ByteArray(b.remaining()); b.get(a); return a
        }
        return NativePipe.yuvToBgr(
            bytes(p[0].buffer), p[0].rowStride,
            bytes(p[1].buffer), p[1].rowStride, p[1].pixelStride,
            bytes(p[2].buffer), p[2].rowStride, p[2].pixelStride,
            w, h,
        )
    }

    private fun openAudio(path: String): Boolean = runCatching {
        val ex = MediaExtractor().apply { setDataSource(path) }
        var has = false
        for (i in 0 until ex.trackCount)
            if (ex.getTrackFormat(i).getString(MediaFormat.KEY_MIME)
                    ?.startsWith("audio/") == true) has = true
        ex.release()
        if (!has) return false
        // No surface is set, so this decodes and plays the AUDIO track only. It is also
        // the clock -- see clockMs().
        audio = MediaPlayer().apply {
            setDataSource(path)
            setOnErrorListener { _, _, _ -> audioOk = false; true }
            // The two moments the position stops being the truth. Both fall back to wall
            // time, which clockMs() keeps re-based off the sound for exactly this.
            setOnSeekCompleteListener {
                audioSeeking = false
                // THE one place the sound restarts after a scrub: the seek having landed
                // is the only moment at which starting it is in time with the picture.
                if (startAudioWhenSeeked && playing) runCatching { audio?.start() }
                startAudioWhenSeeked = false
            }
            setOnCompletionListener { audioEnded = true }
            prepare()
        }
        true
    }.getOrElse {
        android.util.Log.w("ffplayer", "no audio for " + path, it)
        runCatching { audio?.release() }
        audio = null
        false
    }

    private fun durationOf(path: String): Int = runCatching {
        val r = MediaMetadataRetriever().apply { setDataSource(path) }
        val d = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toIntOrNull() ?: 0
        r.release()
        d
    }.getOrDefault(0)

    private companion object {
        const val TIMEOUT_US = 10_000L
        /**
         * How far behind the sound a frame may be and still be worth swapping.
         *
         * One frame at 24 fps is 41 ms, so 60 ms is "late but not visibly so". Tighter and
         * a pipeline running at very nearly real time drops every other frame for no gain;
         * looser and the picture visibly trails the voice, which is the one error a viewer
         * always spots.
         */
        const val LATE_MS = 60
        /** Ahead by less than this is not worth sleeping for. */
        const val EARLY_MS = 8
        /** A single wait is capped so a stalled clock cannot park the pump for ever. */
        const val MAX_WAIT_MS = 500
        /**
         * The most a frame may be scheduled ahead of its timestamp.
         *
         * A cap, not a target: the lead is the MEASURED cost of a frame, and this only
         * stops a pathological one (a first frame that paid for a lazy graph load, say)
         * from pulling the whole schedule forward behind it.
         */
        const val MAX_LEAD_MS = 70
        /** Timestamp rounding, so a seek target is not missed by a fraction of a frame. */
        const val SEEK_SLACK_MS = 2
        /**
         * Longest edge of what Compose is handed.
         *
         * The SWAP is full resolution -- capping that would mean previewing something the
         * real run would not produce. This caps only the drawing copy, which no phone
         * screen can show more of anyway.
         */
        const val MAX_DRAW = 1280
    }
}
