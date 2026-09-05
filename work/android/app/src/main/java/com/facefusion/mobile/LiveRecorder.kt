package com.facefusion.mobile

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.File

/**
 * Record what the Live tab is showing — roadmap 13b.
 *
 * Frames arrive already swapped, as BGR, from the same `liveFrame` pump that draws the
 * preview: `NativePipe.liveFrame` keeps the full-resolution swapped frame natively and
 * copies it out only while a recording is running. So recording costs one buffer copy per
 * frame, not a second trip through the pipeline.
 *
 * ⚠ **ByteBuffer input, not a Surface.** The obvious design is `createInputSurface` plus
 * GL, but the frames here are bytes in memory, not textures — feeding a Surface would mean
 * uploading each frame to a texture only to have the encoder read it back. `VideoSwapper`
 * already established the ByteBuffer path for exactly this reason, and this reuses its
 * hardest-won detail: `getInputImage` rather than a packed I420 blob, because
 * `COLOR_FormatYUV420Flexible` does not imply I420 and this device's AVC encoder is
 * semi-planar. Writing I420 into it puts luma in the right place and chroma in the wrong
 * one — a greyscale picture with green and pink blobs.
 *
 * ⚠ **Timestamps come from the wall clock, never from a frame counter.** Live is variable
 * frame rate by nature: ~15 fps with a face in shot and ~29 without, and slower again on
 * the ncnn backend. A file written with evenly spaced timestamps plays back at the wrong
 * speed — faster where the phone was working hardest — and it looks like a performance
 * problem rather than the timestamp bug it is.
 *
 * **Video only.** Audio is a separate decision: the microphone is already used by the Voice
 * picker, and muxing a second track means deciding what happens when the two disagree about
 * length. Saying so in one line beats shipping a silent track that looks broken.
 */
class LiveRecorder(private val out: File, private val onLog: (String) -> Unit = {}) {

    private var encoder: MediaCodec? = null
    private var muxer: MediaMuxer? = null
    private var track = -1
    private var muxing = false
    private var startNs = 0L
    private var frames = 0

    /** Everything below is touched from the analyzer thread AND from stop() on the main one. */
    private val lock = Any()

    @Volatile private var failed: String? = null
    val error: String? get() = failed

    /**
     * Set by [stop], and the reason [frame] cannot resurrect a finished recording.
     *
     * ⚠ Ordering alone does not close this. The caller clears `LiveEngine.recorder` before
     * stopping, but a frame ALREADY INSIDE the pump has read the old reference and will
     * call [frame] with it -- the lock serialises the two, it does not prevent the second.
     * Without this flag that late frame finds `encoder == null` and `failed == null`, so
     * `ensure` builds a SECOND encoder and muxer over the same path, which is then never
     * stopped: a leaked codec and a file truncated by its own replacement.
     */
    @Volatile private var stopped = false

    /**
     * The size the encoder was configured for.
     *
     * ⚠ `ensure` returns early once an encoder exists, so without this a resolution change
     * mid-recording would feed the old encoder buffers of a new size. The camera does not
     * renegotiate inside one binding, so this should never fire -- which is exactly why it
     * has to say something rather than produce a corrupt file quietly.
     */
    private var encW = 0
    private var encH = 0

    /** How many frames have been written. 0 after a stop that produced nothing usable. */
    val frameCount: Int get() = frames

    /**
     * Bring the encoder up for a frame size that is only known once the first frame arrives.
     *
     * Deliberately lazy: the camera's delivered resolution is negotiated, not requested (see
     * LiveEngine's ResolutionSelector), so the size is a fact about the running session
     * rather than something the caller can state in advance.
     */
    private fun ensure(w: Int, h: Int): Boolean {
        if (encoder != null) {
            if (w == encW && h == encH) return true
            failed = "frame size changed mid-recording (" + encW + "x" + encH +
                     " -> " + w + "x" + h + ")"
            onLog("recorder: $failed")
            return false
        }
        if (failed != null || stopped) return false
        return runCatching {
            // Even dimensions: a 4:2:0 chroma plane is half-size in both axes, and an odd
            // edge leaves the encoder rounding in a direction nothing here controls.
            val ew = w and 1.inv()
            val eh = h and 1.inv()
            val fmt = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, ew, eh)
            fmt.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                           MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            // ~8 Mbps at 720p. Live is a preview-quality feed already downsampled by the
            // camera; spending more bits than the source carries information is waste.
            fmt.setInteger(MediaFormat.KEY_BIT_RATE, (ew * eh * 8).coerceAtLeast(2_000_000))
            // A HINT only, and the reason the timestamps below are wall-clock: the encoder
            // wants a nominal rate for its rate control, but nothing about this feed is
            // actually periodic.
            fmt.setInteger(MediaFormat.KEY_FRAME_RATE, 30)
            fmt.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            val enc = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
            enc.configure(fmt, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            enc.start()
            encoder = enc
            encW = w; encH = h
            muxer = MediaMuxer(out.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            startNs = System.nanoTime()
            onLog("recording ${ew}x${eh} -> ${out.name}")
            true
        }.getOrElse {
            failed = it.message ?: "encoder failed"
            onLog("recorder: $failed")
            false
        }
    }

    /**
     * One swapped frame, BGR, [w]*[h]*3 bytes.
     *
     * Called on the analyzer thread, in line with the pump. Encoding a 720p frame costs a
     * few milliseconds against the pump's ~60, so it is not worth another thread and its
     * queue — and a queue would be the wrong answer anyway: dropping the recording behind
     * the preview would produce a file whose timestamps no longer match what was seen.
     */
    // ⚠ A BLOCK body, not `= synchronized(lock) { ... }`. An expression body infers its
    // type from the block's last expression -- Result<Unit>, from the runCatching below --
    // and then every early `return` in it is a type error. Unit is what this returns.
    fun frame(bgr: ByteArray, w: Int, h: Int) {
      synchronized(lock) {
        // stopped FIRST: a frame that was already inside the pump when the recording ended
        // arrives here afterwards, and must do nothing at all. See [stopped].
        if (stopped || failed != null) return
        if (!ensure(w, h)) return
        val enc = encoder ?: return
        val ew = w and 1.inv()
        val eh = h and 1.inv()
        runCatching {
            drain(false)
            val ix = enc.dequeueInputBuffer(0)
            // Zero timeout, and a DROPPED frame when the encoder is not ready. Blocking
            // here would stall the camera pump, so the preview would stutter because a
            // recording was running -- the recording is the guest, not the host.
            if (ix < 0) return@runCatching
            val ptsUs = (System.nanoTime() - startNs) / 1000
            val img = enc.getInputImage(ix)
            if (img == null) {
                val i420 = NativePipe.bgrToI420(bgr, w, h)
                enc.getInputBuffer(ix)!!.apply { clear(); put(i420) }
                enc.queueInputBuffer(ix, 0, i420.size, ptsUs, 0)
            } else {
                val p = img.planes
                NativePipe.bgrToImagePlanes(
                    bgr, w, h,
                    p[0].buffer, p[0].rowStride, p[0].pixelStride,
                    p[1].buffer, p[1].rowStride, p[1].pixelStride,
                    p[2].buffer, p[2].rowStride, p[2].pixelStride,
                )
                enc.queueInputBuffer(ix, 0, p[0].rowStride * eh * 3 / 2, ptsUs, 0)
            }
            frames++
        }.onFailure {
            failed = it.message ?: "encode failed"
            onLog("recorder: $failed")
        }
      }
    }

    /**
     * Finish the file.
     *
     * Returns it when there is something worth keeping, null otherwise — and DELETES the
     * file in that case. A zero-frame MP4 is not a recording; leaving it on disk would put
     * an unplayable file in front of the user with no way to tell it from a real one.
     */
    fun stop(): File? = synchronized(lock) {
        stopped = true
        val enc = encoder ?: run { cleanupFailed(); return null }
        runCatching {
            // EOS through the input queue, then drain until the encoder says it is done.
            val ix = enc.dequeueInputBuffer(100_000)
            if (ix >= 0)
                enc.queueInputBuffer(ix, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
            drain(true)
        }
        runCatching { enc.stop() }
        runCatching { enc.release() }
        encoder = null
        // ⚠ stop() on a muxer that was never started throws. It is only started once the
        // encoder has produced a format, which never happens if the first frame failed.
        if (muxing) runCatching { muxer?.stop() }
        runCatching { muxer?.release() }
        muxer = null
        muxing = false
        if (frames == 0 || !muxingEverStarted) { cleanupFailed(); return null }
        return out
    }

    private var muxingEverStarted = false

    private fun cleanupFailed() {
        runCatching { out.delete() }
    }

    /** Move whatever the encoder has produced into the muxer. */
    private fun drain(toEnd: Boolean) {
        val enc = encoder ?: return
        val info = MediaCodec.BufferInfo()
        while (true) {
            val ix = enc.dequeueOutputBuffer(info, if (toEnd) 100_000 else 0)
            when {
                ix == MediaCodec.INFO_TRY_AGAIN_LATER -> return
                ix == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    // ⚠ The ONLY place the track may be added. MediaMuxer takes no tracks
                    // after start(), and AVC's csd-0/csd-1 exist only once the encoder has
                    // emitted this -- the same rule VideoSwapper's audio path pays for.
                    if (!muxing) {
                        track = muxer!!.addTrack(enc.outputFormat)
                        muxer!!.start()
                        muxing = true
                        muxingEverStarted = true
                    }
                }
                ix >= 0 -> {
                    val buf = enc.getOutputBuffer(ix)
                    // The codec-config buffer is metadata, already carried by addTrack.
                    val cfg = info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0
                    if (buf != null && info.size > 0 && muxing && !cfg) {
                        buf.position(info.offset)
                        buf.limit(info.offset + info.size)
                        muxer!!.writeSampleData(track, buf, info)
                    }
                    enc.releaseOutputBuffer(ix, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
                else -> return
            }
        }
    }
}
