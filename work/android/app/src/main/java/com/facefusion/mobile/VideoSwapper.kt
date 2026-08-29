package com.facefusion.mobile

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.File
import java.nio.ByteBuffer

/**
 * Decode -> swap -> encode, with the original audio copied through and an optional trim.
 *
 * Deliberately ByteBuffer-based rather than Surface/GL: a Surface pipeline never exposes the
 * pixels to the CPU, and every frame has to reach the NPU as packed BGR anyway.  The
 * decoder's YUV_420_888 output carries arbitrary row/pixel strides (semi-planar chroma shows
 * up as pixelStride 2), so both strides are honoured in yuvToBgr -- assuming tightly packed
 * I420 gives a green-and-magenta image.
 *
 * Trimming seeks to the sync frame at or before trimStartUs and decodes forward, discarding
 * output frames before the mark: seeking straight to an arbitrary timestamp would land on a
 * non-keyframe and decode garbage until the next IDR.  Output timestamps are rebased to zero
 * so the result starts where the trim does.
 */
class VideoSwapper(
    /**
     * Frames per second to WRITE. 0 keeps the input's rate.
     *
     * Lowering it drops frames rather than re-timing them: the encoder is told the new rate
     * and each decoded frame is kept only if it lands in a slot that is still empty. That
     * makes a 30 -> 24 conversion cost 20% LESS NPU time, which is most of the point --
     * a swap is ~19 ms of NPU per frame, so the frames not kept are the frames not swapped.
     */
    private val outputFps: Int = 0,
    private val trimStartUs: Long = 0L,
    private val trimEndUs: Long = Long.MAX_VALUE,
    private val onProgress: (done: Int, total: Int) -> Unit = { _, _ -> },
    private val onFrame: (bgr: ByteArray, w: Int, h: Int) -> Unit = { _, _, _ -> },
    private val onLog: (String) -> Unit = {},
    /**
     * Asked once per decoded frame. Returning true abandons the run.
     *
     * A flag rather than coroutine cancellation: the loop below is blocking codec work, so
     * nothing here would ever reach a suspension point for the coroutine to cancel at.
     */
    private val isCancelled: () -> Boolean = { false },
) {

    private var encTrack = -1

    fun swap(inputPath: String, outputPath: String): Result<String> = runCatching {
        val extractor = MediaExtractor().apply { setDataSource(inputPath) }
        var videoTrack = -1
        var audioTrack = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("video/") && videoTrack < 0) { videoTrack = i; format = f }
            else if (mime.startsWith("audio/") && audioTrack < 0) audioTrack = i
        }
        val vf = format ?: error("no video track in $inputPath")
        val width = vf.getInteger(MediaFormat.KEY_WIDTH)
        val height = vf.getInteger(MediaFormat.KEY_HEIGHT)
        val inFps = if (vf.containsKey(MediaFormat.KEY_FRAME_RATE))
            vf.getInteger(MediaFormat.KEY_FRAME_RATE) else 30
        // Never above the input: duplicating frames would cost a full swap each and add
        // nothing. 0 (or anything silly) means "keep the input rate".
        val fps = if (outputFps in 1..inFps) outputFps else inFps
        val spanUs = (if (trimEndUs == Long.MAX_VALUE) durationUs(vf) else trimEndUs) - trimStartUs
        val expected = ((spanUs / 1_000_000.0) * fps).toInt().coerceAtLeast(1)
        onLog("${width}x$height @ ${inFps}fps" +
              (if (fps != inFps) " -> ${fps}fps" else "") + ", ~$expected frames")

        extractor.selectTrack(videoTrack)
        if (trimStartUs > 0) extractor.seekTo(trimStartUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)

        val decoder = MediaCodec.createDecoderByType(vf.getString(MediaFormat.KEY_MIME)!!)
        vf.setInteger(MediaFormat.KEY_COLOR_FORMAT,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
        decoder.configure(vf, null, null, 0)
        decoder.start()

        val encFormat = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            setInteger(MediaFormat.KEY_BIT_RATE, (width * height * fps * 0.25).toInt())
            setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2)
        }
        val encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        encoder.configure(encFormat, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        encoder.start()

        val muxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        var muxAudio = -1
        var muxing = false
        // Audio is PASS-THROUGH, and the MP4 muxer accepts a strictly narrower set of codecs
        // than MediaExtractor will hand back: Opus, Vorbis, FLAC and raw PCM all extract
        // fine and all make addTrack throw IllegalStateException("Failed to add the track
        // to the muxer").  Reported on 0.1.0 by a OnePlus PJZ110 on an 854x476 clip -- the
        // gate, the decoder and the encoder had all already run, and the whole swap was
        // lost over a track we do nothing to but copy.
        //
        // So audio is best-effort.  On refusal muxAudio stays -1, the copy loop below skips
        // itself, and the video is still written.  The VIDEO addTrack stays fatal: with no
        // video track there is no output at all.
        //
        // addTrack leaves the muxer in its INITIALIZED state when it rejects a format -- it
        // throws before touching mState -- so start() below is still valid after a refusal.
        val addTracks: (MediaFormat) -> Int = { fmt ->
            val v = muxer.addTrack(fmt)
            if (audioTrack >= 0) {
                val af = extractor.getTrackFormat(audioTrack)
                val amime = af.getString(MediaFormat.KEY_MIME) ?: "?"
                muxAudio = runCatching { muxer.addTrack(af) }.getOrElse {
                    onLog("audio: MP4 will not carry $amime, writing video only")
                    -1
                }
            }
            muxer.start()
            v
        }

        val info = MediaCodec.BufferInfo()
        var sawInputEOS = false
        var sawDecodeEOS = false
        var swapped = 0
        var cancelled = false
        // The last output slot already written, for rate conversion. -1 so slot 0 is free.
        var lastSlot = -1L

        while (!sawDecodeEOS) {
            // Checked before the work, not after, so pressing Cancel stops the next frame
            // rather than finishing it first.
            if (isCancelled()) {
                // KEEP what has already been swapped.
                //
                // This used to delete the output and throw, which threw away every frame
                // the NPU had produced -- on a long clip that is minutes of work discarded
                // because the user stopped a run that was already mostly useful.
                //
                // Instead, stop feeding the decoder and fall through to the SAME
                // finalisation an ordinary run uses: signal EOS, drain the encoder, copy
                // the audio, stop the muxer. The only thing cancelling changes is where the
                // video ends.
                onLog("cancelled after $swapped frames -- keeping them")
                cancelled = true
                signalEncoderEOS(encoder)
                sawInputEOS = true
                sawDecodeEOS = true
                break
            }
            if (!sawInputEOS) {
                val inIx = decoder.dequeueInputBuffer(10_000)
                if (inIx >= 0) {
                    val n = extractor.readSampleData(decoder.getInputBuffer(inIx)!!, 0)
                    val pts = extractor.sampleTime
                    if (n < 0 || pts > trimEndUs) {
                        decoder.queueInputBuffer(inIx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        sawInputEOS = true
                    } else {
                        decoder.queueInputBuffer(inIx, 0, n, pts, 0)
                        extractor.advance()
                    }
                }
            }
            val outIx = decoder.dequeueOutputBuffer(info, 10_000)
            if (outIx >= 0) {
                val pts = info.presentationTimeUs
                // frames between the sync point and the trim mark are decoded but dropped
                var inRange = info.size > 0 && pts >= trimStartUs && pts <= trimEndUs
                // Rate conversion, as decimation: which output slot would this frame fill?
                // If that slot is taken, the frame is surplus and is dropped BEFORE the
                // swap, so a lower rate genuinely costs less NPU time rather than just
                // writing fewer frames.
                if (inRange && fps != inFps) {
                    val slot = ((pts - trimStartUs) * fps) / 1_000_000L
                    if (slot <= lastSlot) inRange = false else lastSlot = slot
                }
                if (inRange) {
                    val image = decoder.getOutputImage(outIx)
                    if (image != null) {
                        val bgr = imageToBgr(image, width, height)
                        image.close()
                        val faces = NativePipe.processFrame(bgr, width, height)
                        if (faces < 0) error("native: ${NativePipe.lastError()}")
                        onFrame(bgr, width, height)
                        feedEncoder(encoder, bgr, width, height, pts - trimStartUs)
                        swapped++
                        onProgress(swapped, expected)
                    }
                }
                val eos = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                decoder.releaseOutputBuffer(outIx, false)
                if (eos) { sawDecodeEOS = true; signalEncoderEOS(encoder) }
            }
            muxing = drainEncoder(encoder, muxer, muxing, addTracks = addTracks)
        }
        var flushed = false
        while (!flushed) {
            muxing = drainEncoder(encoder, muxer, muxing, addTracks = addTracks) { flushed = true }
        }

        // Cancelled before a single frame reached the encoder: the muxer never had a track
        // added, and an MP4 with no tracks is not a file anything can open. That is the one
        // case where there is genuinely nothing to keep.
        if (!muxing) {
            decoder.stop(); decoder.release()
            encoder.stop(); encoder.release()
            extractor.release()
            muxer.release()
            File(outputPath).delete()
            error(if (cancelled) "cancelled before any frame was written" else "no frames encoded")
        }

        // Same principle as addTracks above, one stage later: by this point every frame has
        // been swapped and encoded, so a throw in the copy-through would discard the entire
        // run's NPU work for a soundtrack.  A partial audio track is still a valid MP4 --
        // writeSampleData has already committed whatever it accepted.
        if (audioTrack >= 0 && muxAudio >= 0) runCatching {
            val ae = MediaExtractor().apply { setDataSource(inputPath); selectTrack(audioTrack) }
            try {
                if (trimStartUs > 0) ae.seekTo(trimStartUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
                val buf = ByteBuffer.allocate(512 * 1024)
                val ai = MediaCodec.BufferInfo()
                var copied = 0
                while (true) {
                    val n = ae.readSampleData(buf, 0)
                    val pts = ae.sampleTime
                    if (n < 0 || pts > trimEndUs) break
                    if (pts >= trimStartUs) {
                        ai.offset = 0; ai.size = n
                        ai.presentationTimeUs = pts - trimStartUs
                        ai.flags = if (ae.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0)
                            MediaCodec.BUFFER_FLAG_KEY_FRAME else 0
                        muxer.writeSampleData(muxAudio, buf, ai)
                        copied++
                    }
                    ae.advance()
                }
                onLog("audio: $copied packets copied")
            } finally {
                // finally, not a trailing call: on a throw the extractor would otherwise
                // leak a codec handle for the life of the process.
                ae.release()
            }
        }.onFailure { onLog("audio: copy-through failed (${it.javaClass.simpleName}), video kept") }

        decoder.stop(); decoder.release()
        encoder.stop(); encoder.release()
        muxer.stop(); muxer.release()
        extractor.release()
        onLog((if (cancelled) "partial: " else "") +
              "wrote ${File(outputPath).length() / 1024} KB, $swapped frames")
        outputPath
    }

    private fun drainEncoder(
        encoder: MediaCodec, muxer: MediaMuxer, started: Boolean,
        addTracks: (MediaFormat) -> Int,
        onEOS: (() -> Unit)? = null,
    ): Boolean {
        var muxing = started
        val info = MediaCodec.BufferInfo()
        while (true) {
            val ix = encoder.dequeueOutputBuffer(info, 0)
            when {
                ix == MediaCodec.INFO_TRY_AGAIN_LATER -> return muxing
                ix == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    encTrack = addTracks(encoder.outputFormat)
                    muxing = true
                }
                ix >= 0 -> {
                    val out = encoder.getOutputBuffer(ix)!!
                    if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) info.size = 0
                    if (info.size > 0 && muxing) {
                        out.position(info.offset)
                        out.limit(info.offset + info.size)
                        muxer.writeSampleData(encTrack, out, info)
                    }
                    val eos = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    encoder.releaseOutputBuffer(ix, false)
                    if (eos) { onEOS?.invoke(); return muxing }
                }
            }
        }
    }

    /**
     * Hand a BGR frame to the encoder through its own input Image.
     *
     * NOT via getInputBuffer + a packed I420 blob: COLOR_FormatYUV420Flexible does not
     * imply I420, and this device's AVC encoder is semi-planar.  Writing I420 into it put
     * luma in the right place and chroma in the wrong one -- a greyscale picture with green
     * and pink blobs.  getInputImage exposes the real strides, so both layouts work.
     */
    private fun feedEncoder(encoder: MediaCodec, bgr: ByteArray, w: Int, h: Int, ptsUs: Long) {
        val ix = encoder.dequeueInputBuffer(100_000)
        if (ix < 0) return
        val img = encoder.getInputImage(ix)
        if (img == null) {
            // no Image view available: fall back to a packed I420 blob
            val i420 = NativePipe.bgrToI420(bgr, w, h)
            encoder.getInputBuffer(ix)!!.apply { clear(); put(i420) }
            encoder.queueInputBuffer(ix, 0, i420.size, ptsUs.coerceAtLeast(0), 0)
            return
        }
        val p = img.planes
        val ok = NativePipe.bgrToImagePlanes(
            bgr, w, h,
            p[0].buffer, p[0].rowStride, p[0].pixelStride,
            p[1].buffer, p[1].rowStride, p[1].pixelStride,
            p[2].buffer, p[2].rowStride, p[2].pixelStride,
        )
        if (!ok) onLog("encoder planes were not direct buffers")
        val size = p[0].rowStride * h * 3 / 2
        encoder.queueInputBuffer(ix, 0, size, ptsUs.coerceAtLeast(0), 0)
    }

    private fun signalEncoderEOS(encoder: MediaCodec) {
        val ix = encoder.dequeueInputBuffer(100_000)
        if (ix >= 0) encoder.queueInputBuffer(ix, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
    }

    private fun imageToBgr(image: android.media.Image, w: Int, h: Int): ByteArray {
        val p = image.planes
        fun bytes(b: ByteBuffer): ByteArray { val a = ByteArray(b.remaining()); b.get(a); return a }
        return NativePipe.yuvToBgr(
            bytes(p[0].buffer), p[0].rowStride,
            bytes(p[1].buffer), p[1].rowStride, p[1].pixelStride,
            bytes(p[2].buffer), p[2].rowStride, p[2].pixelStride,
            w, h,
        )
    }

    private fun durationUs(f: MediaFormat) =
        if (f.containsKey(MediaFormat.KEY_DURATION)) f.getLong(MediaFormat.KEY_DURATION) else 0L
}
