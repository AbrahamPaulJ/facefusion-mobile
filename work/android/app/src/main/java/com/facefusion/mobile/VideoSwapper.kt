package com.facefusion.mobile

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaMetadataRetriever
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
        // STORED dimensions. For a portrait clip these are landscape, with the upright
        // orientation carried separately as a rotation flag.
        val width = vf.getInteger(MediaFormat.KEY_WIDTH)
        val height = vf.getInteger(MediaFormat.KEY_HEIGHT)

        /*
         * The container's rotation, which MediaCodec does NOT apply.
         *
         * A phone records portrait as LANDSCAPE frames plus a 90-degree flag. Before this,
         * nothing read the flag: frames reached yoloface on their side, which does not
         * detect a rotated face (the same failure as the EXIF bug on the source path), and
         * the output was written landscape with no flag either, so it also PLAYED sideways.
         *
         * It was invisible in the preview because MediaMetadataRetriever applies the flag
         * and MediaCodec does not -- the preview looked upright and only the result was
         * wrong, which is the worst way for this to present.
         *
         * The frames are rotated upright BEFORE detection rather than tagging the output
         * with an orientation hint, because a hint would fix playback and leave the
         * detection failure -- and detection is the part that makes the feature not work.
         *
         * KEY_ROTATION is not always present on an extractor track format; MMR's metadata
         * is the more reliable source, so it is the fallback.
         */
        val rotation = when {
            vf.containsKey(MediaFormat.KEY_ROTATION) -> vf.getInteger(MediaFormat.KEY_ROTATION)
            else -> runCatching {
                val r = MediaMetadataRetriever().apply { setDataSource(inputPath) }
                val d = r.extractMetadata(
                    MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
                r.release()
                d
            }.getOrDefault(0)
        }.let { ((it % 360) + 360) % 360 }

        // What everything downstream sees: 90 and 270 swap the axes.
        val outW = if (rotation == 90 || rotation == 270) height else width
        val outH = if (rotation == 90 || rotation == 270) width else height
        val inFps = if (vf.containsKey(MediaFormat.KEY_FRAME_RATE))
            vf.getInteger(MediaFormat.KEY_FRAME_RATE) else 30
        // Never above the input: duplicating frames would cost a full swap each and add
        // nothing. 0 (or anything silly) means "keep the input rate".
        val fps = if (outputFps in 1..inFps) outputFps else inFps
        val spanUs = (if (trimEndUs == Long.MAX_VALUE) durationUs(vf) else trimEndUs) - trimStartUs
        val expected = ((spanUs / 1_000_000.0) * fps).toInt().coerceAtLeast(1)
        onLog("${width}x$height @ ${inFps}fps" +
              (if (rotation != 0) " rot ${rotation} -> ${outW}x$outH" else "") +
              (if (fps != inFps) " -> ${fps}fps" else "") + ", ~$expected frames")

        extractor.selectTrack(videoTrack)
        if (trimStartUs > 0) extractor.seekTo(trimStartUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)

        val decoder = MediaCodec.createDecoderByType(vf.getString(MediaFormat.KEY_MIME)!!)
        vf.setInteger(MediaFormat.KEY_COLOR_FORMAT,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
        decoder.configure(vf, null, null, 0)
        decoder.start()

        val encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)

        /*
         * ASK the encoder what it will accept, rather than handing it the clip's size.
         *
         * A 196x112 clip made `encoder.start()` throw MediaCodec.CodecException -- with an
         * EMPTY message, so it arrived as a failure with nothing in it. 196 is not a
         * multiple of 16, and this device's AVC encoder has a width alignment of 16. The
         * dimensions a hardware encoder takes are a property of the silicon, so they are
         * asked for here instead of assumed; the alignment is 2 on some parts and 16 on
         * others, and hardcoding either is how this breaks again on a different phone.
         *
         * Aligned DOWN and the frame is cropped to match: rounding up would need padding,
         * and a black seam on two edges of every frame is worse than losing up to 15 px.
         */
        val vcaps = encoder.codecInfo
            .getCapabilitiesForType(MediaFormat.MIMETYPE_VIDEO_AVC).videoCapabilities
        val encW = (outW / vcaps.widthAlignment) * vcaps.widthAlignment
        val encH = (outH / vcaps.heightAlignment) * vcaps.heightAlignment
        if (encW < vcaps.supportedWidths.lower || encH < vcaps.supportedHeights.lower) {
            encoder.release()
            // A clear sentence rather than a CodecException: nothing about the clip can be
            // changed by retrying, and the numbers say exactly why.
            error("this device cannot encode ${outW}x$outH video -- the smallest it takes " +
                  "is ${vcaps.supportedWidths.lower}x${vcaps.supportedHeights.lower}")
        }
        if (encW != outW || encH != outH)
            onLog("encoder wants multiples of ${vcaps.widthAlignment}x" +
                  "${vcaps.heightAlignment}: cropping ${outW}x$outH to ${encW}x$encH")

        // The encoder is sized to the UPRIGHT frame, so the output needs no orientation
        // hint of its own -- the rotation is baked into the pixels.
        val encFormat = MediaFormat.createVideoFormat(
                MediaFormat.MIMETYPE_VIDEO_AVC, encW, encH).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            setInteger(MediaFormat.KEY_BIT_RATE, (encW * encH * fps * 0.25).toInt()
                .coerceAtLeast(64_000))
            setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2)
        }
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
                        // Upright FIRST: everything after this -- detection, the swap, the
                        // preview, the encoder -- works on the frame the viewer will see.
                        val decoded = imageToBgr(image, width, height)
                        image.close()
                        val bgr = if (rotation == 0) decoded
                                  else NativePipe.rotateBgr(decoded, width, height, rotation)
                        val faces = NativePipe.processFrame(bgr, outW, outH)
                        if (faces < 0) error("native: ${NativePipe.lastError()}")
                        onFrame(bgr, outW, outH)
                        feedEncoder(encoder, cropBgr(bgr, outW, outH, encW, encH),
                                    encW, encH, pts - trimStartUs)
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

    /**
     * Centre-crop a BGR frame to what the encoder accepts.
     *
     * Returns the input untouched in the common case, which is every clip whose dimensions
     * already fit. The offsets are forced even so the chroma plane still lines up with the
     * luma when this is subsampled on the way in.
     */
    private fun cropBgr(src: ByteArray, w: Int, h: Int, dw: Int, dh: Int): ByteArray {
        if (dw == w && dh == h) return src
        val x0 = ((w - dw) / 2) and 1.inv()
        val y0 = ((h - dh) / 2) and 1.inv()
        val out = ByteArray(dw * dh * 3)
        for (y in 0 until dh)
            System.arraycopy(src, ((y0 + y) * w + x0) * 3, out, y * dw * 3, dw * 3)
        return out
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
