package com.facefusion.mobile

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import java.io.File
import java.nio.ByteBuffer

/**
 * A video made FROM A STILL: face_animator (the photo's face moved like a driver video's)
 * and the talking photo (the photo, with the lip syncer moving its mouth to a voice).
 *
 * Separate from [VideoSwapper] because the frames do not come from the target: there is
 * either a driver to decode, whose pixels never reach the output, or nothing to decode at
 * all. The output is the PHOTO's size and lasts as long as the driver, or the voice.
 * Every frame goes through `NativePipe.portraitFrame`, which runs the rest of the chosen
 * processors in the pipeline's order (editor, lip syncer, enhancer).
 *
 * The audio is always transcoded to AAC, from the voice when lip syncing and otherwise from
 * the driver when it has sound -- one path, the one [VideoSwapper] falls back to.
 */
class PortraitVideo(
    /** Frames per second to write. A driver slower than this keeps its own rate. */
    private val fps: Int = 25,
    private val lipSync: Boolean = false,
    private val voicePath: String? = null,
    private val voiceTrimStartUs: Long = 0L,
    private val voiceTrimEndUs: Long = Long.MAX_VALUE,
    private val onProgress: (done: Int, total: Int) -> Unit = { _, _ -> },
    private val onFrame: (bgr: ByteArray, w: Int, h: Int) -> Unit = { _, _, _ -> },
    private val onLog: (String) -> Unit = {},
    private val isCancelled: () -> Boolean = { false },
) {

    /**
     * Render [photo] (w x h BGR) into [outputPath]. With [driverPath] the photo is animated
     * and the clip is the driver's length; without one it is the voice's. Returns the path.
     */
    fun run(photo: ByteArray, w: Int, h: Int, driverPath: String?, outputPath: String):
            Result<String> = runCatching {
        val voice = if (lipSync) voicePath else null
        require(driverPath != null || voice != null) { "nothing to make a video from" }

        // The encoder's size: the photo's, aligned DOWN to what this encoder takes and shrunk
        // until it accepts it at this rate, as VideoSwapper does for a clip.
        val vcaps = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC).let {
            try { it.codecInfo.getCapabilitiesForType(MediaFormat.MIMETYPE_VIDEO_AVC).videoCapabilities }
            finally { it.release() }
        }
        var outW = w; var outH = h
        fun aligned(v: Int, a: Int) = v / a * a
        while (true) {
            val ew = aligned(outW, vcaps.widthAlignment); val eh = aligned(outH, vcaps.heightAlignment)
            val ok = ew >= vcaps.supportedWidths.lower && eh >= vcaps.supportedHeights.lower &&
                runCatching { vcaps.areSizeAndRateSupported(ew, eh, fps.toDouble()) }.getOrDefault(false)
            if (ok) { outW = ew; outH = eh; break }
            if (outW < 64 || outH < 64) error("this device's video encoder takes no size of ${w}x$h")
            outW = outW * 7 / 8; outH = outH * 7 / 8
        }
        val frame0 = if (outW == w && outH == h) photo
                     else NativePipe.resizeBgr(photo, w, h, outW, outH) ?: error("resize failed")
        if (outW != w || outH != h) onLog("video is ${outW}x$outH (the photo is ${w}x$h)")
        if (driverPath != null && !NativePipe.setPortrait(frame0, outW, outH))
            error(NativePipe.lastError())

        // The driver, decoded upright at no more than [fps].
        var extractor: MediaExtractor? = null
        var decoder: MediaCodec? = null
        var dw = 0; var dh = 0; var rotation = 0; var rate = fps; var inRate = fps; var expected: Int
        var driverAudio = false
        if (driverPath != null) {
            val ex = MediaExtractor().apply { setDataSource(driverPath) }
            var vt = -1
            for (i in 0 until ex.trackCount) {
                val m = ex.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: continue
                if (m.startsWith("video/") && vt < 0) vt = i
                if (m.startsWith("audio/")) driverAudio = true
            }
            if (vt < 0) { ex.release(); error("the driver has no video track") }
            val vf = ex.getTrackFormat(vt)
            dw = vf.getInteger(MediaFormat.KEY_WIDTH); dh = vf.getInteger(MediaFormat.KEY_HEIGHT)
            rotation = (if (vf.containsKey(MediaFormat.KEY_ROTATION)) vf.getInteger(MediaFormat.KEY_ROTATION)
                        else runCatching {
                            MediaMetadataRetriever().run {
                                setDataSource(driverPath)
                                val r = extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                                    ?.toIntOrNull() ?: 0
                                release(); r
                            }
                        }.getOrDefault(0)).let { ((it % 360) + 360) % 360 }
            val inFps = if (vf.containsKey(MediaFormat.KEY_FRAME_RATE)) vf.getInteger(MediaFormat.KEY_FRAME_RATE) else 30
            rate = minOf(fps, inFps)
            inRate = inFps
            val durUs = if (vf.containsKey(MediaFormat.KEY_DURATION)) vf.getLong(MediaFormat.KEY_DURATION) else 0L
            expected = ((durUs / 1_000_000.0) * rate).toInt().coerceAtLeast(1)
            ex.selectTrack(vt)
            val dec = MediaCodec.createDecoderByType(vf.getString(MediaFormat.KEY_MIME)!!)
            vf.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            dec.configure(vf, null, null, 0)
            dec.start()
            extractor = ex; decoder = dec
            onLog("driver ${dw}x$dh" + (if (rotation != 0) " rot $rotation" else "") +
                  " @ ${inFps}fps -> $rate fps, ~$expected frames")
        } else {
            expected = 0
        }

        // The voice: drives the mouth, and is the soundtrack when there is one.
        var syncing = false
        if (voice != null && NativePipe.hasLipSyncer()) {
            val pcm = AudioDecoder.decode(voice, voiceTrimStartUs, voiceTrimEndUs)
            when {
                pcm == null || pcm.frames == 0 -> onLog("lip sync: the voice has no audio, skipping")
                !NativePipe.setAudio(pcm.samples, pcm.channels, pcm.sampleRate, rate.toDouble()) ->
                    onLog("lip sync: ${NativePipe.lastError()}, skipping")
                else -> {
                    syncing = true
                    if (driverPath == null) expected = (pcm.seconds * rate).toInt().coerceAtLeast(1)
                    onLog("lip sync: %.1fs of voice, %d windows".format(pcm.seconds, NativePipe.melWindowTotal()))
                }
            }
        } else if (voice != null) onLog("lip sync: model not installed, skipping")
        if (driverPath == null && !syncing) error("no voice to make a talking photo from")

        // The soundtrack, decoded and encoded up front: the muxer takes no track after start.
        val audio = runCatching {
            when {
                voice != null && syncing -> VideoSwapper().transcodeToAac(voice, voiceTrimStartUs, voiceTrimEndUs)
                driverPath != null && driverAudio -> VideoSwapper().transcodeToAac(driverPath, 0L, Long.MAX_VALUE)
                else -> null
            }
        }.getOrNull()
        if (audio == null && (syncing || driverAudio)) onLog("audio: could not be prepared, writing video only")

        val encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        val bitrate = (outW.toLong() * outH * rate / 4L).coerceIn(64_000L, Int.MAX_VALUE.toLong()).toInt()
            .coerceIn(vcaps.bitrateRange.lower, vcaps.bitrateRange.upper)
        runCatching {
            encoder.configure(MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, outW, outH).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
                setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
                setInteger(MediaFormat.KEY_FRAME_RATE, rate)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2)
            }, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            encoder.start()
        }.onFailure { e ->
            encoder.release(); decoder?.release(); extractor?.release()
            error(codecWhy(e, "encoder refused ${outW}x$outH @ ${rate}fps"))
        }
        val muxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        var videoTrack = -1; var audioTrack = -1; var muxing = false
        fun drain(endOfStream: Boolean) {
            val info = MediaCodec.BufferInfo()
            while (true) {
                val ix = encoder.dequeueOutputBuffer(info, if (endOfStream) 10_000 else 0)
                when {
                    ix == MediaCodec.INFO_TRY_AGAIN_LATER -> if (!endOfStream) return
                    ix == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        videoTrack = muxer.addTrack(encoder.outputFormat)
                        if (audio != null) audioTrack = runCatching { muxer.addTrack(audio.first) }.getOrDefault(-1)
                        muxer.start(); muxing = true
                    }
                    ix >= 0 -> {
                        val out = encoder.getOutputBuffer(ix)!!
                        if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) info.size = 0
                        if (info.size > 0 && muxing) {
                            out.position(info.offset); out.limit(info.offset + info.size)
                            muxer.writeSampleData(videoTrack, out, info)
                        }
                        val eos = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        encoder.releaseOutputBuffer(ix, false)
                        if (eos) return
                    }
                }
            }
        }

        NativePipe.resetStats()
        val start = System.currentTimeMillis()
        val frameUs = 1_000_000L / rate
        var written = 0
        var cancelled = false
        fun emit(driverBgr: ByteArray?, ptsUs: Long) {
            val frame = frame0.copyOf()
            if (NativePipe.portraitFrame(driverBgr, dw.takeIf { rotation % 180 == 0 } ?: dh,
                                         dh.takeIf { rotation % 180 == 0 } ?: dw, frame, outW, outH,
                                         if (syncing) written else -1) < 0)
                error("native: ${NativePipe.lastError()}")
            onFrame(frame, outW, outH)
            val ix = encoder.dequeueInputBuffer(100_000)
            if (ix >= 0) queueBgrFrame(encoder, ix, frame, outW, outH, ptsUs, onLog)
            else onLog("encoder took no input for 100 ms: one frame dropped")
            written++
            onProgress(written, maxOf(expected, written))
            drain(false)
        }

        if (driverPath == null) {
            for (i in 0 until expected) {
                if (isCancelled()) { cancelled = true; break }
                emit(null, i * frameUs)
            }
        } else {
            val ex = extractor!!; val dec = decoder!!
            val info = MediaCodec.BufferInfo()
            var inEos = false; var outEos = false; var lastSlot = -1L; var t0 = -1L
            while (!outEos) {
                if (isCancelled()) { cancelled = true; break }
                if (!inEos) {
                    val ii = dec.dequeueInputBuffer(10_000)
                    if (ii >= 0) {
                        val n = ex.readSampleData(dec.getInputBuffer(ii)!!, 0)
                        if (n < 0) { dec.queueInputBuffer(ii, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM); inEos = true }
                        else { dec.queueInputBuffer(ii, 0, n, ex.sampleTime, 0); ex.advance() }
                    }
                }
                val oi = dec.dequeueOutputBuffer(info, 10_000)
                if (oi >= 0) {
                    if (info.size > 0) {
                        if (t0 < 0) t0 = info.presentationTimeUs
                        val rel = info.presentationTimeUs - t0
                        // Decimate only when slowing down, and to the NEAREST slot: a plain
                        // floor puts 41666 us * 24 at 0.99998 and drops every other frame.
                        val slot = if (rate >= inRate) lastSlot + 1
                                   else (rel * rate + 500_000L) / 1_000_000L
                        if (slot > lastSlot) {
                            lastSlot = slot
                            val image = dec.getOutputImage(oi)
                            if (image != null) {
                                val p = image.planes
                                fun bytes(b: ByteBuffer): ByteArray { val a = ByteArray(b.remaining()); b.get(a); return a }
                                val bgr = NativePipe.yuvToBgr(bytes(p[0].buffer), p[0].rowStride,
                                    bytes(p[1].buffer), p[1].rowStride, p[1].pixelStride,
                                    bytes(p[2].buffer), p[2].rowStride, p[2].pixelStride, dw, dh)
                                image.close()
                                val upright = if (rotation == 0) bgr else NativePipe.rotateBgr(bgr, dw, dh, rotation)
                                emit(upright, if (rate >= inRate) rel else slot * frameUs)
                            }
                        }
                    }
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outEos = true
                    dec.releaseOutputBuffer(oi, false)
                }
            }
            dec.stop(); dec.release(); ex.release()
        }
        if (cancelled) onLog("cancelled after $written frames -- keeping them")
        val ei = encoder.dequeueInputBuffer(100_000)
        if (ei >= 0) encoder.queueInputBuffer(ei, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
        drain(true)
        if (!muxing) {
            encoder.stop(); encoder.release(); muxer.release(); File(outputPath).delete()
            error(if (cancelled) "cancelled before any frame was written" else "no frames encoded")
        }
        // The soundtrack, cut to the video's length.
        val lengthUs = written * frameUs
        if (audio != null && audioTrack >= 0) runCatching {
            val ai = MediaCodec.BufferInfo()
            var n = 0
            for (pk in audio.second) {
                if (pk.ptsUs > lengthUs) break
                ai.offset = 0; ai.size = pk.bytes.size; ai.presentationTimeUs = pk.ptsUs
                ai.flags = MediaCodec.BUFFER_FLAG_KEY_FRAME
                muxer.writeSampleData(audioTrack, ByteBuffer.wrap(pk.bytes), ai); n++
            }
            onLog("audio: $n AAC packets written" + (if (voice != null && syncing) " (the voice)" else " (the driver's)"))
        }.onFailure { onLog("audio: writing it failed (${it.javaClass.simpleName}), video kept") }
        encoder.stop(); encoder.release()
        muxer.stop(); muxer.release()
        val ms = System.currentTimeMillis() - start
        onLog((if (cancelled) "partial: " else "") + "wrote ${File(outputPath).length() / 1024} KB, $written frames")
        if (written > 0 && ms > 0)
            onLog("%.1f fps  (%.1f ms/frame, %d frames in %.1fs)".format(written * 1000.0 / ms,
                  ms.toDouble() / written, written, ms / 1000.0))
        NativePipe.stageMillis().takeIf { it.isNotEmpty() }?.let(onLog)
        outputPath
    }
}
