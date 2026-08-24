package com.facefusion.mobile

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import java.nio.ByteBuffer

/**
 * Sampled frames straight out of the decoder, as BGR.
 *
 * A second opinion for the content gate, for files [android.media.MediaMetadataRetriever]
 * will not give frames for. The retriever is a convenience wrapper and returns null for
 * timestamps that MediaCodec decodes without complaint; this is the same
 * MediaExtractor + MediaCodec path [VideoSwapper] uses for the swap itself, so anything the
 * swapper can process, the gate can now sample.
 *
 * ⚠ It is a fallback, NOT a rescue. If the codec cannot configure the stream at all -- the
 * clip in work/assets is `video/mp4v-es` at 1366x2160, which is far outside that profile's
 * limits and is rejected outright -- then the file is undecodable, the swapper would fail
 * on it too, and refusing it is the correct answer rather than a false positive.
 *
 * The retriever stays the first choice: it seeks to keyframes and costs no full decode,
 * which is ~11 seeks for a 10 s clip against decoding the whole thing.
 */
object VideoFrames {

    /**
     * Decode [path], handing back one frame per [intervalUs] of presentation time.
     *
     * The first decoded frame is always taken, whatever its timestamp: a clip shorter than
     * one interval still has to be sampled once, and that is exactly the case the retriever
     * was failing on.
     *
     * @return how many frames were handed to [onFrame]; 0 means the file could not be
     *         decoded, which the caller must treat as "unknown", never as "fine".
     */
    fun sample(
        path: String,
        intervalUs: Long,
        maxFrames: Int = 64,
        onFrame: (bgr: ByteArray, w: Int, h: Int) -> Unit,
    ): Int {
        var extractor: MediaExtractor? = null
        var decoder: MediaCodec? = null
        var taken = 0
        try {
            extractor = MediaExtractor().apply { setDataSource(path) }
            var track = -1
            var format: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                if (f.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true) {
                    track = i; format = f; break
                }
            }
            val fmt = format ?: run {
                android.util.Log.w("ffframes", "no video track in " + path)
                return 0
            }
            extractor.selectTrack(track)

            val width = fmt.getInteger(MediaFormat.KEY_WIDTH)
            val height = fmt.getInteger(MediaFormat.KEY_HEIGHT)
            fmt.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)

            decoder = MediaCodec.createDecoderByType(fmt.getString(MediaFormat.KEY_MIME)!!)
            decoder.configure(fmt, null, null, 0)
            decoder.start()

            val info = MediaCodec.BufferInfo()
            var nextSampleUs = 0L
            var sawInputEOS = false
            var sawOutputEOS = false

            while (!sawOutputEOS && taken < maxFrames) {
                if (!sawInputEOS) {
                    val inIx = decoder.dequeueInputBuffer(10_000)
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

                val outIx = decoder.dequeueOutputBuffer(info, 10_000)
                if (outIx >= 0) {
                    // Take the first frame unconditionally, then one per interval.
                    val due = taken == 0 || info.presentationTimeUs >= nextSampleUs
                    if (info.size > 0 && due) {
                        decoder.getOutputImage(outIx)?.let { image ->
                            val p = image.planes
                            fun bytes(b: ByteBuffer): ByteArray {
                                val a = ByteArray(b.remaining()); b.get(a); return a
                            }
                            val bgr = NativePipe.yuvToBgr(
                                bytes(p[0].buffer), p[0].rowStride,
                                bytes(p[1].buffer), p[1].rowStride, p[1].pixelStride,
                                bytes(p[2].buffer), p[2].rowStride, p[2].pixelStride,
                                width, height,
                            )
                            image.close()
                            onFrame(bgr, width, height)
                            taken++
                            nextSampleUs = info.presentationTimeUs + intervalUs
                        }
                    }
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) sawOutputEOS = true
                    decoder.releaseOutputBuffer(outIx, false)
                }
            }
        } catch (t: Throwable) {
            // A codec that will not come up is "unknown", and 0 says so. The throwable is
            // logged rather than rethrown: the caller's contract is the count, but a silent
            // 0 is indistinguishable from "the file is empty" and that cost a debug cycle.
            android.util.Log.w("ffframes", "decode failed for " + path, t)
            return taken
        } finally {
            runCatching { decoder?.stop() }
            runCatching { decoder?.release() }
            runCatching { extractor?.release() }
        }
        android.util.Log.i("ffframes", "decoded " + taken + " frame(s) from " + path)
        return taken
    }
}
