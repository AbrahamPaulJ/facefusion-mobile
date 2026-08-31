package com.facefusion.mobile

import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * PCM out of a container, for the lip syncer.
 *
 * `VideoSwapper` copies audio through COMPRESSED -- extractor to muxer, sample by sample,
 * never touching a PCM value -- which is right for a swap that does not look at sound and
 * useless for one that does. This is the other path: decode to 16-bit PCM so `ffaudio` can
 * take a mel spectrogram off it.
 *
 * Nothing here resamples or mixes down. `ffaudio::resampleToVoiceRate` and
 * `ffaudio::prepareAudio` do both, in C++, measured against upstream's numpy in
 * `work/native/test_ffaudio.py`. Two implementations of the same arithmetic on two sides
 * of the JNI boundary is how they drift.
 *
 * ⚠ The trim is applied HERE, on the same [trimStartUs, trimEndUs] the video path uses,
 * because the audio has to survive it in phase. `ffaudio::extractWindows` produces window
 * k for video frame k, and window k covers the 200 ms FOLLOWING frame k -- so if the audio
 * started at the clip's origin while the video started at the trim, every mouth would be
 * wrong by the trim offset, and it would look like a bad model rather than a bad index.
 *
 * ⚠ Decoder output is NOT always 16-bit. Some devices hand back
 * ENCODING_PCM_FLOAT or 8.24 fixed point; `KEY_PCM_ENCODING` says which, and it is absent
 * on the majority that use 16-bit. Guessing wrong gives noise that still sounds like
 * audio, so the format is read rather than assumed.
 */
object AudioDecoder {

    /** What the decoder produced: interleaved PCM plus the two facts ffaudio needs. */
    class Pcm(val samples: ShortArray, val sampleRate: Int, val channels: Int) {
        /** Frames, i.e. samples per channel. */
        val frames: Int get() = if (channels > 0) samples.size / channels else 0
        val seconds: Double get() = if (sampleRate > 0) frames.toDouble() / sampleRate else 0.0
    }

    private const val TIMEOUT_US = 10_000L

    /**
     * Decode the first audio track of [path] between [trimStartUs] and [trimEndUs].
     *
     * Returns null when the file has no audio track at all, which is not an error: a clip
     * with no sound simply cannot be lip synced, and the caller says so rather than
     * failing. Throws only on a decoder that cannot be configured.
     */
    fun decode(path: String, trimStartUs: Long = 0L, trimEndUs: Long = Long.MAX_VALUE): Pcm? {
        val extractor = MediaExtractor().apply { setDataSource(path) }
        try {
            var track = -1
            var format: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
                if (mime.startsWith("audio/")) { track = i; format = f; break }
            }
            val af = format ?: return null
            extractor.selectTrack(track)
            // Seek to the sync sample at or before the trim, exactly as the video path
            // does; samples before the mark are dropped below rather than decoded blind.
            if (trimStartUs > 0) {
                extractor.seekTo(trimStartUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
            }

            val mime = af.getString(MediaFormat.KEY_MIME)!!
            val codec = MediaCodec.createDecoderByType(mime)
            codec.configure(af, null, null, 0)
            codec.start()

            val out = ArrayList<ShortArray>()
            var totalShorts = 0
            var sampleRate = af.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var channels = af.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            var pcmEncoding = if (af.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                af.getInteger(MediaFormat.KEY_PCM_ENCODING)
            } else {
                android.media.AudioFormat.ENCODING_PCM_16BIT
            }

            val info = MediaCodec.BufferInfo()
            var sawInputEnd = false
            var sawOutputEnd = false

            while (!sawOutputEnd) {
                if (!sawInputEnd) {
                    val inIndex = codec.dequeueInputBuffer(TIMEOUT_US)
                    if (inIndex >= 0) {
                        val buf = codec.getInputBuffer(inIndex)!!
                        val n = extractor.readSampleData(buf, 0)
                        val pts = extractor.sampleTime
                        if (n < 0 || (pts > trimEndUs && trimEndUs != Long.MAX_VALUE)) {
                            codec.queueInputBuffer(inIndex, 0, 0, 0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            sawInputEnd = true
                        } else {
                            codec.queueInputBuffer(inIndex, 0, n, pts, 0)
                            extractor.advance()
                        }
                    }
                }

                val outIndex = codec.dequeueOutputBuffer(info, TIMEOUT_US)
                when {
                    outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        // The OUTPUT format is the authority, not the track format: a
                        // decoder may hand back a different rate, channel count or PCM
                        // encoding than the container advertised.
                        val of = codec.outputFormat
                        sampleRate = of.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        channels = of.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        if (of.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                            pcmEncoding = of.getInteger(MediaFormat.KEY_PCM_ENCODING)
                        }
                    }
                    outIndex >= 0 -> {
                        if (info.size > 0 && info.presentationTimeUs >= trimStartUs) {
                            val buf = codec.getOutputBuffer(outIndex)!!
                            buf.position(info.offset)
                            buf.limit(info.offset + info.size)
                            val chunk = toShorts(buf, info.size, pcmEncoding)
                            out.add(chunk)
                            totalShorts += chunk.size
                        }
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            sawOutputEnd = true
                        }
                        codec.releaseOutputBuffer(outIndex, false)
                    }
                }
            }
            codec.stop()
            codec.release()

            val merged = ShortArray(totalShorts)
            var at = 0
            for (chunk in out) {
                System.arraycopy(chunk, 0, merged, at, chunk.size)
                at += chunk.size
            }
            return Pcm(merged, sampleRate, channels)
        } finally {
            extractor.release()
        }
    }

    /** One decoded buffer to 16-bit samples, whatever the decoder chose to emit. */
    private fun toShorts(buf: ByteBuffer, size: Int, pcmEncoding: Int): ShortArray {
        buf.order(ByteOrder.nativeOrder())
        return when (pcmEncoding) {
            android.media.AudioFormat.ENCODING_PCM_FLOAT -> {
                val fb = buf.asFloatBuffer()
                ShortArray(fb.remaining()) {
                    val v = fb.get(it)
                    val clipped = if (v > 1f) 1f else if (v < -1f) -1f else v
                    (clipped * 32767f).toInt().toShort()
                }
            }
            android.media.AudioFormat.ENCODING_PCM_8BIT -> {
                // Unsigned, centred on 128 -- the one encoding where the zero is not zero.
                ShortArray(size) { (((buf.get(it).toInt() and 0xFF) - 128) * 256).toShort() }
            }
            else -> {
                val sb = buf.asShortBuffer()
                ShortArray(sb.remaining()) { sb.get(it) }
            }
        }
    }
}
