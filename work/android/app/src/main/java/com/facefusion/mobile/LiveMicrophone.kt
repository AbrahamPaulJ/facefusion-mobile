package com.facefusion.mobile

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import android.media.MediaRecorder
import java.io.File
import java.nio.ByteBuffer

/** AAC capture owned and serialized by one LiveRecorder. */
class LiveMicrophone(private val context: Context) {
    private var recorder: MediaRecorder? = null
    private var audio: File? = null
    private var offsetUs = 0L
    @Volatile private var failure: String? = null

    fun start(videoStartNs: Long) {
        val file = File.createTempFile("live_mic_", ".m4a", context.cacheDir)
        audio = file
        val rec = MediaRecorder(context)
        recorder = rec
        rec.setAudioSource(MediaRecorder.AudioSource.MIC)
        rec.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
        rec.setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
        rec.setAudioSamplingRate(44100)
        rec.setAudioChannels(1)
        rec.setAudioEncodingBitRate(128_000)
        rec.setOutputFile(file.absolutePath)
        rec.setOnErrorListener { _, what, extra -> failure = "Microphone error: $what/$extra" }
        rec.prepare()
        offsetUs = (System.nanoTime() - videoStartNs) / 1000
        rec.start()
    }

    fun stop() {
        val rec = recorder ?: return
        try {
            rec.stop()
            check(failure == null) { failure ?: "Microphone failed" }
        } finally {
            rec.release()
            recorder = null
        }
    }

    /** Remux without recompression, retaining the variable video timestamps. */
    fun mergeInto(video: File) {
        val merged = File.createTempFile("live_mux_", ".mp4", video.parentFile)
        val ve = MediaExtractor()
        val ae = MediaExtractor()
        var muxer: MediaMuxer? = null
        try {
            ve.setDataSource(video.absolutePath)
            ae.setDataSource(checkNotNull(audio).absolutePath)
            fun select(extractor: MediaExtractor, prefix: String): MediaFormat {
                val index = (0 until extractor.trackCount).first {
                    extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith(prefix) == true
                }
                extractor.selectTrack(index)
                return extractor.getTrackFormat(index)
            }
            val vf = select(ve, "video/")
            val af = select(ae, "audio/")
            val durationUs = vf.getLong(MediaFormat.KEY_DURATION)
            val writer = MediaMuxer(merged.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            muxer = writer
            val vt = writer.addTrack(vf)
            val at = writer.addTrack(af)
            writer.start()
            val info = MediaCodec.BufferInfo()
            var buffer = ByteBuffer.allocateDirect(1024 * 1024)
            var audioSamples = 0
            // Interleave tracks with at most one sample in memory.
            while (true) {
                val vTime = ve.sampleTime
                val aTime = if (ae.sampleTime < 0) -1 else ae.sampleTime + offsetUs
                val hasAudio = aTime >= 0 && aTime <= durationUs
                if (vTime < 0 && !hasAudio) break
                val isAudio = hasAudio && (vTime < 0 || aTime < vTime)
                val extractor = if (isAudio) ae else ve
                val size = extractor.sampleSize
                check(size in 1..Int.MAX_VALUE.toLong()) { "Invalid sample size: $size" }
                if (size > buffer.capacity()) buffer = ByteBuffer.allocateDirect(size.toInt())
                buffer.clear()
                val read = extractor.readSampleData(buffer, 0)
                check(read > 0) { "Could not read media sample" }
                val flags = if (extractor.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0)
                    MediaCodec.BUFFER_FLAG_KEY_FRAME else 0
                info.set(0, read, if (isAudio) aTime else vTime, flags)
                writer.writeSampleData(if (isAudio) at else vt, buffer, info)
                if (isAudio) audioSamples++
                extractor.advance()
            }
            check(audioSamples > 0) { "No microphone audio captured" }
            writer.stop()
            writer.release()
            muxer = null
            check(merged.renameTo(video)) { "Could not save recording with audio" }
        } finally {
            runCatching { muxer?.release() }
            ve.release()
            ae.release()
            merged.delete()
        }
    }

    fun close() {
        runCatching { recorder?.release() }
        recorder = null
        audio?.delete()
        audio = null
    }
}
