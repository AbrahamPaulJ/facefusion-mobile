package com.facefusion.mobile

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import java.io.File

/**
 * The NSFW content gate, with upstream's policy.
 *
 * It **blocks**: FaceFusion's `content_analyser.py` refuses to process flagged content and
 * so does this port (decided 2026-08-24). Everything here is upstream's rule, ported:
 *
 *  * a still is checked once ([checkImage], `analyse_image`);
 *  * a video is sampled **one frame per second** and refused when more than **10%** of the
 *    samples trip the gate ([checkVideo], `analyse_video`) -- not every frame, which is
 *    what makes a ~5 ms graph cost ~56 ms for a whole clip instead of 1.5 s.
 *
 * ⚠ **This port gates on `nsfw_2` alone.** Upstream votes 2-of-3 across `nsfw_1`, `nsfw_2`
 * and `nsfw_3`, which together are 461 MB against a 266 MB app. A single model cannot
 * reproduce a majority vote, and which way it errs is unmeasured. See docs/roadmap.md 2.
 *
 * ⚠ **A failure to measure is not permission.** `NativePipe.contentScore` returns NaN when
 * the graph did not run, and every verdict below treats NaN as [Verdict.ERROR], never as
 * "allow". A blocking gate that opens when it breaks is worse than no gate, because
 * everything upstream of it believes the content was checked.
 */
object ContentGate {

    /** content_analyser.py:detect_with_nsfw_2 -- flagged above this. */
    const val THRESHOLD = 0.25f

    /** content_analyser.py:analyse_video -- refuse the video above this percentage. */
    const val VIDEO_RATE_PERCENT = 10.0

    /** content_analyser.py:analyse_video -- one sample per second of footage. */
    private const val SAMPLE_INTERVAL_US = 1_000_000L

    /**
     * How far the W8A16 gate's score sits from the fp32 one, measured over 16 held-out
     * frames: +0.075 mean, +0.153 max, **16 of 16 in the same direction**. Only the lower
     * tiers use it -- the fp32 context will not finalize below v79.
     *
     * Reported, never subtracted. Compensating would be a correction constant nobody has
     * tested against content near the threshold, and it would hide the divergence.
     */
    const val QUANTISED_BIAS = 0.087f

    enum class Verdict { ALLOW, BLOCK, ERROR }

    data class Result(
        val verdict: Verdict,
        /** The decision statistic, or the worst one seen across a video's samples. */
        val score: Float,
        /** Videos only: how many samples were taken and how many were flagged. */
        val sampled: Int = 1,
        val flagged: Int = 0,
        val detail: String = "",
    ) {
        val blocked get() = verdict == Verdict.BLOCK
        val ok get() = verdict == Verdict.ALLOW
    }

    private fun judge(score: Float): Verdict = when {
        score.isNaN() -> Verdict.ERROR
        score > THRESHOLD -> Verdict.BLOCK
        else -> Verdict.ALLOW
    }

    /** `analyse_image`: one check on a still. */
    /**
     * One BGR frame to a verdict. The single place a score becomes a decision, so both the
     * still path and both video paths cannot drift apart on how NaN is treated.
     */
    private fun checkBgr(bgr: ByteArray, w: Int, h: Int): Result {
        val score = NativePipe.contentScore(bgr, w, h)
        return Result(judge(score), score,
                      detail = if (score.isNaN()) NativePipe.lastError() else "")
    }

    fun checkImage(bitmap: Bitmap): Result {
        val soft = bitmap.copy(Bitmap.Config.ARGB_8888, false)
            ?: return Result(Verdict.ERROR, Float.NaN, detail = "cannot read image")
        val px = IntArray(soft.width * soft.height)
        soft.getPixels(px, 0, soft.width, 0, 0, soft.width, soft.height)
        return checkBgr(NativePipe.argbToBgr(px, soft.width, soft.height), soft.width, soft.height)
    }

    /**
     * `analyse_video`: sample one frame per second, refuse above a 10% flagged rate.
     *
     * Uses [MediaMetadataRetriever] rather than the decode loop the swap itself runs on:
     * the gate has to answer BEFORE any frame is processed or previewed, and seeking to
     * ~11 timestamps is far cheaper than a second full decode. `OPTION_CLOSEST_SYNC`
     * keeps it to keyframes, so this costs seeks, not decoding.
     */
    /**
     * `analyse_video`: sample one frame per second, refuse above a 10% flagged rate.
     *
     * Two samplers, in cost order. [MediaMetadataRetriever] first, because it seeks to
     * keyframes and never decodes a whole clip -- ~11 seeks for a 10 s video. When it
     * yields NOTHING it is not believed, and the decoder runs instead.
     *
     * ⚠ The retriever really does return null for every timestamp on some files this
     * project's own decoder reads happily -- a 1366x2160 four-frame clip among them, under
     * both CLOSEST_SYNC and CLOSEST. Treating that as "unreadable" refused videos the
     * swapper could process. Fail-closed, so never a way through the gate, but a way to be
     * told no about a good file.
     */
    fun checkVideo(file: File): Result =
        sampleByRetriever(file) ?: sampleByDecoder(file)

    /**
     * @return null when the retriever produced no frames at all, meaning "ask the decoder".
     *         A graph failure is NOT null -- a broken gate is broken in either sampler, and
     *         retrying it would only turn one honest ERROR into two.
     */
    private fun sampleByRetriever(file: File): Result? {
        val r = MediaMetadataRetriever()
        try {
            r.setDataSource(file.absolutePath)
            val durationMs = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: return null

            var sampled = 0
            var flagged = 0
            var worst = Float.NEGATIVE_INFINITY
            var us = 0L
            // `<=` and the max(): a clip shorter than one second still has to be checked.
            val endUs = maxOf(durationMs * 1000L, 1L)
            while (us <= endUs) {
                val frame = r.getFrameAtTime(us, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                    ?: r.getFrameAtTime(us, MediaMetadataRetriever.OPTION_CLOSEST)
                if (frame != null) {
                    val res = checkImage(frame)
                    if (res.verdict == Verdict.ERROR) return res
                    sampled++
                    if (res.blocked) flagged++
                    if (res.score > worst) worst = res.score
                }
                us += SAMPLE_INTERVAL_US
            }
            return if (sampled == 0) null else verdictOf(sampled, flagged, worst)
        } catch (e: Exception) {
            return null
        } finally {
            runCatching { r.release() }
        }
    }

    /** The fallback: the same MediaExtractor + MediaCodec path the swap itself decodes with. */
    private fun sampleByDecoder(file: File): Result {
        var sampled = 0
        var flagged = 0
        var worst = Float.NEGATIVE_INFINITY
        var graphError: Result? = null

        val taken = VideoFrames.sample(file.absolutePath, SAMPLE_INTERVAL_US) { bgr, w, h ->
            if (graphError == null) {
                val res = checkBgr(bgr, w, h)
                if (res.verdict == Verdict.ERROR) {
                    graphError = res
                } else {
                    sampled++
                    if (res.blocked) flagged++
                    if (res.score > worst) worst = res.score
                }
            }
        }
        graphError?.let { return it }
        if (taken == 0 || sampled == 0)
            return Result(Verdict.ERROR, Float.NaN, detail = "no frames could be read")
        return verdictOf(sampled, flagged, worst)
    }

    private fun verdictOf(sampled: Int, flagged: Int, worst: Float): Result {
        val rate = 100.0 * flagged / sampled
        return Result(
            if (rate > VIDEO_RATE_PERCENT) Verdict.BLOCK else Verdict.ALLOW,
            worst, sampled, flagged,
            "%d/%d flagged (%.1f%%)".format(flagged, sampled, rate),
        )
    }

    fun message(what: String, res: Result): String = when (res.verdict) {
        // The decision statistic is NOT in here. It goes to the log, where it is useful
        // for debugging; aimed at the person holding the phone it is only noise wrapped
        // around a refusal, and a threshold is an invitation to work out how to sit under
        // it.
        Verdict.BLOCK -> "Explicit content not allowed in $what"
        // Kept distinct on purpose. A gate that could not run has not refused anything and
        // has not permitted anything either, and saying "not allowed" here would blame the
        // user for our own failure.
        //
        // ⚠ The REASON is included, unlike the BLOCK case above. `detail` already carried
        // the native error -- "content gate: <qnn error>" -- and nothing ever showed it, so
        // a field report of this failure said only that it happened. That cost a day of
        // guessing at a bug the app could name itself. A refusal must not leak a threshold;
        // a FAULT should say what broke.
        Verdict.ERROR ->
            "The content check could not run on $what, so it cannot be processed" +
                (if (res.detail.isNotBlank()) " (${res.detail})" else "")
        Verdict.ALLOW -> ""
    }

    /**
     * A refusal, as opposed to a fault.
     *
     * Thrown so the UI can present it as the finished, user-facing sentence it already is,
     * rather than prefixing it with "Failed:" the way a genuine error deserves.
     */
    class Refused(message: String) : Exception(message)
}
