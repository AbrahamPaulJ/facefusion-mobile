package com.facefusion.mobile

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * The two preview panes: the target frame under the trim handle, and that same frame swapped.
 *
 * The two halves have wildly different costs, and the whole design follows from that:
 *
 *  * [frameAt] is a video seek, tens of milliseconds, safe to run whenever the handle moves;
 *  * [swap] needs the pipeline loaded, and loading it reads ~266 MB of context binaries off
 *    disk. Once warm, a frame costs about what one video frame costs during a real run.
 *
 * So the pipeline is kept WARM across scrubs and invalidated only when something that feeds
 * `NativePipe.init` actually changes. The caller decides when to pay for the first warm-up.
 */
class PreviewEngine {

    /** What a swap preview produced. [faces] is 0 when nothing was detected. */
    data class Swapped(val bitmap: Bitmap?, val faces: Int, val error: String? = null)

    // ------------------------------------------------------------------ the target frame

    private var mmr: MediaMetadataRetriever? = null

    /** Point the engine at the copied-out target. Replaces any previously opened one. */
    fun openTarget(path: String) {
        closeTarget()
        mmr = runCatching {
            MediaMetadataRetriever().apply { setDataSource(path) }
        }.getOrNull()
    }

    fun closeTarget() {
        runCatching { mmr?.release() }
        mmr = null
    }

    /**
     * The frame the swap would actually begin at.
     *
     * ⚠ `OPTION_CLOSEST`, and it has to be. VideoSwapper seeks to the previous sync frame
     * but then DROPS everything before the mark (`pts >= trimStartUs`), so the first frame
     * it encodes is the exact one under the handle, not the keyframe behind it.
     *
     * Using `OPTION_PREVIOUS_SYNC` here was a bug with two faces: it previewed a frame the
     * output would not contain, and because it snaps to keyframes it returned the SAME
     * image for every position inside a GOP -- so on footage with sparse keyframes the
     * preview simply appeared frozen while the handle moved. It is the slower option, which
     * is what the debounce upstream of this is for.
     */
    suspend fun frameAt(timeMs: Float): Bitmap? = withContext(Dispatchers.IO) {
        val r = mmr ?: return@withContext null
        runCatching {
            r.getFrameAtTime((timeMs * 1000).toLong(),
                             MediaMetadataRetriever.OPTION_CLOSEST)
        }.getOrNull()?.let { capped(it) }
    }

    /**
     * Cap the long edge before anything touches the pixels.
     *
     * Not for speed -- the swapper's cost is fixed by the 256² graph, not the frame -- but
     * because a 4K frame is 33 MB of ARGB per copy and this path makes several.
     */
    private fun capped(b: Bitmap): Bitmap {
        val long = maxOf(b.width, b.height)
        if (long <= MAX_EDGE) return b
        val s = MAX_EDGE.toFloat() / long
        val w = (b.width * s).toInt().coerceAtLeast(1)
        val h = (b.height * s).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(b, w, h, true)
    }

    // ------------------------------------------------------------------- the warm pipeline

    /**
     * What the loaded pipeline was built for. When this changes, the pipeline is wrong and
     * has to be torn down -- `NativePipe.init` takes the options ONCE and the native side
     * keeps its own copy for the duration, so editing a slider cannot reach an already
     * loaded pipeline.
     */
    private data class WarmKey(val opts: SwapOptions, val sourceTag: Any?)

    private var warmKey: WarmKey? = null

    /** True when a pipeline is loaded and matches the current options and source. */
    val isWarm: Boolean get() = warmKey != null

    fun invalidate() {
        if (warmKey != null) {
            NativePipe.release()
            warmKey = null
        }
    }

    /**
     * Load the pipeline for [opts] and [source], unless it is already loaded for exactly that.
     *
     * [sourceTag] identifies the source image; anything with sane equality will do (the Uri
     * is what the caller has). Returns null on success, an error string otherwise.
     *
     * [gate] runs AFTER the pipeline comes up and BEFORE the source is read, and a non-null
     * return aborts without reading it. That ordering is forced: a content check needs the
     * models loaded before it can run at all, but `setSource` is already processing -- it
     * detects, aligns and embeds the face. Anything that must not be processed has to be
     * refused in the gap between those two, which is what this hook is.
     */
    suspend fun ensureReady(
        libDir: String,
        modelDir: String,
        opts: SwapOptions,
        sourceTag: Any?,
        sourceBgr: ByteArray,
        sourceW: Int,
        sourceH: Int,
        gate: (suspend () -> String?)? = null,
    ): String? = withContext(Dispatchers.Default) {
        val want = WarmKey(opts, sourceTag)
        if (warmKey == want) return@withContext null
        invalidate()

        if (!NativePipe.init(libDir, libDir, modelDir, opts))
            return@withContext "Could not load the models: " + NativePipe.lastError()

        gate?.invoke()?.let {
            // Refused, or the check could not run. Either way nothing further happens and
            // the pipeline does not stay warm, so the next attempt re-checks rather than
            // inheriting a verdict.
            NativePipe.release()
            return@withContext it
        }

        if (!NativePipe.setSource(sourceBgr, sourceW, sourceH)) {
            NativePipe.release()
            return@withContext "No face found in the source image"
        }
        warmKey = want
        null
    }

    /**
     * Swap [frame], which must be a frame from [frameAt].
     *
     * ⚠ `processFrame` writes the result back into the array ONLY when it found a face, so
     * a zero-face frame comes back byte-identical to the input. Rendering that as "the
     * swapped frame" would show an unswapped image with no hint why, which is exactly the
     * kind of silent wrongness this preview exists to catch -- hence [Swapped.faces].
     */
    suspend fun swap(frame: Bitmap): Swapped = withContext(Dispatchers.Default) {
        if (warmKey == null) return@withContext Swapped(null, 0, "Pipeline is not loaded")
        runCatching {
            val soft = frame.copy(Bitmap.Config.ARGB_8888, false)
            val w = soft.width
            val h = soft.height
            val px = IntArray(w * h)
            soft.getPixels(px, 0, w, 0, 0, w, h)

            val bgr = NativePipe.argbToBgr(px, w, h)
            val faces = NativePipe.processFrame(bgr, w, h)
            if (faces < 0) return@runCatching Swapped(null, 0, NativePipe.lastError())
            if (faces == 0) return@runCatching Swapped(null, 0, null)

            val out = NativePipe.bgrToArgb(bgr, w, h, w, h)
            Swapped(Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888), faces)
        }.getOrElse { Swapped(null, 0, it.message ?: "preview failed") }
    }

    /** Tear everything down. Safe to call repeatedly. */
    fun release() {
        invalidate()
        closeTarget()
    }

    private companion object {
        const val MAX_EDGE = 1920
    }
}
