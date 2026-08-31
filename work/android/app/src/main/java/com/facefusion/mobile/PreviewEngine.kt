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

    /**
     * ⚠ [MediaMetadataRetriever] IS NOT THREAD SAFE, and this one is reached from several
     * coroutines at once.
     *
     * Scrubbing cancels the previous seek job and starts another, but cancellation is
     * cooperative and `getFrameAtTime` is a long blocking native call that does not observe
     * it -- so the old seek is still inside the retriever when the new one enters. Two
     * concurrent reads of one MMR return null or the wrong frame, which surfaces as a pane
     * that just does not move when you drag the handle. [swap] can enter it from a third
     * place, and [closeTarget] can `release()` it from under all of them, which is not a
     * wrong frame but a native crash.
     *
     * A plain monitor rather than a coroutine Mutex: it has to cover open and close too,
     * and those are called from non-suspending code. Every read already runs on
     * Dispatchers.IO, so blocking here costs a queued seek, not a stalled frame.
     */
    private val mmrLock = Any()

    /**
     * Frame count and duration of the open target, or 0 when the file does not say.
     *
     * Read once at open rather than per seek: [frameAt] needs both to turn a millisecond
     * into a frame INDEX, and asking the retriever on every drag would put two metadata
     * extractions in front of every seek.
     */
    private var frameCount = 0
    private var durationMs = 0f

    /**
     * Point the engine at the copied-out target. Replaces any previously opened one.
     *
     * [fps] is the container's own rate, which the caller has already read off the
     * MediaFormat. It is a FALLBACK for the frame count: METADATA_KEY_VIDEO_FRAME_COUNT is
     * absent on plenty of files, and without a count there is no index to ask for.
     */
    fun openTarget(path: String, fps: Int = 0) {
        synchronized(mmrLock) {
            runCatching { mmr?.release() }
            mmr = runCatching {
                MediaMetadataRetriever().apply { setDataSource(path) }
            }.getOrNull()
            runCatching { seeker?.close() }
            seeker = FrameSeeker.open(path)
            frameCount = runCatching {
                mmr?.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)
                    ?.toInt() ?: 0
            }.getOrDefault(0)
            durationMs = runCatching {
                mmr?.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                    ?.toFloat() ?: 0f
            }.getOrDefault(0f)
            // Derived when the container does not publish a count. duration x fps is what
            // the frame count MEANS, and the app already had to read fps to size the run.
            if (frameCount <= 0 && fps > 0 && durationMs > 0f)
                frameCount = (durationMs / 1000f * fps).toInt()
            android.util.Log.d("ffpreview", "openTarget frames=" + frameCount +
                " durationMs=" + durationMs + " fps=" + fps)
        }
    }

    fun closeTarget() {
        synchronized(mmrLock) {
            runCatching { mmr?.release() }
            mmr = null
            runCatching { seeker?.close() }
            seeker = null
        }
    }

    /**
     * The exact-frame decoder, used in preference to the retriever.
     *
     * See [FrameSeeker] for why it has to exist. Kept as a fallback rather than a
     * replacement only because a file it cannot open at all is better served by an
     * approximate frame than by none.
     */
    private var seeker: FrameSeeker? = null

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
        val bmp = synchronized(mmrLock) {
            val r = mmr ?: return@synchronized null
            // DECODED first: the only method that is exact on every file. See FrameSeeker.
            val decoded = seeker?.let { fs ->
                runCatching { fs.frameAt((timeMs * 1000).toLong()) }.getOrNull()?.let { bgr ->
                    val w = fs.outWidth
                    val h = fs.outHeight
                    val long = maxOf(w, h)
                    val sc = if (long > MAX_EDGE) MAX_EDGE.toFloat() / long else 1f
                    val dw = (w * sc).toInt().coerceAtLeast(1)
                    val dh = (h * sc).toInt().coerceAtLeast(1)
                    // Scaled during the colour conversion rather than after it: a 4K frame
                    // is 33 MB of ARGB and this path would otherwise allocate it twice.
                    runCatching {
                        Bitmap.createBitmap(NativePipe.bgrToArgb(bgr, w, h, dw, dh),
                                            dw, dh, Bitmap.Config.ARGB_8888)
                    }.getOrNull()
                }
            }
            if (decoded != null) return@synchronized decoded

            // BY INDEX first, because OPTION_CLOSEST is only a REQUEST.
            //
            // ⚠ Measured on the bench 2026-08-31: seeks to 1646 ms and 3616 ms returned
            // byte-identical frames in freshly allocated Bitmaps -- the retriever snapped
            // both to the same sync frame -- while 8838 ms, in the next GOP, differed. That
            // is a preview that will not follow the handle for most of a drag, and it is
            // not a UI fault: the swap ran on each one and produced a correct result for
            // the frame it was given.
            //
            // getFrameAtTime's option is advisory and a hardware-backed retriever may
            // ignore it; getFrameAtIndex is exact by contract. It needs a frame COUNT, which
            // not every container carries, so OPTION_CLOSEST stays as the fallback -- on a
            // file without the metadata it is still the best of the time-based options.
            val byIndex = if (frameCount > 0 && durationMs > 0f) runCatching {
                val idx = ((timeMs / durationMs) * frameCount).toInt()
                    .coerceIn(0, frameCount - 1)
                val b = r.getFrameAtIndex(idx)
                android.util.Log.d("ffpreview", "  byIndex " + idx + "/" + frameCount +
                    " -> " + (if (b == null) "NULL" else "ok"))
                b
            }.getOrElse {
                android.util.Log.d("ffpreview", "  byIndex threw: " + it)
                null
            } else {
                android.util.Log.d("ffpreview", "  no frame count; time-based seek")
                null
            }
            byIndex ?: runCatching {
                r.getFrameAtTime((timeMs * 1000).toLong(),
                                 MediaMetadataRetriever.OPTION_CLOSEST)
            }.getOrNull()
        }
        bmp?.let { capped(it) }
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
     * WHICH SWAPPER is loaded, or null when nothing is.
     *
     * ⚠ The one thing that actually decides what is in memory. It selects a different
     * context binary -- hyperswap is 256 px and inswapper 128, with different normalisation
     * -- so changing it is the only option change that still costs a reload.
     *
     * This used to be the whole `SwapOptions`, on the reasoning that `NativePipe.init`
     * takes the options once and the native side keeps its own copy. The first half is
     * true; the conclusion was not. Every OTHER field is read once per frame inside
     * ffpipe, so it can simply be pushed to a loaded pipeline -- see [applyOptions].
     */
    private var loadedSwapper: String? = null

    /** The options the loaded pipeline currently has, as last pushed or loaded. */
    private var loadedOpts: SwapOptions? = null

    /**
     * Whose face is currently set on the loaded pipeline.
     *
     * ⚠ SEPARATE from [loadedOpts], and that separation is the point. These were one
     * `WarmKey(opts, sourceTag)`, so changing the SOURCE tore down and reloaded every model
     * -- and no model depends on the source. Only `setSource` does: it analyses one image
     * and keeps the embedding. Reloading ~300 MB of contexts to compute a 512-float vector
     * is work the options genuinely need and the source never did.
     *
     * The comment above is the reasoning that was over-applied: it is true of the options,
     * because init takes them once, and it was extended to the source by proximity.
     */
    private var appliedSource: Any? = null

    /** True when a pipeline is loaded AND a source face has been applied to it. */
    val isWarm: Boolean get() = loadedSwapper != null && appliedSource != null

    fun invalidate() {
        if (loadedSwapper != null) NativePipe.release()
        loadedSwapper = null
        loadedOpts = null
        appliedSource = null
    }

    /**
     * Push per-frame options onto a pipeline that is already loaded.
     *
     * Cheap enough to call unconditionally: it returns immediately when nothing is loaded,
     * when nothing changed, or when the change is one only a reload can serve. That last
     * case is left to [ensureReady] rather than handled here, so there is exactly one place
     * that can load a model.
     */
    fun applyOptions(opts: SwapOptions) {
        if (loadedSwapper == null || loadedSwapper != opts.swapper) return
        if (loadedOpts == opts) return
        if (NativePipe.setOptions(opts)) loadedOpts = opts
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
        // The SMALLEST job that is out of date, of three, in increasing cost:
        //   options  -> push them to the loaded pipeline, no I/O at all
        //   source   -> one gate check and one setSource
        //   swapper  -> a genuine reload, because it is a different model file
        val reload = loadedSwapper != opts.swapper
        if (reload) {
            invalidate()
            if (!NativePipe.init(libDir, libDir, modelDir, opts))
                return@withContext "Could not load the models: " + NativePipe.lastError()
            loadedSwapper = opts.swapper
            loadedOpts = opts
        } else {
            applyOptions(opts)
            if (appliedSource == sourceTag) return@withContext null
        }

        // Dropped BEFORE the attempt, not after it: everything below can fail, and a stale
        // claim about whose face is loaded is worse than no claim.
        appliedSource = null

        gate?.invoke()?.let {
            // Refused, or the check could not run. Either way nothing further happens and
            // the pipeline does not stay warm, so the next attempt re-checks rather than
            // inheriting a verdict.
            invalidate()
            return@withContext it
        }

        if (!NativePipe.setSource(sourceBgr, sourceW, sourceH)) {
            // ⚠ Released, not merely left unapplied, even though the models are fine and a
            // reload is what this change exists to avoid. `ffpipe::setSource` does not clear
            // the previous identity when it finds no face -- it returns false with the old
            // embedding still in place -- so a pipeline that survived a failed source change
            // is one that would swap the PREVIOUS person's face. Reloading is the cheap half
            // of that trade.
            invalidate()
            return@withContext "No face found in the source image"
        }
        appliedSource = sourceTag
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
        if (!isWarm) return@withContext Swapped(null, 0, "Pipeline is not loaded")
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
