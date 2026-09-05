package com.facefusion.mobile

import android.content.Context
import android.graphics.Bitmap
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * The front camera, swapped, on screen. Dev builds only -- see [BuildConfig.DEV_BUILD].
 *
 * Preview ONLY: no encoder, no muxer, no audio. What it costs is therefore what the
 * pipeline costs, which is the point of having it -- 26.6 fps on a 720p file with tracking
 * is the number this has to live up to, minus capture and display.
 *
 * ## Why this is not just VideoSwapper with a camera on the front
 *
 * VideoSwapper owns a decode/encode loop and runs to completion. This is a pump: frames
 * arrive when the sensor produces them, the pipeline is slower than the sensor, and the
 * right answer to a backlog is to throw it away rather than fall further behind.
 * `STRATEGY_KEEP_ONLY_LATEST` does exactly that and also guarantees ONE frame in flight,
 * which is what lets the whole pipeline run on the analyzer thread with no lock of its own.
 *
 * ## Rotation
 *
 * The sensor hands back a landscape buffer with the face rotated 90 degrees in portrait,
 * and yoloface is not asked to find a sideways face -- `estimateFaceAngle` exists for faces
 * that are rotated in the FRAME, not for a frame that is rotated whole.
 * `setOutputImageRotationEnabled(true)` makes CameraX deliver it upright instead, which is
 * cheaper than rotating 2.7 MB per frame here and is the reason camera-core 1.3 is the
 * minimum.
 *
 * ⚠ MIRRORING IS THE DISPLAY'S JOB, NOT THIS CLASS'S. The pipeline sees the true image so
 * the detector gets a face the right way round; [ui.LiveScreen] flips only what is drawn.
 * Swapping a mirrored frame would feed a mirrored face to graphs that were never measured
 * on one.
 */
class LiveEngine {

    /** One frame's worth of result, handed to the UI. */
    data class Shot(val bitmap: Bitmap?, val faces: Int, val fps: Double, val error: String?)

    // Per-stage cost, logged every 30 frames. Live was 6.5 fps on its first run against
    // 26.6 on a file, and no amount of reasoning about which stage was to blame beat
    // asking -- the same lesson the geometry buckets taught.
    private var nStat = 0
    private var msPump = 0.0

    private var provider: ProcessCameraProvider? = null
    private var exec: ExecutorService? = null

    @Volatile private var running = false

    // TWO bitmaps, alternating, and both halves of that matter.
    //
    // Allocating one per frame is out: a 720p ARGB bitmap is 3.7 MB, so 25 fps is 92 MB/s of
    // pure churn. But reusing ONE is worse than it looks, twice over:
    //
    //   * Compose is handed the bitmap as state. Writing new pixels into the same instance
    //     leaves the reference equal, so recomposition never triggers and the feed simply
    //     freezes on frame one while the fps counter happily climbs.
    //   * setPixels would be writing into the exact buffer the compositor is drawing, which
    //     tears under the best of circumstances.
    //
    // Two buffers solve both: the reference changes every frame, and the one being written
    // is never the one on screen.
    private val bufs = arrayOfNulls<Bitmap>(2)
    private var bufIx = 0

    // Frame pacing, measured over a short window rather than since the start, so the number
    // on screen reacts when the phone throttles instead of averaging the throttling away.
    private var windowStart = 0L
    private var windowFrames = 0
    @Volatile private var fps = 0.0

    val isRunning: Boolean get() = running

    // Longest edge of the DISPLAYED bitmap. Above a phone screen's own width there is
    // nothing to see and everything to pay for.
    private val kMaxPreview = 1080

    /**
     * Binds the front camera and starts the pump.
     *
     * The pipeline must already be initialised and hold a source -- this class deliberately
     * does not own that: the same [NativePipe] is shared with the preview and the API, and
     * an engine that re-initialised it on every start would fight them for the models.
     */
    fun start(ctx: Context, owner: LifecycleOwner, onShot: (Shot) -> Unit) {
        if (running) return
        running = true
        windowStart = System.nanoTime(); windowFrames = 0
        val e = Executors.newSingleThreadExecutor()
        exec = e
        val future = ProcessCameraProvider.getInstance(ctx)
        future.addListener({
            if (!running) return@addListener
            val p = runCatching { future.get() }.getOrNull()
            if (p == null) {
                onShot(Shot(null, 0, 0.0, "camera unavailable"))
                running = false
                return@addListener
            }
            provider = p
            // ⚠ setTargetResolution DID NOT WORK and did not complain. Asking it for
            // 1280x720 got a 2736x2736 SQUARE frame -- 7.5 megapixels, 8.1x what was
            // requested -- and every stage paid: 35 ms of YUV conversion, 42 ms of swap
            // (detprep scales with frame area) and 56 ms of display, for 6.9 fps against
            // 26.6 on a 720p file. It is deprecated in camera-core 1.3 and interacts badly
            // with output rotation, which is presumably why it was ignored rather than
            // honoured or refused.
            //
            // ResolutionSelector states the same intent in the API that is actually
            // consulted: nearest supported size to 720p, preferring lower, 16:9.
            val resolution = androidx.camera.core.resolutionselector.ResolutionSelector.Builder()
                .setAspectRatioStrategy(
                    androidx.camera.core.resolutionselector.AspectRatioStrategy
                        .RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(
                    androidx.camera.core.resolutionselector.ResolutionStrategy(
                        android.util.Size(1280, 720),
                        androidx.camera.core.resolutionselector.ResolutionStrategy
                            .FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                .build()
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolution)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .setOutputImageRotationEnabled(true)
                .build()
            analysis.setAnalyzer(e) { img -> onImage(img, onShot) }
            runCatching {
                p.unbindAll()
                p.bindToLifecycle(owner, CameraSelector.DEFAULT_FRONT_CAMERA, analysis)
                // What was actually GRANTED, logged at bind rather than inferred from the
                // first frame -- the gap between asked and granted is the whole story here.
                android.util.Log.i("fflive", "granted ${analysis.resolutionInfo?.resolution}")
            }.onFailure {
                onShot(Shot(null, 0, 0.0, "camera: ${it.javaClass.simpleName}"))
                running = false
            }
        }, androidx.core.content.ContextCompat.getMainExecutor(ctx))
    }

    /**
     * Stop the pump, and DO NOT RETURN until no frame is still inside [onImage].
     *
     * ⚠ This is why the caller may free the pipeline afterwards. `shutdown()` alone only
     * refuses NEW work -- it does not wait for the task already running -- so the first
     * version returned while a frame was mid-`processFrame`, MainActivity called
     * NativePipe.release() underneath it, and the analyzer thread dereferenced a pipeline
     * that no longer existed:
     *
     *     SIGSEGV, null pointer dereference, fault addr 0x88
     *     #00 ffpipe::Pipeline::enhance
     *     #01 Java_..._processFrame
     *     #03 LiveEngine.onImage
     *
     * A frame takes ~60 ms, so the wait is imperceptible; 2 s is a bound, not an
     * expectation. Blocking the caller here is the entire point -- "stopped" has to mean
     * the native side is idle, not that it was asked to be.
     */
    fun stop() {
        running = false
        // Unbind FIRST so no further frames are dispatched, then drain what is in flight.
        runCatching { provider?.unbindAll() }
        provider = null
        exec?.let { e ->
            e.shutdown()
            runCatching { e.awaitTermination(2, java.util.concurrent.TimeUnit.SECONDS) }
        }
        exec = null
        bufs[0] = null; bufs[1] = null
        fps = 0.0
    }

    /**
     * One camera frame, all the way through, on the analyzer thread.
     *
     * ⚠ `close()` in a finally: CameraX hands out a small fixed pool of buffers and a single
     * leaked ImageProxy stalls the stream permanently with no error -- the preview simply
     * stops updating, which reads as "the swap hung".
     */
    private fun onImage(img: ImageProxy, onShot: (Shot) -> Unit) {
        try {
            if (!running) return
            val w = img.width
            val h = img.height
            val p = img.planes

            // DOWNSCALE FOR DISPLAY. The swap still runs at full frame resolution; only
            // what is DRAWN shrinks, and the pane is under 1100 px wide on this phone.
            // Converting at full sensor resolution and letting the GPU shrink it afterwards
            // was pure waste: at 2736x2736 the pixel buffer alone is 30 MB per frame.
            val scale = maxOf(1, (maxOf(w, h) + kMaxPreview - 1) / kMaxPreview)
            val dw = w / scale
            val dh = h / scale
            bufIx = bufIx xor 1
            var bmp = bufs[bufIx]
            if (bmp == null || bmp.width != dw || bmp.height != dh) {
                bmp = Bitmap.createBitmap(dw, dh, Bitmap.Config.ARGB_8888)
                bufs[bufIx] = bmp
            }

            // ONE call: planes in, swapped preview written into bmp's own pixels. The four
            // it replaced spent 23 of Live's 62 ms/frame moving bytes across JNI -- see
            // liveFrame in ffjni.cpp for what each of them was copying.
            val t = System.nanoTime()
            val faces = NativePipe.liveFrame(
                p[0].buffer, p[0].rowStride,
                p[1].buffer, p[1].rowStride, p[1].pixelStride,
                p[2].buffer, p[2].rowStride, p[2].pixelStride,
                w, h, bmp, dw, dh,
            )
            msPump += (System.nanoTime() - t) / 1e6
            if (faces < 0) {
                onShot(Shot(null, 0, fps, NativePipe.lastError()))
                return
            }

            if (++nStat == 30) {
                // One bucket, because there is one call -- but one number cannot say
                // whether a slow window is the swap or the YUV/display work wrapped around
                // it, and those differ by ~30 ms: a face in frame took this pump from 25 to
                // 62 ms and the single figure read as a regression either way.
                //
                // The native stage timers already carry that split and the app never asked
                // for them, so pump MINUS the stage sum is what liveFrame's own two scalar
                // loops cost. Reset per window, so each line is its own 30 frames.
                // `faces` is this frame's count, not the window's -- an indicator of which
                // regime the window was in, not a measurement.
                android.util.Log.i("fflive", "%dx%d -> %dx%d  pump %.1f ms/frame  faces %d"
                    .format(w, h, dw, dh, msPump / 30, faces))
                NativePipe.stageMillis().takeIf { it.isNotEmpty() }
                    ?.let { android.util.Log.i("fflive", "  $it") }
                NativePipe.resetStats()
                nStat = 0; msPump = 0.0
            }

            ++windowFrames
            val now = System.nanoTime()
            val elapsed = (now - windowStart) / 1e9
            if (elapsed >= 0.5) {
                fps = windowFrames / elapsed
                windowStart = now; windowFrames = 0
            }
            onShot(Shot(bmp, faces, fps, null))
        } catch (t: Throwable) {
            // A throw on the analyzer thread would otherwise take the stream down silently.
            onShot(Shot(null, 0, fps, "${t.javaClass.simpleName}: ${t.message ?: ""}"))
        } finally {
            img.close()
        }
    }
}
