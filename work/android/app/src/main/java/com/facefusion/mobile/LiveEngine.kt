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
    private var msYuv = 0.0
    private var msSwap = 0.0
    private var msOut = 0.0

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
            val analysis = ImageAnalysis.Builder()
                // 720p to match what the swap was measured at. CameraX treats this as a
                // request, not a promise, and picks the nearest supported size.
                .setTargetResolution(android.util.Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .setOutputImageRotationEnabled(true)
                .build()
            analysis.setAnalyzer(e) { img -> onImage(img, onShot) }
            runCatching {
                p.unbindAll()
                p.bindToLifecycle(owner, CameraSelector.DEFAULT_FRONT_CAMERA, analysis)
            }.onFailure {
                onShot(Shot(null, 0, 0.0, "camera: ${it.javaClass.simpleName}"))
                running = false
            }
        }, androidx.core.content.ContextCompat.getMainExecutor(ctx))
    }

    fun stop() {
        running = false
        runCatching { provider?.unbindAll() }
        provider = null
        exec?.shutdown(); exec = null
        bufs[0] = null; bufs[1] = null
        fps = 0.0
    }

    // One reusable array per plane. CameraX rotates in place into a fixed buffer pool, so
    // the sizes are stable after the first frame and this allocates exactly three times.
    private val planes = arrayOfNulls<ByteArray>(3)

    private fun plane(i: Int, b: java.nio.ByteBuffer): ByteArray {
        val n = b.remaining()
        var a = planes[i]
        if (a == null || a.size != n) { a = ByteArray(n); planes[i] = a }
        b.get(a)
        return a
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
            // The same converter VideoSwapper feeds from MediaCodec: native, stride-aware,
            // and already carrying the 720p cost this feature is budgeted against. It takes
            // ByteArrays, so the planes are copied out -- the buffers are reused, and the
            // arrays are reused too rather than allocating ~1.4 MB per frame here.
            var t = System.nanoTime()
            val bgr = NativePipe.yuvToBgr(
                plane(0, p[0].buffer), p[0].rowStride,
                plane(1, p[1].buffer), p[1].rowStride, p[1].pixelStride,
                plane(2, p[2].buffer), p[2].rowStride, p[2].pixelStride,
                w, h,
            )
            msYuv += (System.nanoTime() - t) / 1e6; t = System.nanoTime()

            val faces = NativePipe.processFrame(bgr, w, h)
            msSwap += (System.nanoTime() - t) / 1e6; t = System.nanoTime()
            if (faces < 0) {
                onShot(Shot(null, 0, fps, NativePipe.lastError()))
                return
            }

            bufIx = bufIx xor 1
            var bmp = bufs[bufIx]
            if (bmp == null || bmp.width != w || bmp.height != h) {
                bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                bufs[bufIx] = bmp
            }
            bmp.setPixels(NativePipe.bgrToArgb(bgr, w, h, w, h), 0, w, 0, 0, w, h)
            msOut += (System.nanoTime() - t) / 1e6

            if (++nStat == 30) {
                android.util.Log.i("fflive", "%dx%d  yuv %.1f  swap %.1f  out %.1f  ms/frame"
                    .format(w, h, msYuv / 30, msSwap / 30, msOut / 30))
                nStat = 0; msYuv = 0.0; msSwap = 0.0; msOut = 0.0
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
