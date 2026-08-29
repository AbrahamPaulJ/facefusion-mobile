package com.facefusion.mobile

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.lifecycle.lifecycleScope
import com.facefusion.mobile.ui.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.roundToInt

/**
 * Source image + target video -> swapped MP4, entirely on device.
 *
 * The models are NOT bundled: they are ~266 MB and three of the five are non-commercial or
 * GPL-3.0 (see docs/model-audit.md), so the APK stays free of them and they are pushed to
 * the app's files dir -- see work/device/install_app.ps1.  Note that Kotlin nests block
 * comments, so a literal glob of the form slash-star cannot appear in one.
 *
 * The QNN runtime .so files ride in jniLibs and are dlopen'd from nativeLibraryDir.
 *
 * This class holds STATE and LOGIC only; every composable lives in `ui/`.
 */
class MainActivity : ComponentActivity() {

    private var sourceUri by mutableStateOf<Uri?>(null)
    private var sourceThumb by mutableStateOf<Bitmap?>(null)
    private var targetFile by mutableStateOf<File?>(null)
    private var targetName by mutableStateOf<String?>(null)
    private var durationMs by mutableStateOf(0L)
    private var trimStartMs by mutableStateOf(0f)
    private var trimEndMs by mutableStateOf(0f)

    /**
     * The runtime knobs, restored from the last run.  Held here rather than inside the
     * composable so they survive a recomposition and so [runSwap] reads the same object
     * the UI is editing -- [NativePipe.init] takes them once, at load, and the pipeline
     * keeps its own copy for the duration.
     */
    private var opts by mutableStateOf(SwapOptions())
    private var openCard by mutableStateOf("")

    private var status by mutableStateOf("")
    private var log by mutableStateOf("")
    private var busy by mutableStateOf(false)
    private var preparing by mutableStateOf(false)
    private var progress by mutableStateOf(0f)
    private var framesDone by mutableStateOf(0)
    private var framesTotal by mutableStateOf(0)
    private var elapsedS by mutableStateOf(0.0)
    private var preview by mutableStateOf<Bitmap?>(null)
    private var outputFile by mutableStateOf<File?>(null)

    /** The finished still, when the target was an image rather than a video. */
    private var outputImage by mutableStateOf<Bitmap?>(null)

    /**
     * Whether the finished output stopped early because the user cancelled.
     *
     * A cancelled run now KEEPS its frames (VideoSwapper), which is what was asked for --
     * but a 4-second file from a 30-second clip must not read as a completed run, so the
     * pane says so.
     */
    private var outputPartial by mutableStateOf(false)
    private var savedUri by mutableStateOf<Uri?>(null)

    // ---- UI shell
    private var screen by mutableStateOf(Screen.Swap)
    private var advancedOpen by mutableStateOf(false)
    private var modelsVersion by mutableStateOf(0)
    private var deviceUi by mutableStateOf(DeviceUi())
    private var confirmMetered by mutableStateOf(false)

    /**
     * Whether this tier's set is incomplete, as explicit state refreshed from disk.
     *
     * Not a function evaluated during composition. The first version of this interpolated
     * the filename wrongly and reported everything missing forever; making the check a
     * value with one writer means it can be logged, and it is refreshed on resume and while
     * a download runs rather than depending on a cross-thread invalidation.
     */
    private var modelsMissing by mutableStateOf(true)

    /** Asked once, on the first download. Denied only costs the progress notification. */
    private val askNotify = registerForActivityResult(
        ActivityResultContracts.RequestPermission()) { }

    // ---- the two preview panes
    private val previews = PreviewEngine()
    private var originalFrame by mutableStateOf<Bitmap?>(null)
    private var swappedFrame by mutableStateOf<Bitmap?>(null)
    private var previewWarm by mutableStateOf(false)
    private var previewBusy by mutableStateOf(false)
    private var previewNote by mutableStateOf<String?>(null)
    private var targetAspect by mutableStateOf(16f / 9f)

    /**
     * The target when it is a STILL rather than a video.
     *
     * Non-null is the mode flag: `durationMs == 0` follows from it, which is what hides the
     * trim slider and the frame-rate control.
     */
    private var targetImage by mutableStateOf<Bitmap?>(null)

    /** The target video's own frame rate, so the rate control can cap itself to it. */
    private var inputFps by mutableStateOf(30)

    /**
     * The trim handle the previews are following.
     *
     * Both panes used to show the start frame unconditionally, so dragging the END handle
     * changed nothing on screen and read as a broken preview. Whichever handle moved last
     * is the one being looked at.
     */
    private var previewEdge by mutableStateOf(TrimEdge.Start)

    /** The position both panes are previewing: whichever handle was touched last. */
    private val previewAtMs: Float
        get() = if (previewEdge == TrimEdge.End) trimEndMs else trimStartMs
    private var scrubJob: Job? = null

    /** Set by the Cancel button, read by the swap worker. Volatile: different threads. */
    @Volatile private var cancelRequested = false

    private val pickSource = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            sourceUri = uri
            sourceThumb = decodeOriented(uri)
            status = "Source set"
            // A different face means the loaded pipeline is holding the wrong embedding.
            invalidatePreview()
        }
    }
    // OpenDocument rather than GetContent: GetContent takes ONE mime filter, and the
    // target can now be a video or a still.
    private val pickTarget = registerForActivityResult(
        ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) loadTarget(uri)
    }

    /**
     * The models live in a subdirectory of the app's external files dir.
     *
     * It has to be created by the APP, not by `adb push`: a directory created by adb is
     * owned by `shell`, and although the files inside are world-readable the app cannot
     * traverse a shell-owned directory here -- open() fails with nothing but ENOENT to
     * explain it.  mkdirs() on every launch makes the dir app-owned before anything is
     * pushed into it.
     */
    private fun modelDir() = File(getExternalFilesDir(null), "models").apply { mkdirs() }

    /**
     * Decode an image the way the camera meant it to be seen.
     *
     * [BitmapFactory] ignores EXIF orientation. A phone stores a portrait photo as
     * LANDSCAPE pixels plus an Orientation tag, so decoding without applying the tag
     * yields a sideways image -- and that is not a cosmetic thumbnail problem: `yoloface`
     * is not rotation invariant, so a face on its side is simply not detected and the run
     * dies with "no face found in source image". Reported 2026-08-29; the rotated preview
     * and the missing face were one bug, not two.
     *
     * [ImageDecoder] applies the tag itself and also reads HEIC, which is what Samsung
     * cameras write by default and what BitmapFactory is weakest on.
     *
     * ALLOCATOR_SOFTWARE is required, not a preference: every caller reads the pixels back
     * with getPixels, and a hardware bitmap has no pixel array to read. It throws rather
     * than returning null on a bad file, hence runCatching.
     */
    private fun decodeOriented(uri: Uri): Bitmap? = runCatching {
        ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)) { d, _, _ ->
            d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            d.isMutableRequired = true
        }
    }.getOrNull()

    /** As above, for a file the selftest pushed rather than a picked Uri. */
    private fun decodeOriented(file: File): Bitmap? = runCatching {
        ImageDecoder.decodeBitmap(ImageDecoder.createSource(file)) { d, _, _ ->
            d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            d.isMutableRequired = true
        }
    }.getOrNull()

    /**
     * The fp16 canary pair, unpacked out of assets.
     *
     * Rewritten on every call rather than cached: the two total 85 KB, and a half-written
     * file from a killed install would otherwise fail the control forever and pin the
     * verdict at "unknown".
     */
    private fun canaryDir() = File(filesDir, "canary").apply {
        mkdirs()
        for (name in listOf("canary_249.bin", "canary_228.bin")) {
            runCatching {
                assets.open("canary/$name").use { input ->
                    File(this, name).outputStream().use { input.copyTo(it) }
                }
            }
        }
    }

    /**
     * Which `<name>_<tier>.bin` this chip needs.  Measured once per process; the native
     * side re-derives it inside Pipeline::init, so this copy exists only so the UI can
     * name the right files when they are missing.
     */
    private val tier: String by lazy {
        // Resolved against disk, exactly as Pipeline::init resolves it. The tier the UI
        // names has to be the tier the pipeline will open, or "models missing" and "model
        // loaded" contradict each other on any chip whose best tier is not hosted yet.
        // With nothing on disk, name the best one -- that is the set to download.
        tierChain.firstOrNull { File(modelDir(), "yoloface_" + it + ".bin").canRead() }
            ?: tierChain.firstOrNull()
            ?: NativePipe.probeTier(applicationInfo.nativeLibraryDir,
                                    applicationInfo.nativeLibraryDir)
    }

    /**
     * Every tier this chip can load, best first. See [NativePipe.probeTierChain] -- it is
     * NOT "this tier and every older one", so it must not be reconstructed here.
     */
    private val tierChain: List<String> by lazy {
        NativePipe.probeTierChain(applicationInfo.nativeLibraryDir,
                                  applicationInfo.nativeLibraryDir)
            .split(",").map { it.trim() }.filter { it.isNotEmpty() }
    }

    /**
     * Whether the second swapper is on the device at all.
     *
     * `inswapper_128` is converted and one flag away, but it is another 136 MB that
     * install_app.ps1 only pushes when asked. Offering a model the app cannot load would
     * turn a missing file into a failed run, so the choice appears only when it is real.
     */
    private val hasInswapper: Boolean by lazy {
        File(modelDir(), "inswapper_$tier.bin").canRead()
    }

    /**
     * Whether the face enhancer's binary is on the device.
     *
     * Asked of the FILESYSTEM, not [NativePipe.hasEnhancer], because the switch has to be
     * drawable before any pipeline exists -- the Advanced panel opens long before a run
     * initialises one. The native side is still the authority at execution time and skips
     * the stage if the model went away in between.
     *
     * Not `by lazy`: a download can add gpen after the Activity is created, and a lazy
     * would keep saying no for the life of the process.
     */
    private val hasEnhancer: Boolean
        get() = File(modelDir(), "gpen_$tier.bin").canRead()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        BugReport.install(this)
        NativePipe.ensureLoaded()
        modelDir()
        opts = SwapOptions.load(this)

        if (intent?.getStringExtra("selftest") != null) { selfTest(); return }

        probeDevice()

        refreshModelsMissing()

        setContent {
            FaceFusionTheme {
                // The download runs in a service, on its own thread. Rather than trust a
                // cross-thread state write to invalidate exactly the right scope, re-read
                // the disk while it matters; the loop exits as soon as the set is complete.
                LaunchedEffect(modelsMissing, ModelDownload.running) {
                    while (modelsMissing || ModelDownload.running) {
                        refreshModelsMissing()
                        delay(400)
                    }
                }

                // The preview warms ITSELF once both inputs and the models exist.
                //
                // The refresh button used to be the only thing that would pay for the first
                // model load (~266 MB, several seconds), so removing it means nothing would
                // ever ask. force = true because that first load is exactly what a cold
                // refreshSwapped declines to do on its own.
                //
                // Keyed on the inputs rather than run in a loop: it fires when the user
                // finishes picking, and `previewWarm` stops it repeating.
                LaunchedEffect(sourceUri, targetFile, targetImage, modelsMissing) {
                    if (sourceUri != null && (targetFile != null || targetImage != null) &&
                        !modelsMissing && !previewWarm && !busy) {
                        refreshSwapped(force = true)
                    }
                }
                AppScaffold(screen, { screen = it }) { pad ->
                    Box(Modifier.padding(pad)) {
                        when (screen) {
                            Screen.Swap -> SwapScreen(
                                sourceThumb = sourceThumb,
                                hasSource = sourceUri != null,
                                hasTarget = targetFile != null || targetImage != null,
                                durationMs = durationMs,
                                trimStartMs = trimStartMs,
                                trimEndMs = trimEndMs,
                                onTrimChange = ::onTrimChanged,
                                targetAspect = targetAspect,
                                inputFps = inputFps,
                                fmt = ::fmt,
                                preview = PreviewUi(
                                    original = originalFrame,
                                    // During a run the pane becomes the live output, which is
                                    // the same thing one frame later.
                                    swapped = if (busy) preview else swappedFrame,
                                    timeLabel = if (durationMs > 0) fmt(previewAtMs) else "",
                                    warm = previewWarm,
                                    busy = previewBusy,
                                    note = previewNote,
                                ),
                                run = RunUi(busy, preparing, progress, framesDone,
                                            framesTotal, elapsedS),
                                status = status,
                                log = log,
                                opts = opts,
                                onOptsChange = { o ->
                                    opts = o
                                    o.save(this@MainActivity)
                                    // init() consumed the old options; the warm pipeline is
                                    // now showing something the user did not ask for.
                                    invalidatePreview()
                                },
                                hasInswapper = hasInswapper,
                                hasEnhancer = hasEnhancer,
                                openCard = openCard,
                                onToggleCard = { k -> openCard = if (openCard == k) "" else k },
                                advancedOpen = advancedOpen,
                                onToggleAdvanced = { advancedOpen = !advancedOpen },
                                hasOutput = outputFile != null || outputImage != null,
                                outputFile = outputFile,
                                outputImage = outputImage,
                                outputPartial = outputPartial,
                                onSaveFrame = ::saveFrameAt,
                                saved = savedUri != null,
                                savedPath = savedUri?.let { "Movies/FaceFusion/${outputFile?.name}" },
                                onPickSource = { pickSource.launch("image/*") },
                                onPickTarget = {
                                    pickTarget.launch(arrayOf("video/*", "image/*"))
                                },
                                onClearTarget = ::clearTarget,
                                onSwap = { runSwap() },
                                onCancel = { cancelRequested = true; status = "Cancelling..." },
                                modelsMissing = modelsMissing,
                                onDownload = { onDownloadTapped() },
                                onShareLog = { shareBugReport() },
                                onSave = { outputFile?.let { saveToGallery(it) } },
                                onShare = { outputFile?.let { shareVideo(it) } },
                            )

                            Screen.Settings -> SettingsScreen(
                                models = modelRows(),
                                modelDirPath = modelDir().absolutePath,
                                device = deviceUi,
                                onDeleteModel = { m ->
                                    File(modelDir(), m.fileName).delete()
                                    modelsVersion++
                                    invalidatePreview()
                                },
                            )
                        }
                    }

                    if (confirmMetered) {
                        AlertDialog(
                            onDismissRequest = { confirmMetered = false },
                            title = { Text("Download on mobile data?") },
                            text = {
                                Text(
                                    "The models are about 275 MB. You appear to be on a " +
                                        "metered connection, so this may use your data " +
                                        "allowance."
                                )
                            },
                            confirmButton = {
                                TextButton({ confirmMetered = false; beginDownload() }) {
                                    Text("Download anyway")
                                }
                            },
                            dismissButton = {
                                TextButton({ confirmMetered = false }) { Text("Wait for Wi-Fi") }
                            },
                        )
                    }
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        refreshModelsMissing()
    }

    override fun onDestroy() {
        super.onDestroy()
        scrubJob?.cancel()
        previews.release()
    }

    private fun fmt(ms: Float): String {
        val t = (ms / 1000f)
        return "%d:%04.1f".format((t / 60).toInt(), t % 60)
    }

    private fun appendLog(line: String) { log = (log + line + "\n").takeLast(4000) }

    // ------------------------------------------------------------------ device

    /** Measure the HTP once, off the main thread; the probe brings the backend up. */
    private fun probeDevice() = lifecycleScope.launch(Dispatchers.Default) {
        val lib = applicationInfo.nativeLibraryDir
        val raw = runCatching { NativePipe.probeDeviceInfo(lib, lib) }.getOrDefault("ok=0")
        val kv = raw.split(';').mapNotNull {
            val p = it.split('='); if (p.size == 2) p[0] to p[1] else null
        }.toMap()
        val fp16 = runCatching {
            NativePipe.probeFp16(lib, lib, canaryDir().absolutePath)
        }.getOrDefault("unknown")
        withContext(Dispatchers.Main) {
            deviceUi = DeviceUi(
                ok = kv["ok"] == "1",
                arch = kv["arch"]?.toIntOrNull() ?: 0,
                vtcmMb = kv["vtcm"]?.toIntOrNull() ?: 0,
                soc = kv["soc"]?.toIntOrNull() ?: 0,
                tier = kv["tier"].orEmpty(),
                fp16 = fp16,
            )
        }
    }

    /** Everything we know about this run, handed to whatever the user sends it with. */
    private fun shareBugReport() {
        val d = deviceUi
        val npu = if (d.ok)
            "arch v" + d.arch + ", vtcm " + d.vtcmMb + " MB, soc " + d.soc +
                ", tier " + d.tier + ", fp16 " + d.fp16
        else "not measured"
        val models = modelDir().listFiles()?.map { it.name + "  " + it.length() + " bytes" }
            ?.sorted().orEmpty()
        BugReport.share(this, BugReport.compose(this, log, npu, models, status))
        BugReport.clearCrash(this)
    }

    // --------------------------------------------------------------- model download

    /**
     * Whether this tier's set is incomplete.
     *
     * Local and instant -- it asks the filesystem, not the network -- so the overlay can
     * decide whether to appear without a round trip. Reads [modelsVersion] and
     * [ModelDownload.finished] so a delete or a completed download recomposes it.
     */
    /** Recompute from disk. Cheap: a handful of canRead() calls. */
    private fun refreshModelsMissing() {
        val t = deviceUi.tier.ifEmpty { tier }
        val dir = modelDir()
        val absent = listOf("yoloface", "fan2d", "arcface", opts.swapper)
            .filter { !File(dir, it + "_" + t + ".bin").canRead() }
            .toMutableList()
        if (!File(dir, "nsfw_" + t + ".bin").canRead() &&
            !File(dir, "nsfwq_" + t + ".bin").canRead())
            absent += "nsfw"
        val now = absent.isNotEmpty()
        if (now != modelsMissing)
            android.util.Log.i("ffmodels", "missing=" + now + " " + absent)
        modelsMissing = now
    }

    private fun onDownloadTapped() {
        if (android.os.Build.VERSION.SDK_INT >= 33)
            askNotify.launch(android.Manifest.permission.POST_NOTIFICATIONS)
        // Metered is a warning, not a refusal: the user may have no Wi-Fi and still want it.
        if (ModelDownload.isMetered(this)) confirmMetered = true else beginDownload()
    }

    private fun beginDownload() {
        ModelDownload.reset()
        // The whole chain, not one tier: the downloader picks the best tier the manifest
        // actually publishes. Handing it only `tier` would fail outright on a chip whose
        // best tier is not hosted yet.
        DownloadService.start(this, tierChain.joinToString(","))
    }

    // ------------------------------------------------------------------ models

    /** What `loadTarget` pulls out of the container, so the result is not a List<Any>. */
    private data class Loaded(val file: File, val durationMs: Long, val width: Int,
                             val height: Int, val fps: Int)

    /**
     * The inventory the Settings screen lists.
     *
     * Called during composition, so reading [modelsVersion] here is what makes a delete
     * redraw the list -- the files themselves are not observable state.
     */
    private fun modelRows(): List<ModelRow> {
        val ignored = modelsVersion
        check(ignored >= 0)
        val t = deviceUi.tier.ifEmpty { tier }
        val required = listOf(
            "yoloface" to "Face detector",
            "fan2d" to "Face landmarker",
            "arcface" to "Face recogniser",
            "hyperswap" to "Face swapper",
        )
        val optional = listOf(
            "inswapper" to "Face swapper (alternative)",
            "fan685" to "Landmark refiner",
            "nsfw" to "Content checker",
            "nsfwq" to "Content checker (quantised)",
        )
        fun row(name: String, label: String, req: Boolean): ModelRow {
            val f = File(modelDir(), "${name}_$t.bin")
            return ModelRow(label, f.name, if (f.exists()) f.length() else 0L, f.canRead(), req)
        }
        return required.map { (n, l) -> row(n, l, true) } +
            optional.map { (n, l) -> row(n, l, false) }.filter { it.present }
    }

    // ------------------------------------------------------------------ previews

    /** The warm pipeline no longer matches what the UI is asking for. */
    private fun invalidatePreview() {
        previews.invalidate()
        previewWarm = false
        swappedFrame = null
        previewNote = null
    }

    /**
     * The trim handle moved.
     *
     * Debounced rather than acted on per pixel: a drag emits dozens of values a second and
     * each one is a video seek. The swapped pane waits longer again, and only refreshes if
     * the pipeline is ALREADY warm -- the first load costs seconds and must be asked for.
     */
    private fun onTrimChanged(start: Float, end: Float, edge: TrimEdge) {
        trimStartMs = start
        trimEndMs = end
        previewEdge = edge
        scrubJob?.cancel()
        scrubJob = lifecycleScope.launch {
            delay(150)
            // The frame under the handle being dragged, not always the start.
            originalFrame = previews.frameAt(previewAtMs)
            if (previewWarm && !busy) {
                delay(250)
                refreshSwapped(force = false)
            }
        }
    }

    /**
     * Render the swapped pane for the current frame.
     *
     * [force] is the refresh button: it is allowed to pay for the model load. Without it,
     * this is a no-op unless the pipeline is already warm.
     */
    private fun refreshSwapped(force: Boolean) {
        if (busy || previewBusy) return
        if (!previewWarm && !force) return
        val src = sourceUri ?: return
        // An image target has no targetFile -- it is held as a bitmap. Testing the
        // video handle here meant an image never previewed at all.
        if (targetFile == null && targetImage == null) return

        lifecycleScope.launch {
            previewBusy = true
            previewNote = null
            try {
                val frame = originalFrame ?: previews.frameAt(previewAtMs)?.also {
                    originalFrame = it
                } ?: run {
                    previewNote = "Could not read that frame"
                    return@launch
                }

                if (!previewWarm) {
                    val bmp = withContext(Dispatchers.IO) { decodeOriented(src) }
                    if (bmp == null) { previewNote = "Could not read the source image"; return@launch }
                    val soft = bmp.copy(Bitmap.Config.ARGB_8888, false)
                    val px = IntArray(soft.width * soft.height)
                    soft.getPixels(px, 0, soft.width, 0, 0, soft.width, soft.height)

                    val models = modelDir()
                    val t = deviceUi.tier.ifEmpty { tier }
                    val missing = listOf("yoloface", "fan2d", "arcface", opts.swapper)
                        .map { File(models, "${it}_$t.bin") }.filterNot { it.canRead() }
                    if (missing.isNotEmpty()) {
                        previewNote = "Missing ${missing.joinToString { it.name }}"
                        return@launch
                    }

                    val lib = applicationInfo.nativeLibraryDir
                    val err = previews.ensureReady(
                        lib, models.absolutePath, opts, src,
                        NativePipe.argbToBgr(px, soft.width, soft.height), soft.width, soft.height,
                        gate = {
                            // The same check runSwap makes. Without it the preview is a
                            // complete second processing path with no gate on it, and the
                            // gate becomes avoidable by simply never pressing Swap.
                            val v = ContentGate.checkImage(bmp)
                            appendLog("preview source score %+.3f".format(v.score))
                            if (v.ok) null else ContentGate.message("the source image", v)
                        },
                    )
                    if (err != null) { previewNote = err; return@launch }
                    previewWarm = true
                }

                // Every previewed frame, not just the source. The source is checked once,
                // when the pipeline warms; the target is checked here because the trim
                // handle can reach any frame in the clip and this pane displays it.
                ContentGate.checkImage(frame).let { v ->
                    if (!v.ok) {
                        previewNote = ContentGate.message("this frame", v)
                        swappedFrame = null
                        return@launch
                    }
                }

                val out = previews.swap(frame)
                when {
                    out.error != null -> { previewNote = out.error; swappedFrame = null }
                    out.faces == 0 -> {
                        // processFrame leaves the buffer untouched when it finds nothing, so
                        // without this the pane would show the ORIGINAL and look like a
                        // swap that did nothing.
                        previewNote = "No face detected in this frame"
                        swappedFrame = null
                    }
                    else -> { swappedFrame = out.bitmap; previewNote = null }
                }
            } finally {
                previewBusy = false
            }
        }
    }

    /**
     * Copy the picked target out of SAF.
     *
     * A still short-circuits everything video: there is no duration, so no trim, no frame
     * rate, and nothing to seek. `durationMs == 0` is the mode flag the UI reads.
     */
    private fun loadTarget(uri: Uri) {
        if (contentResolver.getType(uri)?.startsWith("image/") == true) {
            loadTargetImage(uri)
            return
        }
        preparing = true
        targetName = uri.lastPathSegment?.substringAfterLast('/')
        lifecycleScope.launch {
            val ok = withContext(Dispatchers.IO) {
                runCatching {
                    // MediaExtractor needs a real path; SAF only gives a stream.
                    val f = File(cacheDir, "target.mp4")
                    contentResolver.openInputStream(uri).use { i ->
                        f.outputStream().use { o -> i!!.copyTo(o) }
                    }
                    val mmr = MediaMetadataRetriever().apply { setDataSource(f.absolutePath) }
                    val d = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                        ?.toLongOrNull() ?: 0L
                    val w = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)
                        ?.toIntOrNull() ?: 0
                    val h = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
                        ?.toIntOrNull() ?: 0
                    // CAPTURE_FRAMERATE is absent on plenty of files (it is a camera tag,
                    // not a container one), so fall back to counting frames over the
                    // duration rather than assuming 30.
                    val fpsMeta = mmr.extractMetadata(
                        MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)?.toFloatOrNull()
                    val frameCount = mmr.extractMetadata(
                        MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull()
                    val fps = when {
                        fpsMeta != null && fpsMeta > 1f -> fpsMeta.roundToInt()
                        frameCount != null && d > 0 -> (frameCount * 1000.0 / d).roundToInt()
                        else -> 30
                    }.coerceIn(1, 240)
                    mmr.release()
                    Loaded(f, d, w, h, fps)
                }
            }
            ok.onSuccess { l ->
                targetImage = null
                targetFile = l.file; durationMs = l.durationMs
                inputFps = l.fps
                trimStartMs = 0f; trimEndMs = l.durationMs.toFloat()
                targetAspect = if (l.width > 0 && l.height > 0)
                    l.width.toFloat() / l.height else 16f / 9f
                status = "Target ready: ${l.width} x ${l.height}, ${fmt(l.durationMs.toFloat())}"
                // Open it for scrubbing and show the first frame straight away.
                previews.openTarget(l.file.absolutePath)
                swappedFrame = null; previewNote = null
                originalFrame = previews.frameAt(0f)
            }.onFailure { status = "Cannot read video: ${it.message}" }
            preparing = false
        }
    }

    /**
     * A still target.
     *
     * Decoded through [decodeOriented] like the source, so a portrait photo is not swapped
     * on its side -- the same EXIF bug, and it would have been reintroduced here by using
     * BitmapFactory for the new path.
     */
    private fun loadTargetImage(uri: Uri) {
        preparing = true
        targetName = uri.lastPathSegment?.substringAfterLast('/')
        lifecycleScope.launch {
            val bmp = withContext(Dispatchers.IO) { decodeOriented(uri) }
            if (bmp == null) {
                status = "Cannot read image"
                preparing = false
                return@launch
            }
            previews.closeTarget()
            targetFile = null
            targetImage = bmp
            durationMs = 0L
            trimStartMs = 0f; trimEndMs = 0f
            targetAspect = if (bmp.height > 0) bmp.width.toFloat() / bmp.height else 1f
            originalFrame = bmp
            swappedFrame = null; previewNote = null
            invalidatePreview()
            originalFrame = bmp
            status = "Target ready: ${bmp.width} x ${bmp.height}"
            preparing = false
        }
    }

    /** Drop the target and everything derived from it. */
    private fun clearTarget() {
        previews.closeTarget()
        targetFile = null
        targetImage = null
        targetName = null
        durationMs = 0L
        trimStartMs = 0f; trimEndMs = 0f
        targetAspect = 16f / 9f
        originalFrame = null
        outputFile = null; outputImage = null; outputPartial = false; savedUri = null
        invalidatePreview()
        status = ""
    }

    private fun runSwap() {
        val src = sourceUri ?: return
        val tgtImage = targetImage
        val tgt = targetFile
        if (tgt == null && tgtImage == null) return
        // The preview holds a loaded pipeline and `g_pipe` is a single global, so the run
        // cannot start until it lets go.
        invalidatePreview()
        cancelRequested = false
        busy = true; progress = 0f; log = ""; outputFile = null; savedUri = null
        outputImage = null; outputPartial = false
        preview = null; framesDone = 0; framesTotal = 0; elapsedS = 0.0

        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) {
                runCatching {
                    val models = modelDir()
                    val missing = listOf("yoloface", "fan2d", "arcface", opts.swapper)
                        .map { File(models, "${it}_$tier.bin") }.filterNot { it.canRead() }
                        .toMutableList()
                    // The gate is mandatory because it blocks. Either build satisfies it:
                    // fp32 (`nsfw_`) is the shipping one and only finalizes on v79, so the
                    // lower tiers carry the quantised `nsfwq_` instead.
                    if (!File(models, "nsfw_$tier.bin").canRead() &&
                        !File(models, "nsfwq_$tier.bin").canRead())
                        missing += File(models, "nsfw_$tier.bin")
                    if (missing.isNotEmpty())
                        error("cannot read ${missing.joinToString { it.name }} in " +
                              "${models.absolutePath} -- run work/device/install_app.ps1")

                    status = "Loading models..."
                    val libDir = applicationInfo.nativeLibraryDir
                    if (!NativePipe.init(libDir, libDir, models.absolutePath, opts))
                        error("init: ${NativePipe.lastError()}")
                    appendLog("weight %.2f  blur %.2f  padding %s  boost %s%s"
                        .format(opts.weight, opts.maskBlur,
                                opts.maskPadding.joinToString("/"), opts.pixelBoostLabel,
                                if (opts.largestOnly) "  largest face only" else ""))

                    status = "Reading source face..."
                    val bmp = decodeOriented(src) ?: error("cannot decode source image")

                    // The content gate, BEFORE anything is processed or previewed. It
                    // blocks, as upstream does, so a refusal ends the run here -- there is
                    // no partial output and nothing reaches the preview surface.
                    status = "content check..."
                    if (NativePipe.contentGateIsQuantised())
                        appendLog("content gate: W8A16 build, biased " +
                                  "+${ContentGate.QUANTISED_BIAS} toward refusing")
                    ContentGate.checkImage(bmp).let {
                        appendLog("source content score %+.3f".format(it.score))
                        if (!it.ok) throw ContentGate.Refused(ContentGate.message("the source image", it))
                    }
                    // The target, gated whichever kind it is. An image target is a
                    // FOURTH processing path, and the gate has to be on it by construction
                    // -- a separate runImageSwap is exactly how a path ends up ungated.
                    if (tgtImage != null) {
                        ContentGate.checkImage(tgtImage).let {
                            appendLog("target content score %+.3f".format(it.score))
                            if (!it.ok)
                                throw ContentGate.Refused(
                                    ContentGate.message("the target image", it))
                        }
                    } else {
                        ContentGate.checkVideo(tgt!!).let {
                            // `detail` is an ARGUMENT, never interpolated into the format
                            // string: it reads "0/11 flagged (0.0%)", and that trailing `%)`
                            // is parsed as a conversion -- UnknownFormatConversionException,
                            // which killed the whole swap after the gate had already passed.
                            appendLog("target content: %s, worst %+.3f".format(it.detail, it.score))
                            if (!it.ok)
                                throw ContentGate.Refused(
                                    ContentGate.message("the target video", it))
                        }
                    }

                    val soft = bmp.copy(Bitmap.Config.ARGB_8888, false)
                    val px = IntArray(soft.width * soft.height)
                    soft.getPixels(px, 0, soft.width, 0, 0, soft.width, soft.height)
                    if (!NativePipe.setSource(NativePipe.argbToBgr(px, soft.width, soft.height),
                                              soft.width, soft.height))
                        error("source: ${NativePipe.lastError()}")
                    appendLog("source ready (${soft.width}x${soft.height})")

                    val t0 = System.currentTimeMillis()

                    // ---- a still target: one frame, no codecs, no muxer.
                    if (tgtImage != null) {
                        status = "Swapping..."
                        val ts = tgtImage.copy(Bitmap.Config.ARGB_8888, false)
                        val tpx = IntArray(ts.width * ts.height)
                        ts.getPixels(tpx, 0, ts.width, 0, 0, ts.width, ts.height)
                        val bgr = NativePipe.argbToBgr(tpx, ts.width, ts.height)
                        val faces = NativePipe.processFrame(bgr, ts.width, ts.height)
                        if (faces < 0) error("native: ${NativePipe.lastError()}")
                        appendLog("$faces face(s) swapped")
                        val argb = NativePipe.bgrToArgb(bgr, ts.width, ts.height,
                                                        ts.width, ts.height)
                        val bmpOut = Bitmap.createBitmap(argb, ts.width, ts.height,
                                                         Bitmap.Config.ARGB_8888)
                        // Written to a file as well as held in memory, so Save and Share
                        // work exactly as they do for a video.
                        val png = File(getExternalFilesDir(null),
                            "swapped_${System.currentTimeMillis()}.png")
                        png.outputStream().use {
                            bmpOut.compress(Bitmap.CompressFormat.PNG, 100, it)
                        }
                        withContext(Dispatchers.Main) { outputImage = bmpOut }
                        appendLog("total %.1f s".format(
                            (System.currentTimeMillis() - t0) / 1000.0))
                        return@runCatching png
                    }

                    val out = File(getExternalFilesDir(null),
                        "swapped_${System.currentTimeMillis()}.mp4")
                    status = "Swapping..."
                    var lastPreview = 0L

                    VideoSwapper(
                        outputFps = opts.outputFps,
                        trimStartUs = (trimStartMs * 1000).toLong(),
                        trimEndUs = if (trimEndMs >= durationMs) Long.MAX_VALUE
                                    else (trimEndMs * 1000).toLong(),
                        onProgress = { done, total ->
                            framesDone = done; framesTotal = total
                            progress = if (total > 0) done.toFloat() / total else 0f
                            elapsedS = (System.currentTimeMillis() - t0) / 1000.0
                        },
                        onFrame = { bgr, w, h ->
                            // throttle: a Bitmap per frame is pure allocation churn and the
                            // eye cannot use more than a few updates a second anyway
                            val now = System.currentTimeMillis()
                            if (now - lastPreview > 250) {
                                lastPreview = now
                                val pw = 480
                                val ph = (h.toLong() * pw / w).toInt().coerceAtLeast(1)
                                val argb = NativePipe.bgrToArgb(bgr, w, h, pw, ph)
                                preview = Bitmap.createBitmap(argb, pw, ph, Bitmap.Config.ARGB_8888)
                            }
                        },
                        onLog = { appendLog(it) },
                        isCancelled = { cancelRequested },
                    ).swap(tgt!!.absolutePath, out.absolutePath).getOrThrow()

                    appendLog("total %.1f s".format((System.currentTimeMillis() - t0) / 1000.0))
                    out
                }
            }
            result.onSuccess {
                outputFile = it; progress = 1f
                // The run kept whatever it had when Cancel was pressed, so say which it is.
                outputPartial = cancelRequested
                status = "Done - ${it.length() / 1024} KB"
            }.onFailure {
                // A refusal is already a finished sentence aimed at the user, and it is not
                // a fault: prefixing it with "Failed:" and dumping a stack trace would
                // present a working safety check as a crash.
                if (it.message == "cancelled") {
                    // Asked for, not gone wrong: no "Failed:", no stack trace.
                    status = "Cancelled"
                } else if (it is ContentGate.Refused) {
                    status = it.message ?: "Explicit content not allowed"
                } else {
                    status = "Failed: ${it.message}"
                    appendLog(it.stackTraceToString().take(700))
                }
            }
            NativePipe.release()
            busy = false
        }
    }

    private fun saveToGallery(file: File) {
        lifecycleScope.launch {
            // An image result goes to Pictures, a video to Movies. Same button, and the
            // file itself says which -- the alternative is a second button that is dead
            // for whichever kind of target you did not pick.
            val image = outputImage
            val r = withContext(Dispatchers.IO) {
                if (image != null)
                    GallerySaver.saveImage(this@MainActivity, image, file.name)
                else
                    GallerySaver.save(this@MainActivity, file)
            }
            r.onSuccess {
                savedUri = it
                status = if (image != null) "Saved to Pictures/FaceFusion"
                         else "Saved to Movies/FaceFusion"
            }.onFailure { status = "Save failed: ${it.message}" }
        }
    }

    /**
     * Lift the frame currently on screen out of the finished video and save it as a PNG.
     *
     * MediaMetadataRetriever with OPTION_CLOSEST, the same option PreviewEngine uses and
     * for the same reason: OPTION_PREVIOUS_SYNC snaps to a keyframe, so asking for the
     * frame you are looking at would hand back a different one whenever the scrub position
     * sits inside a GOP.
     */
    private fun saveFrameAt(positionMs: Int) {
        val file = outputFile ?: return
        lifecycleScope.launch {
            val r = withContext(Dispatchers.IO) {
                runCatching {
                    val mmr = MediaMetadataRetriever().apply { setDataSource(file.absolutePath) }
                    val bmp = mmr.getFrameAtTime(positionMs * 1000L,
                                                 MediaMetadataRetriever.OPTION_CLOSEST)
                    mmr.release()
                    bmp ?: error("no frame at that position")
                }.mapCatching { bmp ->
                    val name = file.nameWithoutExtension + "_%06dms.png".format(positionMs)
                    GallerySaver.saveImage(this@MainActivity, bmp, name).getOrThrow()
                }
            }
            r.onSuccess { status = "Frame saved to Pictures/FaceFusion" }
             .onFailure { status = "Save frame failed: ${it.message}" }
        }
    }

    private fun shareVideo(file: File) {
        val uri = savedUri ?: run {
            status = "Save to gallery first, then share"
            return
        }
        // The mime has to match what was actually saved, or the chooser offers apps that
        // cannot open it -- an image result went out as video/mp4 before image targets
        // existed, and would have silently kept doing so.
        val image = outputImage != null
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
            type = if (image) "image/png" else "video/mp4"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }, if (image) "Share swapped image" else "Share swapped video"))
    }

    /**
     * Headless self-test, so the in-process DSP path can be checked over adb:
     *   adb shell am start -n com.facefusion.mobile/.MainActivity --es selftest 1
     *   adb logcat -s ffselftest
     */
    private fun selfTest() {
        val tag = "ffselftest"
        lifecycleScope.launch(Dispatchers.Default) {
            fun say(s: String) = android.util.Log.i(tag, s)
            try {
                val models = modelDir()
                models.listFiles()?.forEach { say("  ${it.name} ${it.length()} readable=${it.canRead()}") }
                val libDir = applicationInfo.nativeLibraryDir
                say("tier: $tier")
                say("fp16: ${NativePipe.probeFp16(libDir, libDir, canaryDir().absolutePath)}")
                if (!NativePipe.init(libDir, libDir, models.absolutePath, opts)) {
                    say("INIT FAILED: ${NativePipe.lastError()}"); return@launch
                }
                say("QNN init OK")
                say("content gate: " +
                    (if (NativePipe.contentGateIsQuantised()) "W8A16 (biased)" else "fp32"))
                // The app's OWN external files dir first. /sdcard/Download is owned by
                // whichever app adb pushed through, mode 660, so this app cannot read it
                // and File.exists() answers false with no hint why -- the real app never
                // hits that because SAF hands it a content URI with access attached.
                fun asset(name: String): File {
                    val mine = File(getExternalFilesDir(null), name)
                    return if (mine.canRead()) mine else File("/sdcard/Download/$name")
                }
                val srcFile = asset("ff_source.jpg")
                val tgtFile = asset("ff_target.mp4")
                // Gate whatever assets are present, and say so per asset: this is the only
                // way the JNI path gets exercised over adb, and a gate that is never run
                // is a gate nobody knows is broken.
                // ⚠ These BLOCK. They used to print the verdict and carry on, which is
                // worse than not checking at all: the log said "gate source: BLOCK" and
                // then a swapped selftest.mp4 appeared next to it. A gate that reports
                // without refusing is decoration.
                if (srcFile.exists()) {
                    val b = decodeOriented(srcFile)
                    if (b != null) ContentGate.checkImage(b).let {
                        say("gate source: %s score %+.4f %s"
                            .format(it.verdict, it.score, it.detail))
                        if (!it.ok) { say("SELFTEST REFUSED: source"); return@launch }
                    }
                }
                if (tgtFile.exists()) ContentGate.checkVideo(tgtFile).let {
                    say("gate target: %s worst %+.4f %s"
                        .format(it.verdict, it.score, it.detail))
                    if (!it.ok) { say("SELFTEST REFUSED: target"); return@launch }
                }
                if (!srcFile.exists() || !tgtFile.exists()) {
                    say("SELFTEST PARTIAL: DSP reachable, no test assets"); return@launch
                }
                val bmp = decodeOriented(srcFile) ?: return@launch
                val soft = bmp.copy(Bitmap.Config.ARGB_8888, false)
                val px = IntArray(soft.width * soft.height)
                soft.getPixels(px, 0, soft.width, 0, 0, soft.width, soft.height)
                if (!NativePipe.setSource(NativePipe.argbToBgr(px, soft.width, soft.height),
                                          soft.width, soft.height)) {
                    say("SOURCE FAILED: ${NativePipe.lastError()}"); return@launch
                }
                val out = File(getExternalFilesDir(null), "selftest.mp4")
                val t1 = System.currentTimeMillis()
                VideoSwapper(onProgress = { d, t -> if (d % 25 == 0) say("frame $d/$t") },
                             onLog = { say(it) })
                    .swap(tgtFile.absolutePath, out.absolutePath)
                    .fold({ say("SELFTEST OK -> $it in ${(System.currentTimeMillis() - t1) / 1000.0} s") },
                          { say("SWAP FAILED: ${it.message}") })
            } catch (t: Throwable) {
                android.util.Log.e(tag, "SELFTEST EXCEPTION", t)
            } finally {
                NativePipe.release()
            }
        }
    }
}
