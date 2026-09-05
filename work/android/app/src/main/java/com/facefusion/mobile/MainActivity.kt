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
import androidx.compose.ui.res.stringResource

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

    /**
     * The DRIVING audio for lip sync -- deliberately independent of [targetFile]. Its own
     * file, its own timeline, no relation to the target's trim. See [VideoSwapper]'s
     * `voicePath` doc for why this exists: syncing a clip to its own original audio has no
     * effect to have, so the UI requires this before Lip Sync can run at all (below).
     */
    private var voiceFile by mutableStateOf<File?>(null)
    private var voiceName by mutableStateOf<String?>(null)
    /** True while the microphone is capturing a driving voice. */
    private var recordingVoice by mutableStateOf(false)

    /**
     * Bumped every time a target is loaded or cleared.
     *
     * ⚠ Exists because `targetFile` CANNOT serve as a state key. Every target is copied to
     * the same path -- `File(cacheDir, "target.mp4")` -- and `File.equals` compares path
     * strings, so the File for a new video is EQUAL to the File for the old one. A
     * `LaunchedEffect` keyed on it therefore does not re-fire, and the auto-warm that draws
     * the swapped pane never runs for any target after the first. Measured on the bench:
     * loading a second clip produced no `autowarm fired` line at all.
     *
     * It was invisible until 0.4.5 because the pane fell back to the PREVIOUS target's
     * swapped frame; clearing that stale frame is what turned a wrong preview into an
     * empty one, and an empty one is what got reported.
     */
    private var targetVersion by mutableStateOf(0)
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

    private var statusText by mutableStateOf("")

    /**
     * Whether [status] is a FAILURE, as opposed to progress or a refusal.
     *
     * Kept beside the text instead of being recovered from it. SwapScreen used to decide
     * this with `status.startsWith("Failed")`, which reads a sentence written for the user
     * -- so the moment that sentence is translated, the bug-report button disappears in
     * every language except English.
     *
     * A content refusal is deliberately NOT an error: nothing malfunctioned, and offering
     * to file a bug about a working gate is noise. That matches what the prefix test did,
     * since the gate's own sentence never began with "Failed".
     */
    private var statusIsError by mutableStateOf(false)

    /**
     * The status line. Assigning it clears [statusIsError]; use [failStatus] for a failure.
     *
     * A property rather than a plain field so that every one of the two dozen
     * `status = "..."` sites keeps meaning "this is not an error" without each having to
     * say so.
     */
    private var status: String
        get() = statusText
        set(value) { statusText = value; statusIsError = false }

    /** A status that IS a failure, and should offer the bug report. */
    private fun failStatus(value: String) { statusText = value; statusIsError = true }
    private var log by mutableStateOf("")
    private var busy by mutableStateOf(false)
    private var preparing by mutableStateOf(false)
    private var progress by mutableStateOf(0f)
    private var framesDone by mutableStateOf(0)
    private var framesTotal by mutableStateOf(0)
    private var elapsedS by mutableStateOf(0.0)
    private var preview by mutableStateOf<Bitmap?>(null)
    private var outputFile by mutableStateOf<File?>(null)

    /**
     * Whether the finished output stopped early because the user cancelled.
     *
     * A cancelled run now KEEPS its frames (VideoSwapper), which is what was asked for --
     * but a 4-second file from a 30-second clip must not read as a completed run, so the
     * pane says so.
     */
    private var outputPartial by mutableStateOf(false)
    private var savedUri by mutableStateOf<Uri?>(null)

    /**
     * Where the last save actually landed, written by whoever saved it.
     *
     * Not rebuilt during composition from `outputFile`: a still goes to Pictures and a
     * video to Movies, and the pane used to name Movies for both -- so an image result
     * reported a path with nothing at it.
     */
    private var savedPathLabel by mutableStateOf<String?>(null)

    // ---- UI shell
    private var screen by mutableStateOf(Screen.Swap)

    // ---- Live (dev builds only; the tab does not exist otherwise) ----
    private val live = LiveEngine()
    private var liveFrame by mutableStateOf<Bitmap?>(null)
    private var liveFps by mutableStateOf(0.0)
    private var liveFaces by mutableStateOf(0)
    private var liveNote by mutableStateOf<String?>(null)
    // ⚠ Compose state, NOT live.isRunning. A plain field on the engine is invisible to
    // recomposition, so the first build showed a running feed under a button still saying
    // "Start" -- the pixels updated because the bitmap reference changed and nothing else
    // did.
    private var liveRunning by mutableStateOf(false)
    // Off = the forced fast preset. Kept out of SwapOptions on purpose: it is a property of
    // this screen, not of a swap, and persisting it would let a Live choice change what a
    // file run does.
    private var liveUseMySettings by mutableStateOf(false)
    private val askCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startLive()
        else liveNote = "Camera permission denied"
    }
    private var advancedOpen by mutableStateOf(false)
    private var modelsVersion by mutableStateOf(0)
    private var deviceUi by mutableStateOf(DeviceUi())

    /** "" | "qnn" | "ncnn" -- the runtime pinned in Settings, mirrored for composition. */
    private var forcedBackend by mutableStateOf("")
    private var confirmMetered by mutableStateOf(false)

    /**
     * A processor whose model is missing was tapped; the name to name in the prompt.
     *
     * The Processors row lists every stage whether or not its model is on the device,
     * so tapping one that is absent has to lead somewhere. It leads here, and here leads
     * to the same download the Settings inventory starts -- there is one downloader and
     * it fetches whatever the manifest says is missing.
     */
    private var confirmModel by mutableStateOf<String?>(null)

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

    /** The debounced redraw owned by [previewOptionsChanged]. One at a time. */
    private var refreshJob: Job? = null

    /**
     * A redraw that was asked for while one was already running.
     *
     * ⚠ [refreshSwapped] used to simply `return` when `previewBusy`, and nothing ever asked
     * again -- so a request that arrived during a preview swap was DROPPED. Scrubbing
     * produces exactly that: the seek job waits, sets the frame, waits again and calls
     * refreshSwapped, and if the previous swap is still running at that instant the new
     * position is silently discarded and the swapped pane keeps showing the old one until
     * something else happens to trigger it. The whole point of a preview is that it follows
     * the handle, so it coalesces now instead of dropping.
     */
    private var refreshPending = false
    private var refreshPendingForce = false

    /** Set by the Cancel button, read by the swap worker. Volatile: different threads. */
    @Volatile private var cancelRequested = false

    /**
     * The file names the hosted manifest publishes for this device's tier.
     *
     * Empty until the fetch lands, and empty forever when offline -- both mean "no row
     * offers a download", which is the honest state rather than a button that cannot work.
     */
    /**
     * What the manifest publishes for this tier: file name -> its LENGTH.
     *
     * The length is why this is a map and not a set. It is the same field
     * `ModelDownload.missing` re-fetches on, so a row that says "update available"
     * and the downloader that would act on it cannot disagree.
     */
    private var hostedFiles by mutableStateOf<Map<String, Long>>(emptyMap())

    /**
     * Where the pipeline was, last time this screen had it.
     *
     * If the API server used it in between, what is loaded is the server's source and the
     * server's options, and the preview's warm flag would otherwise be a promise about
     * somebody else's face. See [PipeGuard.sequence].
     */
    private var lastPipeSeq = -2

    /**
     * The ONE place a URI becomes the source face.
     *
     * Extracted so the gallery picker and the camera capture cannot drift apart. They must
     * not: the content gate runs on the source in refreshSwapped, and it gets there because
     * previewOptionsChanged() is called here. A second path that set sourceUri and forgot
     * this call would be an ungated source -- which is the failure mode the gate's path
     * list exists to prevent, arriving by way of a convenience button.
     */
    private fun setSourceFrom(uri: Uri) {
        sourceUri = uri
        sourceThumb = decodeOriented(uri)
        // A different face means the loaded pipeline is holding the wrong embedding.
        previewOptionsChanged()
    }

    private val pickSource = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) setSourceFrom(uri)
    }

    private val takeSourcePhoto = registerForActivityResult(
        ActivityResultContracts.TakePicture()) { ok ->
        val uri = pendingCapture; pendingCapture = null
        if (ok && uri != null) setSourceFrom(uri)
    }
    /**
     * Where the system camera is writing, between launching it and its result arriving.
     *
     * The contracts report only success/failure -- the URI is the one WE supplied, so it
     * has to survive the round trip here. Cleared on the way out either way, so a cancelled
     * capture cannot be mistaken for the next one.
     */
    private var pendingCapture: Uri? = null

    /** A content:// URI for a new file under files/captures, via the manifest's provider. */
    private fun newCaptureUri(ext: String): Uri {
        val dir = File(filesDir, "captures").apply { mkdirs() }
        val f = File(dir, "cap_${System.currentTimeMillis()}.$ext")
        return androidx.core.content.FileProvider.getUriForFile(
            this, "$packageName.fileprovider", f)
    }

    private val takePhoto = registerForActivityResult(
        ActivityResultContracts.TakePicture()) { ok ->
        val uri = pendingCapture; pendingCapture = null
        if (ok && uri != null) loadTarget(uri)
    }
    private val recordVideo = registerForActivityResult(
        ActivityResultContracts.CaptureVideo()) { ok ->
        val uri = pendingCapture; pendingCapture = null
        if (ok && uri != null) loadTarget(uri)
    }

    /**
     * Launch the system camera for a target.
     *
     * ⚠ The CAMERA permission is REQUESTED first even though the system camera app is what
     * actually opens the lens. An app that DECLARES android.permission.CAMERA -- which this
     * one does, for the Live tab -- must also hold it before ACTION_IMAGE_CAPTURE will
     * start, or the intent fails. An app that never declared it would need no such thing,
     * which is why this looks unnecessary and is not.
     */
    private fun capture(video: Boolean, forSource: Boolean = false) {
        if (checkSelfPermission(android.Manifest.permission.CAMERA) !=
            android.content.pm.PackageManager.PERMISSION_GRANTED) {
            pendingCaptureIsVideo = video
            pendingCaptureForSource = forSource
            askCameraForCapture.launch(android.Manifest.permission.CAMERA)
            return
        }
        val uri = newCaptureUri(if (video) "mp4" else "jpg")
        pendingCapture = uri
        when {
            // A source face is an identity, so there is no video form of it -- the source
            // capture is stills only, and forSource is never combined with video.
            forSource -> takeSourcePhoto.launch(uri)
            video -> recordVideo.launch(uri)
            else -> takePhoto.launch(uri)
        }
    }

    private var pendingCaptureIsVideo = false
    private var pendingCaptureForSource = false
    private val askCameraForCapture = registerForActivityResult(
        ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) capture(pendingCaptureIsVideo, pendingCaptureForSource)
        else status = getString(R.string.status_camera_denied)
    }

    // OpenDocument rather than GetContent: GetContent takes ONE mime filter, and the
    // target can now be a video or a still.
    private val pickTarget = registerForActivityResult(
        ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) loadTarget(uri)
    }
    // Audio or video -- upstream's own `source_paths` takes either and reads whichever
    // track is there, so a dubbed line saved as a short video should not be rejected for
    // carrying pixels it will never use.
    private val pickVoice = registerForActivityResult(
        ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) loadVoice(uri)
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
     * Which `<name>_<tier>.bin` this chip needs, resolved against disk on EVERY read.
     *
     * ⚠ A `by lazy` here was half of the 0.2.0 "downloads the models, then says there are
     * no models" bug. It is first read before anything is downloaded, when the only answer
     * available is the best tier the chip could load; the download then lands a DIFFERENT
     * tier, and the cached value never catches up. A getter costs a few `canRead()` calls
     * -- [ModelPaths] caches the expensive half, the probe.
     */
    private val tier: String get() = ModelPaths.tier(this)

    /**
     * Every tier this chip can load, best first. See [NativePipe.probeTierChain] -- it is
     * NOT "this tier and every older one", so it must not be reconstructed here.
     */
    private val tierChain: List<String> get() = ModelPaths.tierChain(this)

    /**
     * Whether the second swapper is on the device at all.
     *
     * `inswapper_128` is converted and one flag away, but it is another 136 MB that
     * install_app.ps1 only pushes when asked. Offering a model the app cannot load would
     * turn a missing file into a failed run, so the choice appears only when it is real.
     */
    private val hasInswapper: Boolean by lazy {
        ModelPaths.present(modelDir(), tier, "inswapper")
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
        get() = ModelPaths.present(modelDir(), tier, "gpen")

    /**
     * Whether the lip syncer's binary is on the device.
     *
     * Same rule as [hasEnhancer], and for the same reason: asked of the filesystem so the
     * chip is drawable before a pipeline exists, with the native side still the authority
     * at execution time.
     */
    private val hasLipSyncer: Boolean
        // edtalk only. wav2lip was accepted here while ffpipe still fell back to it; both
        // went together, deliberately, because offering Lip Sync for a model the pipeline
        // now refuses is worse than saying the model is missing.
        get() = ModelPaths.present(modelDir(), tier, "edtalk")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        BugReport.install(this)
        NativePipe.ensureLoaded()
        // BEFORE the first probe, and that ordering is the whole point: ModelPaths caches
        // the backend and the tier chain on first ask, so a runtime pinned in Settings and
        // applied any later would be ignored for the life of the process.
        // adb: am start -n com.facefusion.mobile/.MainActivity --es backend ncnn|qnn|auto
        //
        // The runtime, settable from the shell, BEFORE ModelPaths.apply pushes it down.
        // Settings has the same control, but a chip that has to be tapped cannot be part of
        // a scripted device run -- and every other verification in this project is a script
        // (install_app.ps1, run_cli.ps1, the selftest below). The non-Qualcomm path is the
        // one that most needs that, because it is the path no device here selects on its
        // own.
        //
        // Safe to leave in a shipping build for the reason the `api` extra is not: this
        // picks between two runtimes that both run the same gated pipeline. It cannot open
        // a port, and it cannot reach anything the Settings screen does not already offer.
        intent?.getStringExtra("backend")?.let {
            val want = if (it == "auto") "" else it
            if (want == "" || want == "qnn" || want == "ncnn")
                ModelPaths.setForcedBackend(this, want)
        }
        ModelPaths.apply(this)
        forcedBackend = ModelPaths.forcedBackend(this)
        modelDir()
        opts = SwapOptions.load(this)
        ApiService.restore(this)

        // adb: ... --es selftest 1 --es enhance 1 --es lipsync 1   (or `0` to force OFF)
        //
        // The enhancer is opt-in and off by default, so the selftest never touched it --
        // which meant gpen had never run through the APK on EITHER backend, and on ncnn it
        // is the one model with a history of being wrong rather than slow. Not persisted:
        // this changes the run, not the user's settings.
        //
        // `lipsync` is here for the same reason and it is the ONLY way the lip syncer gets
        // exercised over adb: the chip needs a target with an audio track and a hand on the
        // screen, and a stage nobody can run from a script is a stage that only breaks in
        // the field.
        //
        // ⚠ Both extras are THREE-STATE, and that is not cosmetic. They used to only ever
        // turn a stage ON, so a run without them inherited `opts` from the user's SAVED
        // settings -- and a "plain" baseline measured with the enhancer and the lip syncer
        // still enabled from an earlier run looks exactly like a plain one in the log.
        // That corrupted a real measurement (roadmap 9, 2026-09-04). `--es lipsync 0`
        // forces OFF; omitting it still means "whatever is saved".
        if (intent?.getStringExtra("selftest") != null) {
            fun flag(name: String): Boolean? = intent?.getStringExtra(name)?.let {
                it != "0" && !it.equals("false", ignoreCase = true)
            }
            flag("enhance")?.let { opts = opts.copy(faceEnhance = it) }
            flag("lipsync")?.let { opts = opts.copy(lipSync = it) }
            // `--es voice <name>`: a driving-audio file under the app's own files dir (or
            // /sdcard/Download), scriptable over adb without going through the SAF picker.
            // Bench-only wiring for `VideoSwapper.voicePath` -- see `runSwap`'s doc for why
            // the real UI never lets this stay null while Lip Sync is on.
            selfTest(intent?.getStringExtra("voice")); return
        }

        // adb: am start -n com.facefusion.mobile/.MainActivity --es api start
        //
        // LOOPBACK ONLY, and deliberately not parameterised: this Activity is exported, so
        // an extra that could bind 0.0.0.0 would be a remote-exposure switch any app on the
        // phone could flip without the user seeing the screen. Opening the port to the
        // network stays a deliberate touch on a switch that says what it does. The token is
        // required either way.
        if (intent?.getStringExtra("api") == "start")
            toggleApi(on = true, lan = false, remember = false)

        sweepOrphanedOutputs()
        probeDevice()

        refreshModelsMissing()
        refreshHostedFiles()

        // adb: am start -n com.facefusion.mobile/.MainActivity --es download 1
        //
        // The same call the Download button makes, so the DOWNLOADER can be verified on a
        // device rather than read. It resolves the manifest for whichever runtime is
        // active, which is the half that differs between them -- QNN tiers live under
        // `tiers`, the ncnn set under its own key with an `ncnn/` prefix on every filename
        // -- and that difference is invisible from the outside until a file lands in the
        // wrong place or not at all.
        //
        // Nothing here is privileged: it fetches only what the manifest publishes for a
        // runtime the app already chose, and every file is still SHA256-verified before it
        // takes its real name.
        if (intent?.getStringExtra("download") != null) onDownloadTapped()

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
                    // The Settings inventory is not observable state -- it asks the
                    // filesystem during composition and reads modelsVersion to know when
                    // to ask again. Without this, a model that finished downloading while
                    // that screen was open stayed listed as missing.
                    modelsVersion++
                    // ~300 MB is long enough that the user has put the phone down, and the
                    // notification is the only thing that has been speaking to them. This
                    // says so in the app too, but only for a download that actually ENDED
                    // here: the loop also exits on the first composition of an install that
                    // already has its models, and announcing a download to someone who did
                    // not start one is worse than saying nothing.
                    if (announceDownload) {
                        announceDownload = false
                        if (!modelsMissing) toast(getString(R.string.toast_models_ready))
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
                // finishes picking, and cannot repeat because refreshing changes none of
                // its keys.
                //
                // ⚠ It used to require `!previewWarm` as well, which quietly made a WARM
                // pipeline the case it would not redraw for. That was survivable only
                // because both target paths went cold on their own -- the photo path by
                // releasing the whole pipeline, the video path by leaving `preview` set so
                // a stale frame filled the pane instead. Now that a target change keeps the
                // pipeline (see clearPreviewFrames), this is what draws the new target, and
                // the guard would have left the pane empty.
                LaunchedEffect(sourceUri, targetVersion, modelsMissing) {
                    android.util.Log.d("ffpreview", "autowarm fired: src=" +
                        (sourceUri != null) + " tgt=" +
                        (targetFile != null || targetImage != null) +
                        " missing=" + modelsMissing + " run=" + busy)
                    if (sourceUri != null && (targetFile != null || targetImage != null) &&
                        !modelsMissing && !busy) {
                        refreshSwapped(force = true)
                    }
                }
                AppScaffold(
                    screen,
                    {
                        // Leaving the tab stops the feed. Without this the camera keeps
                        // running behind the Swap screen, PipeGuard stays held so a swap
                        // reports the NPU busy, and the pipeline Live configured is still
                        // the loaded one.
                        if (screen == Screen.Live && it != Screen.Live) stopLive()
                        screen = it
                    },
                    // Shown on BOTH lines now. It was dev-only for one reason -- the live
                    // path had no content gate -- and that reason is gone: the camera is
                    // sampled in LiveEngine and the source is checked where it is picked.
                    showLive = true,
                ) { pad ->
                    Box(Modifier.padding(pad)) {
                        when (screen) {
                            Screen.Swap -> SwapScreen(
                                sourceThumb = sourceThumb,
                                hasSource = sourceUri != null,
                                hasTarget = targetFile != null || targetImage != null,
                                imageTarget = targetImage != null,
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
                                    // the same thing one frame later. It KEEPS that frame
                                    // after the run: the run invalidated the preview to hand
                                    // the pipeline over, so falling back to swappedFrame here
                                    // meant the pane emptied itself the instant the swap
                                    // finished and sat on "Preparing preview..." for good --
                                    // nothing re-warms a pipeline the finished run released.
                                    swapped = if (busy) preview else (swappedFrame ?: preview),
                                    timeLabel = if (durationMs > 0) fmt(previewAtMs) else "",
                                    warm = previewWarm,
                                    busy = previewBusy,
                                    note = previewNote,
                                ),
                                run = RunUi(busy, preparing, progress, framesDone,
                                            framesTotal, elapsedS),
                                status = status,
                                statusIsError = statusIsError,
                                log = log,
                                opts = opts,
                                onOptsChange = { o ->
                                    // Only the SWAPPER selects a different model file.
                                    // Everything else is a per-frame value the loaded
                                    // pipeline can simply be told about, so it must not
                                    // send the preview cold -- going cold is what made a
                                    // slider cost a model reload.
                                    val reloads = o.swapper != opts.swapper
                                    opts = o
                                    o.save(this@MainActivity)
                                    // init() consumed the old options; the warm pipeline is
                                    // now showing something the user did not ask for. This
                                    // REDRAWS -- clearing the pane and leaving it cleared is
                                    // how "Preparing preview..." became permanent, since
                                    // nothing was left to ask for the next one.
                                    previewOptionsChanged(reloads = reloads)
                                },
                                hasInswapper = hasInswapper,
                                hasEnhancer = hasEnhancer,
                                hasLipSyncer = hasLipSyncer,
                                onRequestModel = { confirmModel = it },
                                openCard = openCard,
                                onToggleCard = { k -> openCard = if (openCard == k) "" else k },
                                // A still needs no run, so it has no output FILE -- what
                                // there is to save is the pane itself.
                                hasOutput = if (targetImage != null) swappedFrame != null
                                            else outputFile != null,
                                outputFile = outputFile,
                                outputPartial = outputPartial,
                                onSaveFrame = ::saveFrameAt,
                                onSavePreviewFrame = ::savePreviewFrame,
                                saved = savedUri != null,
                                savedPath = savedPathLabel,
                                onPickSource = { pickSource.launch("image/*") },
                                onPickTarget = {
                                    pickTarget.launch(arrayOf("video/*", "image/*"))
                                },
                                onClearSource = ::clearSource,
                                onCaptureSource = { capture(video = false, forSource = true) },
                                onCapturePhoto = { capture(video = false) },
                                onCaptureVideo = { capture(video = true) },
                                onClearTarget = ::clearTarget,
                                onDeleteOutput = ::discardOutput,
                                hasVoice = voiceFile != null,
                                voiceName = voiceName,
                                onPickVoice = { pickVoice.launch(arrayOf("audio/*", "video/*")) },
                                onClearVoice = ::clearVoice,
                                recordingVoice = recordingVoice,
                                onToggleRecordVoice = ::toggleVoiceRecording,
                                onSwap = { runSwap() },
                                onCancel = {
                                    cancelRequested = true
                                    status = getString(R.string.status_cancelling)
                                },
                                modelsMissing = modelsMissing,
                                onDownload = { onDownloadTapped() },
                                onShareLog = { shareBugReport() },
                                // A still saves the pane; a video saves its file.
                                onSave = {
                                    if (targetImage != null) saveSwappedStill()
                                    else outputFile?.let { saveToGallery(it) }
                                },
                                onShare = { shareResult() },
                            )

                            Screen.Live -> LiveScreen(
                                sourceThumb = sourceThumb,
                                onPickSource = { pickSource.launch("image/*") },
                                onClearSource = ::clearSource,
                                onCaptureSource = { capture(video = false, forSource = true) },
                                frame = liveFrame,
                                running = liveRunning,
                                onToggleRun = { toggleLive() },
                                fps = liveFps,
                                faces = liveFaces,
                                useMySettings = liveUseMySettings,
                                onUseMySettings = { liveUseMySettings = it },
                                note = liveNote,
                                modelsReady = !modelsMissing,
                                onDownload = { onDownloadTapped() },
                            )
                            Screen.Settings -> SettingsScreen(
                                sections = modelSections(),
                                modelDirPath = modelDir().absolutePath,
                                device = deviceUi,
                                onDownloadModel = { onDownloadTapped() },
                                onDeleteModel = { m ->
                                    // Every file of the row, not just the one it is named
                                    // after: an ncnn model is a param/bin pair.
                                    m.files.forEach { File(modelDir(), it).delete() }
                                    modelsVersion++
                                    // Hard: the file behind the loaded pipeline just went
                                    // away, and the redraw is what reports which one.
                                    previewOptionsChanged(hard = true)
                                },
                                onApiToggle = ::toggleApi,
                                onApiLan = ::setApiLan,
                                onShareBugReport = { shareBugReport() },
                                forcedBackend = forcedBackend,
                                // null in a QNN-only build, which is what hides the
                                // control rather than drawing a dead one.
                                onForceBackend =
                                    if (NativePipe.hasNcnnBackend()) ::onForceBackend
                                    else null,
                            )
                        }
                    }

                    confirmModel?.let { name ->
                        AlertDialog(
                            onDismissRequest = { confirmModel = null },
                            title = { Text(stringResource(R.string.proc_get_title, name)) },
                            text = { Text(stringResource(R.string.proc_get_body)) },
                            confirmButton = {
                                TextButton({ confirmModel = null; onDownloadTapped() }) {
                                    Text(stringResource(R.string.proc_get_confirm))
                                }
                            },
                            dismissButton = {
                                TextButton({ confirmModel = null }) {
                                    Text(stringResource(R.string.proc_get_cancel))
                                }
                            },
                        )
                    }

                    if (confirmMetered) {
                        AlertDialog(
                            onDismissRequest = { confirmMetered = false },
                            title = { Text(stringResource(R.string.dl_metered_title)) },
                            text = {
                                Text(
                                    stringResource(R.string.dl_metered_body)
                                )
                            },
                            confirmButton = {
                                TextButton({ confirmMetered = false; beginDownload() }) {
                                    Text(stringResource(R.string.dl_metered_confirm))
                                }
                            },
                            dismissButton = {
                                TextButton({ confirmMetered = false }) {
                                    Text(stringResource(R.string.dl_metered_wait))
                                }
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

    /**
     * The same start-the-server extra, for an app that is ALREADY running.
     *
     * `am start` on a live Activity delivers here, not to onCreate, so without this the
     * documented headless line silently did nothing whenever the app happened to be open --
     * which looks identical to the server crashing on startup.
     */
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        if (intent.getStringExtra("api") == "start")
            toggleApi(on = true, lan = false, remember = false)
    }

    /**
     * Pin the neural runtime, or "" for automatic.
     *
     * ⚠ The ORDER here is not arrangeable. `NativePipe.release()` must run FIRST, while the
     * old backend is still the active one: the pipeline's handles are freed through the
     * seam, and although they now carry the backend that opened them (ffnn.cpp), a pipeline
     * half-owned by a runtime the process has moved off is not a thing to keep around.
     * Then the setting, then the caches, then a fresh probe.
     *
     * Everything the previous runtime produced is discarded with it -- the loaded models,
     * the swapped preview, and the answer to "which models are missing", which is a
     * different question on each backend.
     */
    private fun onForceBackend(value: String) {
        if (value == forcedBackend) return
        NativePipe.release()
        ModelPaths.setForcedBackend(this, value)
        forcedBackend = value
        deviceUi = DeviceUi()
        probeDevice()
        refreshModelsMissing()
        refreshHostedFiles()
        // Hard: the models behind the loaded pipeline are not merely stale, they are the
        // other runtime's file format.
        previewOptionsChanged(hard = true)
        appendLog("runtime: " + (if (value.isEmpty()) "automatic" else value))
    }

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
                // Reported even when ok=0: on a part with no Hexagon the HTP probe is
                // MEANT to fail, and this is the row that says what is running instead.
                backend = kv["backend"].orEmpty(),
                gpu = kv["gpu"] == "1",
            )
        }
    }

    /**
     * Ask the manifest what exists for this tier. One ~2 KB request, once per launch.
     *
     * Failure is silent and total: [hostedFiles] stays empty, so Settings offers no
     * downloads at all rather than offering one that cannot succeed.
     */
    private fun refreshHostedFiles() = lifecycleScope.launch(Dispatchers.IO) {
        val hosted = runCatching {
            ModelDownload.manifestFor(tierChain.joinToString(","))
                .second.associate { it.name to it.bytes }
        }.getOrDefault(emptyMap())
        withContext(Dispatchers.Main) { hostedFiles = hosted }
    }

    // ------------------------------------------------------------------ remote API

    /**
     * Start, stop, or rebind the HTTP server.
     *
     * Rebinding is a restart, not a flag flip: the bind address is fixed when the socket is
     * created, so a switch that changed [lan] on a live server would report an address it
     * is not listening on.
     */
    /**
     * The LAN preference, which is NOT the same question as "is the server running".
     *
     * Separated from [toggleApi] because routing it through there made it a no-op whenever
     * the server was off: `toggleApi` stops and returns when `on` is false, so the
     * preference was never written and the switch snapped back. The two switches were never
     * meant to depend on each other -- one opens a port, the other says which address it
     * binds when it does.
     *
     * A running server is REBOUND, because the address is fixed when the socket opens: a
     * live flag change would leave it reporting an address it is not listening on.
     */
    private fun setApiLan(lan: Boolean) {
        if (lan == ApiService.allowLan) return
        ApiService.setLan(this, lan)
        if (ApiService.running) {
            ApiService.stop(this)
            ApiService.start(this, lan)
        }
    }

    private fun toggleApi(on: Boolean, lan: Boolean, remember: Boolean = true) {
        if (!on) { ApiService.stop(this); return }
        if (android.os.Build.VERSION.SDK_INT >= 33)
            askNotify.launch(android.Manifest.permission.POST_NOTIFICATIONS)
        if (ApiService.running && lan != ApiService.allowLan) ApiService.stop(this)
        ApiService.start(this, lan, remember)
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
        // Say so when it does not go. The share used to be fire-and-forget, so a device
        // with nothing that accepts text/plain -- or a report too large for a Binder
        // transaction -- produced a button that did visibly nothing.
        val err = BugReport.share(this, BugReport.compose(this, log, npu, models, status))
        if (err != null) {
            status = getString(R.string.status_share_failed, err)
            return
        }
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
        // ModelPaths, not a fourth copy of the same list, and NOT deviceUi.tier -- see the
        // warning on ModelPaths.tier.
        val absent = ModelPaths.missing(this, tier, opts.swapper)
        val now = absent.isNotEmpty()
        if (now != modelsMissing)
            android.util.Log.i("ffmodels", "missing=" + now + " " + absent)
        modelsMissing = now
    }

    /**
     * Record a tier this chip loaded and would not run, and re-ask what is missing.
     *
     * Called after EVERY init, successful or not. A device can reject its best tier and
     * come up on the next one, and that rejection is worth keeping either way -- it is a
     * property of the silicon, and re-proving it costs a full context load per launch.
     *
     * The `refreshModelsMissing` is the half that matters to the user. Rejecting v81 drops
     * it from the chain, which makes v73 the tier this device wants and v73 is not on
     * disk -- so the download overlay has to re-ask, or they are left looking at an error
     * with nothing on screen that would act on it.
     */
    private fun noteTierRejection() {
        val bad = NativePipe.rejectedTier()
        if (bad.isBlank() || bad in ModelPaths.rejectedTiers(this)) return
        appendLog("tier $bad will not execute on this device; falling back")
        ModelPaths.rejectTier(this, bad)
        refreshModelsMissing()
    }

    /**
     * A short confirmation, on top of the status line.
     *
     * The status line is where a save already reported itself, and it is easy to miss: it
     * sits below the panes, it is one line among several, and on a save the user is usually
     * looking at the pane they just saved rather than at it. A toast says the thing landed
     * without the user having to go looking for the sentence that says so.
     */
    private fun toast(text: String) {
        android.widget.Toast.makeText(this, text, android.widget.Toast.LENGTH_SHORT).show()
    }

    private fun onDownloadTapped() {
        if (android.os.Build.VERSION.SDK_INT >= 33)
            askNotify.launch(android.Manifest.permission.POST_NOTIFICATIONS)
        // Metered is a warning, not a refusal: the user may have no Wi-Fi and still want it.
        if (ModelDownload.isMetered(this)) confirmMetered = true else beginDownload()
    }

    /** True only between starting a download and reporting that it finished. */
    private var announceDownload = false

    private fun beginDownload() {
        ModelDownload.reset()
        announceDownload = true
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
    /**
     * Every model SET on this device: the one in use, then any other that has files.
     *
     * A phone that has run both runtimes holds two, and before this only the active one
     * appeared anywhere -- so the ~600 MB of ncnn weights an NPU build never touches had no
     * screen on which they existed, and no way to be deleted short of switching runtime
     * back and forth.
     *
     * The inactive sets are read from DISK rather than from the chip or the manifest: what
     * matters about them is only that they are taking up space.
     */
    private fun modelSections(): List<ModelSection> {
        val active = tier
        val others = ModelPaths.variantsOnDisk(this).filter { it != active }
        fun title(v: String) =
            if (v == ModelPaths.NCNN_TIER) getString(R.string.set_models_gpu)
            else getString(R.string.set_models_npu, v)
        return listOf(ModelSection(title(active), "", modelRows(active), active = true)) +
            others.map { ModelSection(title(it), "", modelRows(it), active = false) }
    }

    private fun modelRows(t: String): List<ModelRow> {
        val ignored = modelsVersion
        check(ignored >= 0)
        // "Required" has to mean what ModelPaths.missing() ENFORCES, or this screen tells
        // a user everything is fine about a device that then refuses to swap.
        //
        // Which swapper is required depends on the one selected: missing() checks
        // opts.swapper, so hyperswap is not inherently the required one and inswapper is
        // not inherently optional -- the SELECTED one is required and the other is the
        // alternative.
        val alt = if (opts.swapper == "inswapper") "hyperswap" else "inswapper"
        val required = listOf(
            "yoloface" to getString(R.string.model_detector),
            "fan2d" to getString(R.string.model_landmarker),
            "arcface" to getString(R.string.model_recogniser),
            opts.swapper to getString(R.string.model_swapper),
        )
        // The content gate is required -- missing() blocks a run without it and
        // ffpipe::init will not come up -- but it is satisfied by EITHER build, so neither
        // file is required ON ITS OWN. fp32 `nsfw_` only finalizes on v79 and up; below
        // that the quantised `nsfwq_` is the only one that exists. So the PAIR is what is
        // required, and a row is only worth flagging when NEITHER is on the device --
        // otherwise a v79 phone, which correctly has just `nsfw_`, would be told the
        // `nsfwq_` it must never download is missing and required.
        //
        // ⚠ "v79 and up" was wrong and is now measured: the fp32 gate does NOT build for
        // v81. qnn-context-binary-generator refuses it with "no properties registered for
        // q::QNN_Gelu", so v81 ships `nsfwq_` like the tiers BELOW v79 do. fp32 is a v79
        // fact, not a floor -- which is exactly why this row tests the PAIR and never the
        // arch (docs/traps.md #10).
        val gateOk = ModelPaths.present(modelDir(), t, "nsfw") ||
                     ModelPaths.present(modelDir(), t, "nsfwq2")
        val gate = listOf(
            "nsfw" to getString(R.string.model_content_checker),
            "nsfwq2" to getString(R.string.model_content_checker_quantised),
        )
        val optional = listOf(
            alt to getString(R.string.model_swapper_alt),
            // It was absent from BOTH lists, so a 28 MB model that is on the device, that
            // the Advanced panel offers a switch for, and that /health reports as present,
            // was invisible on the one screen whose job is to say what is installed.
            "gpen" to getString(R.string.model_enhancer),
            // Added the day the lip syncer shipped, and it is the SECOND time this list
            // has caused exactly this bug: the comment above records gpen being invisible
            // on the one screen whose job is to say what is installed. A hardcoded list
            // beside a manifest-driven downloader will drift again -- what saves it is
            // that `row()` reads ModelPaths and `hostedFiles`, so a name added here is
            // enough and a name forgotten is invisible rather than broken.
            // wav2lip gets no row: ffpipe no longer opens it, so it is neither a download
            // that would ever run nor an installed capability the screen can honestly
            // claim. A device that still holds the file keeps a dead ~44 MB until the
            // app's data is cleared.
            "edtalk" to getString(R.string.model_lip_syncer_256),
            "fan685" to getString(R.string.model_landmark_refiner),
        )
        // What can be fetched is whatever the MANIFEST publishes for this tier -- asked
        // once over the network, not guessed here. The guess it replaces had the gate wrong
        // in both directions: a v79 phone would have been offered `nsfwq`, which its tier
        // does not carry and which would therefore never arrive, while a deleted `nsfw` row
        // disappeared from the list entirely, because only gpen was allowed to show itself
        // while absent. Offline the set is empty and no row offers a download, which is
        // correct -- there is nothing to download from.
        fun row(name: String, label: String, req: Boolean): ModelRow {
            // Filenames come from ModelPaths, never from a pattern here: ncnn needs a
            // param/bin PAIR named after the ONNX graph, QNN one context binary with the
            // tier in the name, and a row that spelled either out would be wrong on the
            // other backend.
            val files = ModelPaths.filesFor(t, name).map { File(modelDir(), it) }
            if (files.isEmpty()) return ModelRow(label, name, 0L, false, req)
            val present = files.all { it.canRead() }
            val len = files.sumOf { if (it.exists()) it.length() else 0L }
            // A pair is shown as ONE row under the name the user thinks in. The .param is
            // 30 KB beside a 400 MB .bin, so listing both would be noise, and half a pair
            // is not a state worth a row of its own -- it is simply "not installed".
            val f = files.first()
            val hostedLen = files.sumOf { hostedFiles[it.name] ?: 0L }
                .takeIf { files.all { p -> p.name in hostedFiles } }
            return ModelRow(label, f.name, len, present, req,
                            files = files.map { it.name },
                            downloadable = files.all { it.name in hostedFiles },
                            // Present, published, and a different file from the published
                            // one. Offline `hostedFiles` is empty and nothing is ever called
                            // outdated, which is the right answer when there is nothing to
                            // compare against -- never a scary row because the network is.
                            outdated = present && hostedLen != null && hostedLen != len,
                            // In the queue that is running RIGHT NOW, rather than "some
                            // download exists". The bulk button skips the optional models,
                            // so the two stopped meaning the same thing.
                            fetching = files.any { it.name in ModelDownload.queued })
        }
        return (required.map { (n, l) -> row(n, l, true) } +
            gate.map { (n, l) -> row(n, l, !gateOk) }.filter { it.present || it.downloadable } +
            optional.map { (n, l) -> row(n, l, false) }.filter { it.present || it.downloadable })
            // One row per FILE SET, not per logical name. The two gate names resolve to two
            // different context binaries on QNN and to the SAME `nsfw_2_sim` pair on ncnn --
            // the quantised build exists because a QNN tier below v79 cannot finalize the
            // fp32 gate, which is a QNN fact and means nothing here. Without this the ncnn
            // inventory lists "Content checker" twice, both describing one file.
            .distinctBy { it.fileName }
    }

    // ------------------------------------------------------------------ previews

    /**
     * The warm pipeline no longer matches what the UI is asking for.
     *
     * Tears the native pipeline DOWN, so this is for the cases that need the global gone --
     * a run about to init its own, or a model deleted underneath it. An options change must
     * not come through here: see [previewOptionsChanged].
     *
     * `preview` is cleared with the rest. It holds the last live frame of the previous run,
     * and the swapped pane falls back to it, so leaving it set would show the last frame of
     * the LAST target after picking a new one.
     */
    private fun invalidatePreview() {
        previews.invalidate()
        previewWarm = false
        clearPreviewFrames()
    }

    /**
     * The DISPLAYED preview is stale. What is loaded is not.
     *
     * A new target changes neither of the two things the warm pipeline depends on: the warm
     * key is the options and the SOURCE, and target frames are handed to `processFrame` one
     * at a time. So a target change has to forget the pictures on screen and nothing else.
     *
     * ⚠ Both target paths got this wrong, in opposite directions, and one fix closes both:
     *
     *  - the PHOTO path called [invalidatePreview], which tears the native pipeline down.
     *    Every context reloaded on the next preview to show a frame the loaded models were
     *    already able to swap. Invisible on the NPU and seconds of "Loading models" on the
     *    CPU backend, which is where it was noticed.
     *  - the VIDEO path cleared `swappedFrame` but not `preview`, and the pane resolves
     *    `swappedFrame ?: preview`. `preview` holds the last live frame of the last run, so
     *    picking a new video could show the PREVIOUS target's swapped face over the new
     *    target's original -- the exact failure invalidatePreview's own comment describes,
     *    in the one path that did not call it.
     */
    private fun clearPreviewFrames() {
        swappedFrame = null
        previewNote = null
        preview = null
    }

    /**
     * An option (or the source face) changed, so the pane is showing a stale swap.
     *
     * Deliberately NOT [invalidatePreview]: that releases the native pipeline, and a weight
     * slider emits a value per pixel of drag, so a release would land in the middle of a
     * preview swap running on another thread -- a use-after-free reached by dragging.
     * [PreviewEngine.ensureReady] keys on the options and the source and rebuilds itself
     * when they differ, so marking the pane stale is enough and the rebuild happens once,
     * inside the coroutine that owns the pipeline.
     *
     * Debounced for the same reason: a drag would otherwise ask for a full model reload
     * dozens of times a second. The wait outlives previewBusy so a change made DURING a
     * refresh still lands, rather than being dropped by refreshSwapped's early return.
     */
    private fun previewOptionsChanged(hard: Boolean = false, reloads: Boolean = true) {
        // `hard` is for a model that went away underneath the loaded pipeline. Still not
        // while a swap is reading it: the redraw below re-checks the files and reports the
        // missing one, which is the useful half of tearing it down anyway.
        if (hard && !previewBusy) previews.invalidate()
        // Cold only when the pipeline really has to be rebuilt. `reloads` is false for a
        // per-frame option, whose new value refreshSwapped pushes to the warm pipeline --
        // going cold there would re-decode the source and re-run the gate to change a float.
        if (hard || reloads) previewWarm = false
        swappedFrame = null
        previewNote = null
        preview = null
        savedUri = null; savedPathLabel = null
        refreshJob?.cancel()
        refreshJob = lifecycleScope.launch {
            delay(500)
            while (previewBusy) delay(100)
            refreshSwapped(force = true)
        }
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
            android.util.Log.d("ffpreview", "seek to " + previewAtMs + " edge=" + edge)
            val got = previews.frameAt(previewAtMs)
            // Size only. The pixel sampling that lived here is what proved the retriever was
            // returning identical images for different timestamps (see FrameSeeker); it did
            // its job and does not belong in a shipping build.
            android.util.Log.d("ffpreview", "  frame=" +
                (if (got == null) "NULL" else got.width.toString() + "x" + got.height))
            originalFrame = got
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
    /**
     * Why a redraw did not happen.
     *
     * ⚠ Every `return` below is a silent one: the pane keeps whatever it had and nothing
     * says why, which is exactly how "stuck on Preparing preview" and "the frame does not
     * update on seek" arrive as reports with no evidence attached. The same shape as the
     * QNN error that was thrown away for two releases -- a decision made and not recorded.
     * One line per refusal, at debug level, on a tag nothing else uses.
     */
    private fun skipRefresh(why: String) {
        android.util.Log.d("ffpreview", "refresh skipped: " + why +
            "  warm=" + previewWarm + " busy=" + previewBusy + " run=" + busy +
            " missing=" + modelsMissing + " src=" + (sourceUri != null) +
            " tgt=" + (targetFile != null || targetImage != null))
    }

    private fun refreshSwapped(force: Boolean) {
        if (busy) { skipRefresh("a run owns the pipeline"); return }
        // Coalesced, not dropped: remember that the pane is out of date and redraw once the
        // in-flight one lets go. `force` is OR-ed in so a cold request cannot be downgraded
        // by a warm one arriving behind it.
        if (previewBusy) {
            refreshPending = true
            refreshPendingForce = refreshPendingForce || force
            skipRefresh("one already running; queued")
            return
        }
        if (!previewWarm && !force) { skipRefresh("cold and not forced"); return }
        val src = sourceUri ?: run { skipRefresh("no source"); return }
        // An image target has no targetFile -- it is held as a bitmap. Testing the
        // video handle here meant an image never previewed at all.
        if (targetFile == null && targetImage == null) { skipRefresh("no target"); return }
        android.util.Log.d("ffpreview", "refresh start force=" + force + " warm=" + previewWarm)

        lifecycleScope.launch {
            // The API server may be mid-request. Wait briefly rather than bounce: a preview
            // that quietly does not appear is the bug this whole pass has been about.
            if (!PipeGuard.acquire("preview", 4000)) {
                previewNote = pipeBusyMessage()
                return@launch
            }
            // Someone else has had the pipeline since this screen last used it, so the warm
            // flag describes their source, not ours.
            if (!PipeGuard.uninterrupted(lastPipeSeq)) {
                previews.invalidate()
                previewWarm = false
            }
            lastPipeSeq = PipeGuard.sequence

            previewBusy = true
            previewNote = null
            // Per-frame options onto the WARM pipeline, before the frame is swapped with
            // them. No-ops when nothing is loaded or nothing changed; when the swapper
            // changed it does nothing and ensureReady below does the reload instead.
            previews.applyOptions(opts)
            // Same reasoning, for the lip syncer's driving audio -- decoded once per voice
            // file (see PreviewEngine.applyVoice), called on every refresh rather than only
            // at cold warm-up so a voice picked or changed AFTER the preview is already
            // warm still reaches it. The fps has to match what a real run would index
            // frames at, or the preview and the eventual output disagree on which mouth
            // belongs to which timestamp.
            if (opts.lipSync) {
                val previewFps = if (opts.outputFps in 1..inputFps) opts.outputFps else inputFps
                previews.applyVoice(voiceFile?.absolutePath, previewFps.toDouble())
            }
            try {
                val frame = originalFrame ?: previews.frameAt(previewAtMs)?.also {
                    originalFrame = it
                } ?: run {
                    previewNote = getString(R.string.status_cannot_read_frame)
                    return@launch
                }

                if (!previewWarm) {
                    val bmp = withContext(Dispatchers.IO) { decodeOriented(src) }
                    if (bmp == null) {
                        previewNote = getString(R.string.status_cannot_read_source)
                        return@launch
                    }
                    val soft = bmp.copy(Bitmap.Config.ARGB_8888, false)
                    val px = IntArray(soft.width * soft.height)
                    soft.getPixels(px, 0, soft.width, 0, 0, soft.width, soft.height)

                    val models = modelDir()
                    val t = tier
                    val missing = listOf("yoloface", "fan2d", "arcface", opts.swapper)
                        .filterNot { ModelPaths.present(models, t, it) }
                    if (missing.isNotEmpty()) {
                        previewNote = "Missing ${missing.joinToString()}"
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
                            // The score alone reads "NaN" and stops there. When the gate
                            // FAULTS, the reason is the whole story and it was being
                            // dropped on the floor.
                            appendLog("preview source score %+.3f".format(v.score) +
                                      (if (v.detail.isNotBlank()) "  [" + v.detail + "]" else ""))
                            if (v.ok) null else ContentGate.message(this@MainActivity, R.string.gate_subject_source_image, v)
                        },
                    )
                    // Before the error branch, because a rejection is worth recording even
                    // when the fallback then succeeded and there is no error to report.
                    noteTierRejection()
                    if (err != null) {
                        // The LOG too, not just the pane. A bug report carries the log and
                        // the status line; it does not carry the pane. So the one report
                        // this project most needed to explain -- an 8 Elite Gen 5 whose
                        // v81 tier loads and will not execute -- arrived with the failure
                        // filling the screen and "-- run log --  (empty)" underneath it,
                        // and the user had to photograph the pane to say what happened.
                        appendLog(err)
                        previewNote = err
                        return@launch
                    }
                    previewWarm = true
                }

                // Every previewed frame, not just the source. The source is checked once,
                // when the pipeline warms; the target is checked here because the trim
                // handle can reach any frame in the clip and this pane displays it.
                ContentGate.checkImage(frame).let { v ->
                    if (!v.ok) {
                        previewNote = ContentGate.message(this@MainActivity, R.string.gate_subject_this_frame, v)
                        swappedFrame = null
                        return@launch
                    }
                }

                val out = previews.swap(frame, previewAtMs, voiceFile?.absolutePath)
                android.util.Log.d("ffpreview", "  swapped faces=" + out.faces +
                                                " err=" + out.error)
                when {
                    out.error != null -> { previewNote = out.error; swappedFrame = null }
                    out.faces == 0 -> {
                        // processFrame leaves the buffer untouched when it finds nothing, so
                        // without this the pane would show the ORIGINAL and look like a
                        // swap that did nothing.
                        previewNote = getString(R.string.status_no_face)
                        swappedFrame = null
                    }
                    else -> {
                        swappedFrame = out.bitmap; previewNote = null
                        // For a still this pane IS the result, so a new one is a different
                        // image from the one the Save button reported saving.
                        if (targetImage != null) { savedUri = null; savedPathLabel = null }
                    }
                }
            } finally {
                previewBusy = false
                PipeGuard.release()
                // Whatever arrived while this one held the pipeline. Reads the CURRENT
                // position rather than a remembered one -- the seek job keeps
                // `originalFrame` up to date, so the redraw lands on the newest frame
                // rather than replaying the one that was missed.
                if (refreshPending) {
                    refreshPending = false
                    val f = refreshPendingForce
                    refreshPendingForce = false
                    refreshSwapped(force = f)
                }
            }
        }
    }

    /**
     * Copy the picked target out of SAF.
     *
     * A still short-circuits everything video: there is no duration, so no trim, no frame
     * rate, and nothing to seek. `durationMs == 0` is the mode flag the UI reads.
     */
    /**
     * The file's real name, for a `content://` URI.
     *
     * `uri.lastPathSegment` is NOT the filename for most SAF providers -- it is the
     * provider's own document id, which is very often just a number ("118", "msf:42").
     * `OpenableColumns.DISPLAY_NAME` is the column every provider is required to answer
     * correctly; the segment is only a fallback for the rare URI a query fails on.
     */
    private fun displayName(uri: Uri): String? {
        val queried = runCatching {
            contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME),
                                   null, null, null)?.use { c ->
                if (c.moveToFirst()) c.getString(0) else null
            }
        }.getOrNull()
        return queried ?: uri.lastPathSegment?.substringAfterLast('/')
    }

    private fun loadTarget(uri: Uri) {
        if (contentResolver.getType(uri)?.startsWith("image/") == true) {
            loadTargetImage(uri)
            return
        }
        preparing = true
        targetName = displayName(uri)
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
                    val storedW = mmr.extractMetadata(
                        MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
                    val storedH = mmr.extractMetadata(
                        MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
                    // ⚠ ROTATION-CORRECTED, and it has to be. VIDEO_WIDTH/HEIGHT are the
                    // STORED dimensions: a phone films portrait by recording 1920x1080 and
                    // setting a 90 degree flag, so this reports 16:9 for a clip every other
                    // part of the app treats as 9:16.
                    //
                    // FrameSeeker already decodes AND rotates (see its outWidth/outHeight),
                    // so the pane was being sized 16:9 for a 9:16 bitmap and ContentScale.Fit
                    // did the only thing it could -- letterbox it into grey side bars. It
                    // also chose the stacked layout over the side-by-side one, which exists
                    // precisely for portrait footage. Output was never affected: VideoSwapper
                    // reads the same flag itself.
                    val rot = mmr.extractMetadata(
                        MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
                    val swap = rot == 90 || rot == 270
                    val w = if (swap) storedH else storedW
                    val h = if (swap) storedW else storedH
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
                // The old result belongs to the old target. Dropping it here is also the
                // fix for a stale-result bug: nothing cleared `outputFile` on a new pick, so
                // the pane went on offering to save and share the PREVIOUS video.
                discardOutput()
                targetImage = null
                targetFile = l.file; durationMs = l.durationMs
                inputFps = l.fps
                trimStartMs = 0f; trimEndMs = l.durationMs.toFloat()
                targetAspect = if (l.width > 0 && l.height > 0)
                    l.width.toFloat() / l.height else 16f / 9f
                status = getString(R.string.status_target_ready_video,
                                   l.width, l.height, fmt(l.durationMs.toFloat()))
                // Open it for scrubbing and show the first frame straight away.
                previews.openTarget(l.file.absolutePath, l.fps)
                targetVersion++
                // `preview` too, which this path used to leave set -- see clearPreviewFrames.
                clearPreviewFrames()
                originalFrame = previews.frameAt(0f)
            }.onFailure {
                status = getString(R.string.status_cannot_read_video, it.message ?: "")
            }
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
        targetName = displayName(uri)
        lifecycleScope.launch {
            val bmp = withContext(Dispatchers.IO) { decodeOriented(uri) }
            if (bmp == null) {
                status = getString(R.string.status_cannot_read_image)
                preparing = false
                return@launch
            }
            discardOutput()
            previews.closeTarget()
            targetFile = null
            targetImage = bmp
            durationMs = 0L
            trimStartMs = 0f; trimEndMs = 0f
            targetAspect = if (bmp.height > 0) bmp.width.toFloat() / bmp.height else 1f
            // The frames, not the pipeline. The double `originalFrame = bmp` this replaces
            // was working around invalidatePreview clearing state this path had just set.
            clearPreviewFrames()
            targetVersion++
            originalFrame = bmp
            status = getString(R.string.status_target_ready_image, bmp.width, bmp.height)
            preparing = false
        }
    }

    /**
     * The lip syncer's DRIVING audio -- a file the user chose deliberately, never the
     * target's own track. See [voiceFile]'s doc and [VideoSwapper]'s `voicePath`.
     *
     * Copied to a real path for the same reason [loadTarget] copies the target: both
     * [MediaMetadataRetriever] (the `has_audio` check here) and [AudioDecoder] need one,
     * and SAF only ever hands back a stream.
     */
    private fun loadVoice(uri: Uri) {
        voiceName = displayName(uri)
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    val f = File(cacheDir, "voice.audio")
                    contentResolver.openInputStream(uri).use { i ->
                        f.outputStream().use { o -> i!!.copyTo(o) }
                    }
                    val mmr = MediaMetadataRetriever().apply { setDataSource(f.absolutePath) }
                    val hasAudio = mmr.extractMetadata(
                        MediaMetadataRetriever.METADATA_KEY_HAS_AUDIO) == "yes"
                    val durMs = mmr.extractMetadata(
                        MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
                    mmr.release()
                    if (!hasAudio) error(getString(R.string.status_voice_no_audio))
                    Pair(f, durMs)
                }
            }
            result.onSuccess { (f, durMs) ->
                voiceFile = f
                status = getString(R.string.status_voice_ready, fmt(durMs.toFloat()))
            }.onFailure {
                voiceFile = null
                status = getString(R.string.status_cannot_read_audio, it.message ?: "")
            }
            // Not `opts`, so nothing else asks the swapped pane to redraw for it. Cheap:
            // the pipeline stays warm, `applyVoice` just decodes the new file on the next
            // refresh -- no reload, which is what `reloads = false` says.
            if (opts.lipSync) previewOptionsChanged(reloads = false)
        }
    }

    /**
     * Record the driving voice with the phone's own microphone.
     *
     * The lip syncer needs a voice that is NOT the target's own audio, and until now the
     * only way to supply one was to already have the file. Recording one is the obvious
     * missing half, and it is the same shape as capturing a source face with the camera.
     *
     * ⚠ The result goes through [loadVoice] like any picked file rather than being assigned
     * to voiceFile directly. That is what keeps the has_audio check, the duration read, the
     * status line and the preview refresh in ONE place -- a recording that set voiceFile
     * itself would be a second path that silently skips all four.
     *
     * MPEG_4 + AAC because that is what AudioDecoder already reads: a raw PCM or AMR file
     * would decode, but through a different branch nobody measured.
     */
    private var recorder: android.media.MediaRecorder? = null

    private fun toggleVoiceRecording() {
        if (recordingVoice) { stopVoiceRecording(); return }
        if (checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) !=
            android.content.pm.PackageManager.PERMISSION_GRANTED) {
            askMic.launch(android.Manifest.permission.RECORD_AUDIO)
            return
        }
        val f = File(cacheDir, "voice_rec.m4a")
        val r = android.media.MediaRecorder(this)
        val ok = runCatching {
            r.setAudioSource(android.media.MediaRecorder.AudioSource.MIC)
            r.setOutputFormat(android.media.MediaRecorder.OutputFormat.MPEG_4)
            r.setAudioEncoder(android.media.MediaRecorder.AudioEncoder.AAC)
            r.setAudioSamplingRate(44100)
            r.setAudioChannels(1)
            r.setAudioEncodingBitRate(128_000)
            r.setOutputFile(f.absolutePath)
            r.prepare()
            r.start()
        }.isSuccess
        if (!ok) {
            runCatching { r.release() }
            status = getString(R.string.status_record_failed)
            return
        }
        recorder = r
        recordedTo = f
        recordingVoice = true
        status = getString(R.string.status_recording)
    }

    private fun stopVoiceRecording() {
        val r = recorder ?: return
        recorder = null
        recordingVoice = false
        // stop() THROWS when nothing was captured -- a recording ended within a few tens of
        // milliseconds of starting produces no frames and no valid file. Treated as "too
        // short" rather than an error, because that is what the user did.
        val ok = runCatching { r.stop() }.isSuccess
        runCatching { r.release() }
        val f = recordedTo
        if (ok && f != null && f.length() > 0) loadVoice(Uri.fromFile(f))
        else status = getString(R.string.status_record_too_short)
    }

    private var recordedTo: File? = null
    private val askMic = registerForActivityResult(
        ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) toggleVoiceRecording()
        else status = getString(R.string.status_mic_denied)
    }

    private fun clearVoice() {
        voiceFile = null
        voiceName = null
        if (opts.lipSync) previewOptionsChanged(reloads = false)
    }

    /**
     * Where a finished video lands until the user saves it or moves on.
     *
     * `getExternalFilesDir`, not `cacheDir`: a result the user has not saved yet must not
     * evaporate because the OS wanted space. The price is that NOTHING else ever cleans
     * this directory -- "Clear cache" does not touch it, only "Clear data" does -- so every
     * path that drops the reference has to delete the file too. See [discardOutput].
     */
    private fun outputDir(): File = getExternalFilesDir(null) ?: filesDir

    /**
     * Drop the finished video AND the file behind it.
     *
     * Setting `outputFile = null` on its own leaked one MP4 per run into
     * `Android/data/<pkg>/files`, where the user cannot reach it and clearing the cache does
     * not remove it -- reported from the field. A still target never had the bug: its result
     * is a Bitmap that only ever reaches disk through Save.
     *
     * Safe after a gallery save. `GallerySaver` COPIES into MediaStore, so what the user
     * kept is a different file and this does not touch it.
     */
    private fun discardOutput() {
        outputFile?.delete()
        outputFile = null; outputPartial = false; savedUri = null; savedPathLabel = null
    }

    /**
     * Delete finished videos that nothing can reach any more.
     *
     * `outputFile` is Activity state with no saved-instance backing, so process death --
     * and plain rotation, which recreates this Activity -- orphans whatever was on screen.
     * `onCreate` is therefore the one moment at which no output is reachable BY DEFINITION,
     * which is what makes it the only safe place to sweep.
     *
     * Scoped to `swapped_*.mp4` in this one directory. The API server writes its own
     * `api_out.mp4` to `cacheDir`, which the OS already manages, and `selftest.mp4` is a
     * debug artefact somebody may still want to pull over adb.
     */
    private fun sweepOrphanedOutputs() {
        val swept = outputDir().listFiles { f: File ->
            f.isFile && f.name.startsWith("swapped_") && f.name.endsWith(".mp4")
        }?.count { it.delete() } ?: 0
        if (swept > 0) android.util.Log.i("ffmain", "swept $swept orphaned output(s)")
    }

    /**
     * Drop the target — and ONLY the target.
     *
     * A finished video deliberately survives this. The trash icon sits on the TARGET pane
     * and says "Remove target"; taking an unsaved result with it means one mistap destroys
     * minutes of NPU time with no undo. The output pane and its Save/Share row are gated on
     * `outputFile`, not on the target, so they stay usable with nothing loaded.
     *
     * The file is still not leaked: it goes when a new target is picked, when the next run
     * starts, or in [sweepOrphanedOutputs] on the next launch. At most one can be waiting.
     */
    /**
     * Drop the source face.
     *
     * The warm pipeline is holding this face's EMBEDDING, so the preview has to be
     * invalidated for exactly the reason picking a different source does: otherwise the
     * swapped pane keeps showing a face the user has just removed.
     */
    /**
     * Why the pipeline could not be taken, naming whoever has it.
     *
     * ⚠ All three callers used to print "the remote API is using the NPU" no matter who
     * actually held it. PipeGuard has recorded the holder since it was written and nothing
     * asked. Reported from the field as Live blaming the API on the GPU backend -- where
     * the preview holds the pipe far longer, so it is the preview that loses the race.
     * The message also claimed an NPU on devices whose whole point is not having one.
     */
    private fun pipeBusyMessage(): String = getString(R.string.status_pipe_busy,
        getString(when (PipeGuard.holder) {
            "preview" -> R.string.pipe_holder_preview
            "swap" -> R.string.pipe_holder_swap
            "api", "api-stop" -> R.string.pipe_holder_api
            "live" -> R.string.pipe_holder_live
            else -> R.string.pipe_holder_other
        }))

    private fun clearSource() {
        sourceUri = null
        sourceThumb = null
        status = ""
        previewOptionsChanged()
    }

    private fun clearTarget() {
        previews.closeTarget()
        targetFile = null
        targetImage = null
        targetName = null
        durationMs = 0L
        trimStartMs = 0f; trimEndMs = 0f
        targetAspect = 16f / 9f
        originalFrame = null
        targetVersion++
        invalidatePreview()
        status = ""
    }

    /**
     * The video path, and only the video path.
     *
     * A still target used to have a branch in here, reached by the Swap button. It is gone
     * with the button: the swapped PANE is already that image -- full resolution, same
     * pipeline, same options, and gated on both the source and the frame in
     * [refreshSwapped] -- so a run could only spend a model reload to produce a second copy
     * of what was on screen. An unreachable branch that processes pixels is exactly the
     * kind of thing a content-gate audit has to keep re-proving, so it is not left behind.
     */
    /**
     * Start or stop the live feed.
     *
     * Live owns the pipeline for as long as it runs, through the same [PipeGuard] the API
     * and the screen contend for -- a camera pump and a preview refresh both calling
     * processFrame on one global is exactly what that guard exists to prevent.
     */
    /**
     * ⚠ Live must not survive the activity leaving the foreground. CameraX unbinds itself
     * -- it is lifecycle-bound -- but nothing else does: the pipeline would stay loaded and
     * PipeGuard would stay HELD, so the next run anywhere in the app reports the NPU busy
     * with no way to clear it short of killing the process.
     */
    override fun onStop() {
        super.onStop()
        if (liveRunning) stopLive()
    }

    private fun toggleLive() {
        if (liveRunning) { stopLive(); return }
        liveNote = null
        if (checkSelfPermission(android.Manifest.permission.CAMERA) !=
            android.content.pm.PackageManager.PERMISSION_GRANTED) {
            askCamera.launch(android.Manifest.permission.CAMERA)
            return
        }
        startLive()
    }

    private fun startLive() {
        val src = sourceUri ?: return
        lifecycleScope.launch {
            if (!PipeGuard.acquire("live", 5000)) {
                liveNote = pipeBusyMessage(); return@launch
            }
            // The preview holds a warm pipeline configured for the Swap screen. Live needs
            // its own configuration, so the preview's is dropped rather than mutated --
            // sharing it would leave the Swap screen warm for options Live had changed.
            previews.invalidate()
            previewWarm = false

            val base = SwapOptions.load(this@MainActivity)
            // The forced preset. Tracking ON is the whole reason this is watchable; the
            // enhancer and pixel boost are the two settings that most easily turn 25 fps
            // into single digits, so they are pinned unless the override says otherwise.
            val opts = if (liveUseMySettings) base else base.copy(
                faceEnhance = false, pixelBoost = 1, lipSync = false, trackPeriod = 4,
            )
            val ok = withContext(Dispatchers.Default) {
                val models = modelDir()
                val libDir = applicationInfo.nativeLibraryDir
                if (!NativePipe.init(libDir, libDir, models.absolutePath, opts)) return@withContext false
                NativePipe.setTrackPeriod(opts.trackPeriod)
                val bmp = decodeOriented(src) ?: return@withContext false
                val soft = bmp.copy(Bitmap.Config.ARGB_8888, false)
                val px = IntArray(soft.width * soft.height)
                soft.getPixels(px, 0, soft.width, 0, 0, soft.width, soft.height)
                NativePipe.setSource(NativePipe.argbToBgr(px, soft.width, soft.height),
                                     soft.width, soft.height)
            }
            if (!ok) {
                liveNote = "cannot start: ${NativePipe.lastError()}"
                NativePipe.release(); PipeGuard.release(); return@launch
            }
            liveRunning = true
            // THE GATE, on the live path. The source is already checked where it is picked
            // (see the source_image branch above), so what is left is the camera itself --
            // and a camera is the one input the user can change without touching the app.
            //
            // The threshold is set HERE rather than inside LiveEngine because the dev line
            // deletes ContentGate.kt: the engine takes a number and knows nothing about the
            // gate, so this single assignment is the whole of what dev has to remove.
            live.gateThreshold = ContentGate.THRESHOLD
            live.start(this@MainActivity, this@MainActivity) { shot ->
                // The analyzer thread hands the result straight to Compose state, which is
                // safe for snapshot state and avoids a per-frame main-thread post.
                if (shot.error != null) liveNote = shot.error
                // A refusal, or a check that could not run -- which is also a refusal: `ok`
                // is ALLOW alone, here as everywhere else. The engine has already stopped
                // its pump; this releases the pipeline and the camera, and says why.
                if (shot.gate != LiveEngine.Gate.None) {
                    liveNote = getString(
                        if (shot.gate == LiveEngine.Gate.Blocked) R.string.gate_blocked
                        else R.string.gate_error,
                        getString(R.string.gate_subject_this_frame))
                    runOnUiThread { stopLive() }
                }
                if (shot.bitmap != null) {
                    liveFrame = shot.bitmap
                    liveFaces = shot.faces
                    liveFps = shot.fps
                }
            }
        }
    }

    private fun stopLive() {
        // Idempotent: reached from the button, from leaving the tab, and from onStop, and
        // two of those can fire for one user action. Releasing the pipeline twice is a
        // crash of exactly the kind this method was written to fix.
        if (!liveRunning) return
        live.stop()
        liveRunning = false
        NativePipe.setTrackPeriod(0)
        NativePipe.release()
        PipeGuard.release()
        liveFrame = null; liveFps = 0.0; liveFaces = 0
    }

    private fun runSwap() {
        val src = sourceUri ?: return
        val tgt = targetFile ?: return
        // Defence in depth: the Swap button in SwapScreen is already disabled without one
        // when Lip Sync is on, but this is the actual place a no-op run would happen, so it
        // is checked again here rather than trusted from the UI alone.
        if (opts.lipSync && voiceFile == null) return
        // The preview holds a loaded pipeline and `g_pipe` is a single global, so the run
        // cannot start until it lets go.
        invalidatePreview()
        cancelRequested = false
        busy = true; progress = 0f; log = ""
        // Deletes the PREVIOUS run's file, not merely the reference to it.
        discardOutput()
        preview = null; framesDone = 0; framesTotal = 0; elapsedS = 0.0

        lifecycleScope.launch {
            // The run owns the pipeline for its whole length, minutes on a long clip. The
            // API server gets 503 for the duration, which is the honest answer.
            if (!PipeGuard.acquire("swap", 5000)) {
                status = pipeBusyMessage()
                busy = false
                return@launch
            }
            val result = withContext(Dispatchers.Default) {
                runCatching {
                    val models = modelDir()
                    val missing = listOf("yoloface", "fan2d", "arcface", opts.swapper)
                        .filterNot { ModelPaths.present(models, tier, it) }
                        .toMutableList()
                    // The gate is mandatory because it blocks. Either build satisfies it:
                    // fp32 (`nsfw_`) only finalizes on v79, so every other tier carries the
                    // quantised `nsfwq2_` -- and ncnn has neither distinction.
                    if (!ModelPaths.present(models, tier, "nsfw") &&
                        !ModelPaths.present(models, tier, "nsfwq2"))
                        missing += "nsfw"
                    if (missing.isNotEmpty())
                        error("cannot read ${missing.joinToString()} for tier $tier in " +
                              "${models.absolutePath} -- run work/device/install_app.ps1")

                    status = getString(R.string.status_loading_models)
                    val libDir = applicationInfo.nativeLibraryDir
                    val ok = NativePipe.init(libDir, libDir, models.absolutePath, opts)
                    noteTierRejection()
                    if (!ok) error("init: ${NativePipe.lastError()}")
                    appendLog("weight %.2f  blur %.2f  padding %s  boost %s%s"
                        .format(opts.weight, opts.maskBlur,
                                opts.maskPadding.joinToString("/"), opts.pixelBoostLabel,
                                if (opts.largestOnly) "  largest face only" else ""))

                    status = getString(R.string.status_reading_source)
                    val bmp = decodeOriented(src) ?: error("cannot decode source image")

                    // The content gate, BEFORE anything is processed or previewed. It
                    // blocks, as upstream does, so a refusal ends the run here -- there is
                    // no partial output and nothing reaches the preview surface.
                    status = getString(R.string.status_content_check)
                    if (NativePipe.contentGateIsQuantised())
                        appendLog("content gate: W8A16 build, biased " +
                                  "+${ContentGate.QUANTISED_BIAS} toward refusing")
                    ContentGate.checkImage(bmp).let {
                        appendLog("source content score %+.3f".format(it.score))
                        if (!it.ok) throw ContentGate.Refused(ContentGate.message(this@MainActivity,
                                                                R.string.gate_subject_source_image, it))
                    }
                    // The target, sampled across the clip.
                    ContentGate.checkVideo(tgt).let {
                        // `detail` is an ARGUMENT, never interpolated into the format
                        // string: it reads "0/11 flagged (0.0%)", and that trailing `%)`
                        // is parsed as a conversion -- UnknownFormatConversionException,
                        // which killed the whole swap after the gate had already passed.
                        appendLog("target content: %s, worst %+.3f".format(it.detail, it.score))
                        if (!it.ok)
                            throw ContentGate.Refused(
                                ContentGate.message(this@MainActivity,
                                                    R.string.gate_subject_target_video, it))
                    }

                    val soft = bmp.copy(Bitmap.Config.ARGB_8888, false)
                    val px = IntArray(soft.width * soft.height)
                    soft.getPixels(px, 0, soft.width, 0, 0, soft.width, soft.height)
                    if (!NativePipe.setSource(NativePipe.argbToBgr(px, soft.width, soft.height),
                                              soft.width, soft.height))
                        error("source: ${NativePipe.lastError()}")
                    appendLog("source ready (${soft.width}x${soft.height})")

                    val t0 = System.currentTimeMillis()

                    val out = File(outputDir(),
                        "swapped_${System.currentTimeMillis()}.mp4")
                    status = getString(R.string.status_swapping)
                    var lastPreview = 0L

                    VideoSwapper(
                        outputFps = opts.outputFps,
                        trackPeriod = opts.trackPeriod,
                        lipSync = opts.lipSync,
                        voicePath = voiceFile?.absolutePath,
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
                status = getString(R.string.status_done, it.length() / 1024)
                // ⚠ RE-WARM THE PREVIEW. `runSwap` opens with invalidatePreview(), which
                // clears previewWarm -- and `onTrimChanged` only redraws the swapped pane
                // `if (previewWarm)`, on the reasoning that the first model load costs
                // seconds and has to be asked for. True before the first run; false after
                // one, where the models are demonstrably loaded and the user has already
                // paid. The effect was a swapped pane frozen on its last frame while
                // seeking moved the original beside it, which reads as the swap being
                // wrong rather than the pane being stale.
                //
                // A forced refresh here rebuilds the engine once and puts previewWarm back,
                // so every later seek takes the normal warm path.
                refreshSwapped(force = true)
            }.onFailure {
                // A refusal is already a finished sentence aimed at the user, and it is not
                // a fault: prefixing it with "Failed:" and dumping a stack trace would
                // present a working safety check as a crash.
                if (it.message == "cancelled") {
                    // Asked for, not gone wrong: no "Failed:", no stack trace.
                    status = getString(R.string.status_cancelled)
                } else if (it is ContentGate.Refused) {
                    // The gate's own finished sentence, already localized. NOT an
                    // error: nothing malfunctioned, so this offers no bug report.
                    status = it.message ?: getString(R.string.gate_blocked_generic)
                } else {
                    failStatus(getString(R.string.status_failed, it.message ?: ""))
                    appendLog(it.stackTraceToString().take(700))
                }
            }
            NativePipe.release()
            PipeGuard.release()
            busy = false
        }
    }

    /** The finished video, into the shared Movies collection. */
    private fun saveToGallery(file: File) {
        lifecycleScope.launch {
            val r = withContext(Dispatchers.IO) { GallerySaver.save(this@MainActivity, file) }
            r.onSuccess {
                savedUri = it
                savedPathLabel = "Movies/FaceFusion/" + file.name
                status = getString(R.string.status_saved_movies)
                toast(getString(R.string.toast_saved_to, savedPathLabel!!))
            }.onFailure {
                status = getString(R.string.status_save_failed, it.message ?: "")
            }
        }
    }

    /**
     * The swapped STILL, which is the pane rather than a file.
     *
     * There is no run behind a still and so no output file to copy: the bitmap on screen is
     * the result, produced by the same pipeline at the image's own resolution, and it is
     * what goes to Pictures. Both the source and this exact frame were gated in
     * [refreshSwapped] before it was ever drawn, so saving what is displayed cannot save
     * anything the gate has not already passed.
     */
    private fun saveSwappedStill() {
        val bmp = swappedFrame ?: return
        lifecycleScope.launch {
            val name = "facefusion_%d.png".format(System.currentTimeMillis())
            val r = withContext(Dispatchers.IO) {
                GallerySaver.saveImage(this@MainActivity, bmp, name)
            }
            r.onSuccess {
                savedUri = it
                savedPathLabel = "Pictures/FaceFusion/" + name
                status = getString(R.string.status_saved_pictures)
                toast(getString(R.string.toast_saved_to, savedPathLabel!!))
            }.onFailure {
                status = getString(R.string.status_save_failed, it.message ?: "")
            }
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
    /**
     * Save the frame the SWAPPED pane is showing, as an image.
     *
     * Distinct from [saveFrameAt], which reads a frame back out of the finished video and
     * therefore needs a run to have happened first. This one saves what is already on
     * screen -- so a single still can be pulled out of a clip without swapping the clip,
     * which on the CPU backend is the difference between seconds and many minutes.
     *
     * The bitmap is the pane's own, at the resolution the pipeline produced it, not the
     * scaled-down thing being displayed.
     */
    private fun savePreviewFrame() {
        val bmp = swappedFrame ?: preview ?: return
        lifecycleScope.launch {
            val stem = targetName?.substringBeforeLast('.')?.take(40) ?: "facefusion"
            // The timestamp only means something for a video; a still has exactly one frame
            // and "_000000ms" on it is noise.
            val name = if (durationMs > 0)
                           stem + "_swapped_%06dms.png".format(previewAtMs.toInt())
                       else stem + "_swapped.png"
            val r = withContext(Dispatchers.IO) {
                GallerySaver.saveImage(this@MainActivity, bmp, name)
            }
            r.onSuccess {
                status = getString(R.string.status_frame_saved)
                toast(getString(R.string.toast_saved_to, "Pictures/FaceFusion/" + name))
            }
             .onFailure {
                 status = getString(R.string.status_save_frame_failed, it.message ?: "")
             }
        }
    }

    private fun saveFrameAt(positionMs: Int) {
        val file = outputFile ?: return
        // Hoisted out of the mapCatching that used to build it, so the toast can name the
        // file it just wrote rather than the directory it went into.
        val name = file.nameWithoutExtension + "_%06dms.png".format(positionMs)
        lifecycleScope.launch {
            val r = withContext(Dispatchers.IO) {
                runCatching {
                    val mmr = MediaMetadataRetriever().apply { setDataSource(file.absolutePath) }
                    val bmp = mmr.getFrameAtTime(positionMs * 1000L,
                                                 MediaMetadataRetriever.OPTION_CLOSEST)
                    mmr.release()
                    bmp ?: error("no frame at that position")
                }.mapCatching { bmp ->
                    GallerySaver.saveImage(this@MainActivity, bmp, name).getOrThrow()
                }
            }
            r.onSuccess {
                status = getString(R.string.status_frame_saved)
                toast(getString(R.string.toast_saved_to, "Pictures/FaceFusion/" + name))
            }
             .onFailure {
                 status = getString(R.string.status_save_frame_failed, it.message ?: "")
             }
        }
    }

    private fun shareResult() {
        val uri = savedUri ?: run {
            status = getString(R.string.status_save_first)
            return
        }
        // The mime has to match what was actually saved, or the chooser offers apps that
        // cannot open it -- an image result went out as video/mp4 before image targets
        // existed, and would have silently kept doing so.
        val image = targetImage != null
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
            type = if (image) "image/png" else "video/mp4"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }, getString(if (image) R.string.share_swapped_image
                     else R.string.share_swapped_video)))
    }

    /**
     * Headless self-test, so the in-process DSP path can be checked over adb:
     *   adb shell am start -n com.facefusion.mobile/.MainActivity --es selftest 1
     *   adb logcat -s ffselftest
     */
    private fun selfTest(voiceName: String? = null) {
        val tag = "ffselftest"
        lifecycleScope.launch(Dispatchers.Default) {
            fun say(s: String) = android.util.Log.i(tag, s)
            try {
                val models = modelDir()
                models.listFiles()?.forEach { say("  ${it.name} ${it.length()} readable=${it.canRead()}") }
                val libDir = applicationInfo.nativeLibraryDir
                // Which RUNTIME, first and by name. The CLI printed a hardcoded "NPU" over
                // an ncnn run once and the label cost an afternoon three weeks later
                // (docs/backends.md); the same mistake was sitting in this function as a
                // hardcoded "QNN init OK".
                val backend = ModelPaths.backend(this@MainActivity)
                say("backend: $backend" +
                    (if (forcedBackend.isNotEmpty()) " (forced)" else " (auto)"))
                // WHICH UNIT, not just which runtime. ncnn on the GPU and ncnn on the CPU
                // differ by ~1.8x and by the thermal behaviour that decided the backend, so
                // a run that does not say which one it took cannot be compared with one
                // that does -- this is the CLI's hardcoded "NPU" label again, one level in.
                if (backend == ModelPaths.NCNN_TIER) {
                    // Asked directly, not read off `deviceUi`: this path returns from
                    // onCreate before probeDevice() ever runs, so that field is still its
                    // default and would report "no GPU" on every device.
                    val gpu = NativePipe.probeDeviceInfo(libDir, libDir).contains("gpu=1")
                    say("vulkan: " + (if (gpu) "yes, GPU preferred" else
                                      "NO, everything on the CPU"))
                }
                say("tier: $tier")
                // The fp16 canary is a QNN measurement and brings QNN up to take it. On
                // ncnn there is nothing to ask, and asking anyway would start the runtime
                // this run is supposed to be avoiding.
                if (backend != ModelPaths.NCNN_TIER)
                    say("fp16: ${NativePipe.probeFp16(libDir, libDir, canaryDir().absolutePath)}")
                val initOk = NativePipe.init(libDir, libDir, models.absolutePath, opts)
                noteTierRejection()
                if (!initOk) {
                    say("INIT FAILED: ${NativePipe.lastError()}"); return@launch
                }
                say("$backend init OK")
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
                val voiceFile = voiceName?.let(::asset)?.takeIf { it.canRead() }
                if (voiceName != null && voiceFile == null) say("voice: cannot read $voiceName")
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
                say("lip syncer on device: ${NativePipe.hasLipSyncer()}, asked: ${opts.lipSync}" +
                    (voiceFile?.let { ", voice: ${it.name}" } ?: ""))
                VideoSwapper(onProgress = { d, t -> if (d % 25 == 0) say("frame $d/$t") },
                             onLog = { say(it) },
                             trackPeriod = opts.trackPeriod,
                             lipSync = opts.lipSync,
                             voicePath = voiceFile?.absolutePath)
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
