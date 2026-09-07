package com.facefusion.mobile.ui

import android.graphics.Bitmap
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.Image
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.foundation.clickable
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.facefusion.mobile.FaceDetectorCard
import com.facefusion.mobile.FaceMaskerCard
import com.facefusion.mobile.FaceSwapperCard
import com.facefusion.mobile.BatchItem
import com.facefusion.mobile.BatchState
import com.facefusion.mobile.ModelDownload
import com.facefusion.mobile.OptionSegments
import com.facefusion.mobile.OptionSlider
import com.facefusion.mobile.OptionSteps
import com.facefusion.mobile.SwapOptions
import java.io.File
import kotlin.math.roundToInt
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import com.facefusion.mobile.R
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.PlayArrow

/** Everything the two preview panes need to draw themselves. */
data class PreviewUi(
    val original: Bitmap? = null,
    val swapped: Bitmap? = null,
    val timeLabel: String = "",
    /** Whether the pipeline is loaded. Until it is, the swapped pane cannot draw anything. */
    val warm: Boolean = false,
    val busy: Boolean = false,
    /** "No face detected", or an error. Shown in place of the image. */
    val note: String? = null,
    /**
     * What the detector found in [original], five floats per face, in the ORIGINAL's own
     * pixel coordinates. Null when the overlay is off or nothing has been asked yet.
     */
    val faceBoxes: FloatArray? = null,
    /** The face picked as the reference, if any -- drawn solid while the rest dim. */
    val referenceBox: FloatArray? = null,
)

/**
 * Which trim handle the user is dragging.
 *
 * The previews follow the handle under the finger. Before this, both panes always showed
 * the START frame, so dragging the end handle appeared to do nothing at all -- the
 * "slider doesn't move the preview" report was this, not a stale pane.
 */
enum class TrimEdge { Start, End }

/** Progress of an actual swap run. */
data class RunUi(
    val busy: Boolean = false,
    val preparing: Boolean = false,
    val progress: Float = 0f,
    val framesDone: Int = 0,
    val framesTotal: Int = 0,
    val elapsedS: Double = 0.0,
)

@Composable
fun SwapScreen(
    sourceThumb: Bitmap?,
    hasSource: Boolean,
    hasTarget: Boolean,
    /**
     * The target is a STILL.
     *
     * There is nothing to run: the swapped pane already holds the finished image, at full
     * resolution and through the same pipeline a run would use. So the Swap button is not
     * drawn at all -- pressing it would spend seconds reloading the models to produce a
     * second copy of the picture already on screen.
     */
    imageTarget: Boolean,
    durationMs: Long,
    trimStartMs: Float,
    trimEndMs: Float,
    onTrimChange: (Float, Float, TrimEdge) -> Unit,
    /** width / height of the target. Below 1 the panes go side by side. */
    targetAspect: Float,
    /** The target video's own rate, and the cap on what can be chosen. */
    inputFps: Int,
    /** The target's own pixel size, upright. The cap on what output sizes are offered. */
    targetW: Int,
    targetH: Int,
    fmt: (Float) -> String,
    preview: PreviewUi,
    run: RunUi,
    status: String,
    /**
     * Whether [status] describes a FAILURE, decided by the Activity rather than re-derived
     * here.
     *
     * This used to be `status.startsWith("Failed")` -- a test on a string that is shown to
     * the user. Translating the status would have silently removed the bug-report button in
     * every language but English, which is precisely the language whose users are least
     * likely to need it.
     */
    statusIsError: Boolean,
    log: String,
    opts: SwapOptions,
    onOptsChange: (SwapOptions) -> Unit,
    hasInswapper: Boolean,
    hasEnhancer: Boolean,
    hasLipSyncer: Boolean,
    /**
     * A processor whose model is not on the device was tapped.
     *
     * Two arguments: the chip's LABEL, which is localized and only ever shown, and the
     * MODEL name the downloader knows it by ("gpen", "edtalk"). ⚠ It used to pass the
     * label alone, which left the prompt's Continue with nothing to ask for but "the
     * missing set" -- and the missing set excludes exactly these two models by name.
     */
    onRequestModel: (label: String, model: String) -> Unit,
    /** Whether the detector's boxes are drawn over the ORIGINAL pane. */
    showFaceBoxes: Boolean,
    /** Turn the face overlay on or off. Detection runs only while it is on. */
    onToggleFaceBoxes: () -> Unit,
    /**
     * A face in the ORIGINAL pane was tapped, in that image's own pixel coordinates:
     * upstream's `face_selector_mode = reference`. Tapping the chosen one again clears it.
     */
    onPickFace: (Float, Float) -> Unit,
    /**
     * The run queue, item one being the VISIBLE target -- roadmap 14.
     *
     * Size 1 is the ordinary single-clip screen and draws no queue at all: one source,
     * many targets is a mode you enter by picking several files, not by finding a switch.
     */
    batch: List<BatchItem>,
    /** Drop a queued clip. Only offered while it is still waiting. */
    onRemoveFromBatch: (Int) -> Unit,
    /** Add more clips to the queue, leaving the visible target alone. */
    onAddToBatch: () -> Unit,
    /** Show a finished batch clip in the output pane, by its index in [batch]. */
    onOpenBatchOutput: (Int) -> Unit,
    /**
     * The output on screen is already in the gallery, put there by the batch's auto-save.
     *
     * Hides the Save button rather than disabling it: a greyed control still asks the user
     * to work out why, and the answer -- "because it is already saved" -- is better said by
     * the label that replaces it.
     */
    outputAutoSaved: Boolean,
    /** Copy every finished batch clip straight to the gallery. */
    batchAutoSave: Boolean,
    onBatchAutoSave: (Boolean) -> Unit,
    openCard: String,
    onToggleCard: (String) -> Unit,
    /** There is something to save: a finished video, or a swapped still on the pane. */
    hasOutput: Boolean,
    /** The finished video, for the output pane. Null when the target was a still. */
    outputFile: File?,
    /** True when the run was cancelled, so the output is only as long as it got. */
    outputPartial: Boolean,
    onSaveFrame: (Int) -> Unit,
    saved: Boolean,
    savedPath: String?,
    onPickSource: () -> Unit,
    onPickTarget: () -> Unit,
    /** The lip syncer's driving audio -- see [onPickVoice]'s doc, and `VideoSwapper.voicePath`. */
    hasVoice: Boolean,
    voiceName: String?,
    /** The loaded voice's full length, ms. 0 until a file is loaded. */
    voiceDurationMs: Long,
    /**
     * The part of the voice that drives the lips, ms -- the audio equivalent of
     * [trimStartMs]/[trimEndMs] on the target. Only this range is decoded for the mouth
     * and copied into the output's audio track.
     */
    voiceTrimStartMs: Float,
    voiceTrimEndMs: Float,
    onVoiceTrimChange: (Float, Float) -> Unit,
    /** Playback of the loaded voice: the playhead position and whether it is running. */
    voicePosMs: Float,
    voicePlaying: Boolean,
    onVoicePlayPause: () -> Unit,
    onVoiceSeek: (Float) -> Unit,
    /**
     * Pick the file that DRIVES the mouth -- deliberately not the target. Only shown once
     * Lip Sync is on, because syncing a clip to the audio it already has has nothing to
     * fix: this is upstream's actual use for the feature (dubbing a different voice onto
     * the target), not a way to verify the target's own performance.
     */
    onPickVoice: () -> Unit,
    onClearVoice: () -> Unit,
    /** Microphone capture of the driving voice, in-app. */
    recordingVoice: Boolean,
    onToggleRecordVoice: () -> Unit,
    /**
     * Save the frame currently shown in the SWAPPED pane, as an image.
     *
     * Distinct from [onSaveFrame], which takes a position and reads it out of the
     * FINISHED video. This one needs no argument because the frame is already on
     * screen, and it works before any run has happened.
     */
    onSavePreviewFrame: () -> Unit,
    onClearSource: () -> Unit,
    /** Shoot the source face with the camera. Stills only -- a source is an identity. */
    onCaptureSource: () -> Unit,
    /** Take a still / record a clip with the system camera, as the target. */
    onCapturePhoto: () -> Unit,
    onCaptureVideo: () -> Unit,
    onClearTarget: () -> Unit,
    /** Delete the rendered file. Confirms first -- see the dialog at the end of this file. */
    onDeleteOutput: () -> Unit,
    onSwap: () -> Unit,
    onCancel: () -> Unit,
    modelsMissing: Boolean,
    onDownload: () -> Unit,
    onShareLog: () -> Unit,
    onSave: () -> Unit,
    onShare: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val idle = !run.busy && !run.preparing

    // The result pane sizes itself from the TARGET's own dimensions (60%, see below), so
    // it needs the screen width to clamp against; `screenH` still feeds the fallback
    // height and the video output pane.
    val screenH = LocalConfiguration.current.screenHeightDp
    val screenW = LocalConfiguration.current.screenWidthDp
    // Side-by-side panes are half as wide, so they can afford to be taller: the pair costs
    // ONE pane's height instead of two, which is the whole reason portrait gets this layout.
    //
    // ⚠ Budgets are TIGHT on purpose. Every dp the panes take is a dp the Swap button is
    // pushed below the fold; the report "the button vanished after picking a target" was
    // this arithmetic, not a rendering bug. The processor card is collapsed by default (see
    // processorsExpanded) and the trim card too, so the panes are the only large blocks on
    // a fresh target -- and even so their combined height has to leave room for the button.
    val paneHeight = if (targetAspect < 1f) (screenH - 480).coerceIn(140, 300).dp
                     else ((screenH - 560) / 2).coerceIn(100, 240).dp

    // ONE instance for every pane, which is what makes them zoom together (item 4).
    val zoom = remember { ZoomState() }

    // Which processor's settings sheet is open: "swapper", "enhancer", "lipsync", or null.
    //
    // Local and saveable rather than hoisted into MainActivity like [openCard], because it
    // is transient view state with no bearing on a run -- MainActivity holds what the swap
    // needs to know, and which sheet is showing is not that. rememberSaveable so a rotation
    // does not close it mid-adjustment.
    var settingsFor by rememberSaveable { mutableStateOf<String?>(null) }
    var confirmDeleteOutput by rememberSaveable { mutableStateOf(false) }
    // The processor stages collapse under their title. Default CLOSED: three chips plus a
    // caption cost ~160 dp standing, and a fresh target already carries the workbench row
    // and the swapped pane -- the Swap button used to be pushed below the fold by exactly
    // that much. The header's "n / 3" readout still says how many stages are armed, so a
    // folded card is not a silent one. The state is saved so a rotation does not silently
    // re-open a row the user just folded away.
    var processorsExpanded by rememberSaveable { mutableStateOf(false) }
    // Output settings live on ONE foldable card, default CLOSED: "Clip" (the trim) is the
    // first item, followed by output size and frame rate -- per-run tuning that should not
    // push the Swap button off the first screen.
    var trimExpanded by rememberSaveable { mutableStateOf(false) }
    // The voice playback and clip/trim controls fold under their own card, below the
    // processors, same default CLOSED: the voice only matters once Lip Sync is on and a
    // clip is loaded, and a standing playback row pushed the inputs further down.
    var voiceSettingsExpanded by rememberSaveable { mutableStateOf(false) }
    // The log panel folds under its caption. Default CLOSED -- it is a debug readout,
    // and a standing 170 dp panel below the buttons made the page longer than it needed
    // to be on every screen, not just while something was running.
    var logExpanded by rememberSaveable { mutableStateOf(false) }
    // The batch queue lives in a foldable card, default CLOSED, placed just above Output
    // settings. Batch is a mode you enter deliberately, not something every fresh target
    // needs standing open.
    var batchMenuExpanded by rememberSaveable { mutableStateOf(false) }
    // The finished output (video player + save/share/delete) folds under its own card,
    // default CLOSED, above the log.
    var outputResultExpanded by rememberSaveable { mutableStateOf(false) }
    Column(
        modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 18.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        // ---------------------------------------------------------------- processors
        //
        // Above Advanced, and named the way FaceFusion names them. The enhancer is a
        // PROCESSOR -- a stage that either runs or does not -- and burying its on/off
        // switch three taps deep inside "Advanced", next to blend weights and detector
        // thresholds, filed a yes/no question with the dials. FaceFusion puts the same two
        // side by side at the top; so does this now.
        //
        // face_swapper is drawn selected and is not clickable: this app IS the swapper, and
        // a control that cannot be turned off should still be visible, because the row is
        // there to say WHICH stages will run.
        run {
            // The stages live on their own card -- see SectionCard -- so the chips read as
            // one thing that belongs together rather than a row adrift in the scroll. The
            // trailing readout says how many of the three stages are armed, which the
            // chips themselves only imply.
            SectionCard(
                stringResource(R.string.swap_processors),
                trailing = {
                    Text(
                        "${listOf(true, opts.faceEnhance, opts.lipSync).count { it }} / 3",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                },
                collapsible = true,
                expanded = processorsExpanded,
                onToggle = { processorsExpanded = !processorsExpanded },
            ) {
            // Styled after upstream FaceFusion's own web UI, which is what these controls
            // are a port of: ON is a solid accent chip with a contrasting label and a
            // filled disc holding a check; OFF is a plain surface chip with a flat grey
            // disc and no check. The accent is the theme's monochrome primary (dark on
            // light scheme, light on dark), defined in ui/Theme.kt.
            //
            // Deliberately NOT Material3's default FilterChip look, which says "selected"
            // with a faint tonal wash and a bare tick. Upstream's row is the thing a user
            // arriving from the desktop app already knows how to read.
            @Composable
            fun ProcessorChip(
                name: String,
                /** What the downloader calls this stage's model; "" for one always present. */
                model: String,
                installed: Boolean,
                on: Boolean,
                available: Boolean,
                onToggle: () -> Unit,
                // The gear, drawn INSIDE the chip at its trailing edge. Null for a chip
                // with nothing to configure; also hidden while the model is missing, where
                // the chip's job is to offer the download and settings would be settings
                // for something that cannot run.
                onSettings: (() -> Unit)? = null,
            ) {
                val active = installed && on && available
                val clickable = idle && (!installed || available)
                Surface(
                    onClick = { if (installed) onToggle() else onRequestModel(name, model) },
                    enabled = clickable,
                    shape = RoundedCornerShape(8.dp),
                    // The chip's fill matches the other controls (cards, buttons) -- the
                    // theme's surface -- not the page background. Selection is carried by
                    // the border (primary when a stage is ON, outlineVariant when off) and
                    // the check disc, never by a fill.
                    color = MaterialTheme.colorScheme.surface,
                    border = BorderStroke(
                        // A single hairline in both states: the border is a separator, not
                        // a selection bar -- selection reads from the primary colour, the
                        // check disc and the bold label.
                        1.dp,
                        if (active) MaterialTheme.colorScheme.primary.copy(alpha = 0.69f)
                        else MaterialTheme.colorScheme.outlineVariant,
                    ),
                ) {
                    Row(
                        Modifier.padding(horizontal = 12.dp, vertical = 9.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Box(
                            Modifier
                                .size(18.dp)
                                .clip(CircleShape)
                                .background(
                                    when {
                                        // Filled with the accent (text colour) now that the
                                        // chip itself is page-coloured.
                                        active -> MaterialTheme.colorScheme.primary.copy(alpha = 0.69f)
                                        // A model that is not on the device gets a hollow
                                        // disc, so "off" and "not installed" are not the
                                        // same picture. Upstream has no such state.
                                        !installed -> Color.Transparent
                                        else -> MaterialTheme.colorScheme.outlineVariant
                                    }
                                )
                                .then(
                                    if (!installed)
                                        Modifier.border(1.dp, MaterialTheme.colorScheme.outline,
                                                        CircleShape)
                                    else Modifier
                                ),
                            contentAlignment = Alignment.Center,
                        ) {
                            if (active) {
                                Icon(Icons.Default.Check, null, Modifier.size(12.dp),
                                     tint = MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.69f))
                            } else if (!installed) {
                                Icon(Icons.Default.Add, null, Modifier.size(12.dp),
                                     tint = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }
                        Text(
                            name,
                            style = MaterialTheme.typography.labelLarge,
                            color = when {
                                active -> MaterialTheme.colorScheme.onBackground
                                !installed -> MaterialTheme.colorScheme.onSurfaceVariant
                                else -> MaterialTheme.colorScheme.onSurface
                            },
                            fontWeight = if (active) FontWeight.SemiBold
                                        else FontWeight.Normal,
                            // ⚠ weight(fill = false) is what keeps every gear THE SAME
                            // SIZE. Row does not wrap, it SQUEEZES, and with two chips
                            // across a phone the squeeze landed on whichever child had no
                            // weight -- the icon. So face_swapper and face_enhancer, which
                            // share a row, drew a visibly smaller gear than lip_syncer,
                            // which has its row to itself. A weighted child is measured
                            // with what is LEFT after the unweighted ones, so the label now
                            // absorbs the shortfall (it ellipsises) and the gear never
                            // changes size. fill = false so a short label still does not
                            // stretch the chip.
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.weight(1f, fill = false),
                        )
                        // Its own clickable inside the chip's, which Compose resolves to the
                        // innermost -- so the gear opens settings and does NOT also toggle
                        // the stage underneath it. 22 dp of touch target inside a 36 dp
                        // chip is below the 48 dp guideline, but a chip that grew to hold a
                        // 48 dp box would no longer fit two across a phone, which is the
                        // layout constraint this row is already built around.
                        if (installed && onSettings != null) {
                            Icon(
                                Icons.Default.Settings,
                                stringResource(R.string.swap_proc_settings, name),
                                Modifier
                                    .size(18.dp)
                                    .clip(CircleShape)
                                    .clickable(enabled = idle) { onSettings() },
                                tint = if (active) MaterialTheme.colorScheme.onBackground
                                       else MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }

            // Two rows of two, upstream's shape. Three chips do not fit across a phone and
            // Row does not wrap -- it SQUEEZES, so labels lose their shape rather than
            // moving down, and these are upstream's identifiers.
            //
            // 8 dp both ways here, unlike the Material chips this replaces: a Surface has
            // no enforced 48 dp interactive box padding it out, so the spacing asked for
            // is the spacing seen.
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    // face_swapper is always on and cannot be turned off: this app IS the
                    // swapper. It still gets a chip, because the row exists to say WHICH
                    // stages run.
                    ProcessorChip(
                        name = stringResource(R.string.swap_proc_swapper),
                        // Always installed -- missing() makes it a REQUIRED model, so the
                        // app never reaches this row without it.
                        model = "",
                        installed = true, on = true, available = true, onToggle = {},
                        // Opens with the Face Swapper card already expanded, so the
                        // weight slider -- the knob most runs actually touch -- is there
                        // on arrival rather than one tap further in. Only when nothing
                        // else is open, so a card the user deliberately left open on a
                        // previous visit is respected.
                        onSettings = {
                            settingsFor = "swapper"
                            if (openCard.isEmpty()) onToggleCard("swapper")
                        },
                    )
                    ProcessorChip(
                        name = stringResource(R.string.swap_proc_enhancer),
                        model = "gpen",
                        installed = hasEnhancer,
                        on = opts.faceEnhance,
                        available = true,
                        onToggle = { onOptsChange(opts.copy(faceEnhance = !opts.faceEnhance)) },
                        onSettings = { settingsFor = "enhancer" },
                    )
                }
                // ⚠ `available` is false only once a PHOTO is picked. `durationMs > 0`
                // alone was false on an empty screen, so the chip greyed out the moment the
                // app opened and looked broken beside face_enhancer, which needs no target.
                // There is nothing to say no about until there is a target.
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    ProcessorChip(
                        name = stringResource(R.string.swap_proc_lip_syncer),
                        // edtalk, matching hasLipSyncer -- wav2lip is not offered because
                        // ffpipe no longer opens it.
                        model = "edtalk",
                        installed = hasLipSyncer,
                        on = opts.lipSync,
                        available = !hasTarget || durationMs > 0,
                        onToggle = { onOptsChange(opts.copy(lipSync = !opts.lipSync)) },
                        onSettings = { settingsFor = "lipsync" },
                    )
                }
                }
            }
        }

        // The processors' knobs used to sit inline here, each under the chip that turns it
        // on. They are behind that chip's own GEAR now: with three processors the inline
        // form pushed the source and target panes off the first screen whenever two stages
        // were enabled, and the panes are the primary path. They did not move far: one tap,
        // on the chip they already belong to, instead of a scroll down to Advanced.
        //
        // Each still reads and writes the SAME `opts` field it always did, so nothing about
        // the native side changed -- only where the control is drawn.

        // Only while Lip Sync is ON. It is a
        // REQUIRED input, not a tuning knob, so it sits in the workbench row between
        // source and target rather than in Advanced: the Swap button stays disabled
        // without one (see its `enabled` below), because syncing a clip to the audio it
        // already has has nothing to fix -- upstream's lip syncer exists to dub a
        // DIFFERENT voice on, and running it on the target's own track can only cost
        // face quality with no corrective benefit.

// ---------------------------------------------------------------- voice: listen, trim
        //
        // A voice is invisible, so the only way to check what was picked is to hear it.
        // Play previews exactly the trimmed selection -- it starts at the trim start and
        // stops at the trim end -- while the seekbar can scrub anywhere in the file. The
        // range slider below chooses the part that actually DRIVES the lips, and it is the
        // same two-handle control the video gets, because it is the same decision: keep
        // only the part that matters.
        if (opts.lipSync && hasVoice && voiceDurationMs > 0) {
            SectionCard(
                stringResource(R.string.swap_voice_settings),
                collapsible = true,
                expanded = voiceSettingsExpanded,
                onToggle = { voiceSettingsExpanded = !voiceSettingsExpanded },
            ) {
                // The playback and trim sliders read too high-contrast against the theme,
                // so every part (thumb, active and inactive track) has its opacity cut by
                // 31%: 69% alpha keeps the same hue with a much softer contrast -- the
                // same treatment the output and trim sliders already use.
                val voiceSliderColors = SliderDefaults.colors(
                    thumbColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.69f),
                    activeTrackColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.69f),
                    inactiveTrackColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.69f),
                )
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    IconButton(onVoicePlayPause, enabled = idle, modifier = Modifier.size(36.dp)) {
                        if (voicePlaying) {
                            // Two bars, drawn rather than an icon: the icons artifact this app
                            // carries (material3's transitive icons-core) has PlayArrow but no
                            // Pause, and extended-icons is a heavy addition for one glyph.
                            val pauseTint = MaterialTheme.colorScheme.onSurfaceVariant
                            Canvas(Modifier.size(16.dp)) {
                                val bar = 4.dp.toPx()
                                val gap = 3.dp.toPx()
                                val top = 0.dp.toPx()
                                val bottom = size.height
                                drawRoundRect(
                                    color = pauseTint,
                                    topLeft = Offset(0f, top),
                                    size = Size(bar, bottom - top),
                                    cornerRadius = CornerRadius(1.dp.toPx()),
                                )
                                drawRoundRect(
                                    color = pauseTint,
                                    topLeft = Offset(bar + gap, top),
                                    size = Size(bar, bottom - top),
                                    cornerRadius = CornerRadius(1.dp.toPx()),
                                )
                            }
                        } else {
                            Icon(Icons.Default.PlayArrow,
                                 stringResource(R.string.swap_voice_play),
                                 Modifier.size(20.dp),
                                 tint = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                    Slider(
                        value = voicePosMs.coerceIn(0f, voiceDurationMs.toFloat()),
                        onValueChange = onVoiceSeek,
                        valueRange = 0f..voiceDurationMs.toFloat(),
                        enabled = idle,
                        modifier = Modifier.weight(1f),
                        colors = voiceSliderColors,
                    )
                    Text("${fmt(voicePosMs)} / ${fmt(voiceDurationMs.toFloat())}",
                         style = MaterialTheme.typography.bodySmall,
                         fontFamily = FontFamily.Monospace,
                         modifier = Modifier.padding(start = 8.dp))
                }
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Caption(stringResource(R.string.swap_voice_trim), Modifier.weight(1f))
                    Text("${fmt(voiceTrimStartMs)} – ${fmt(voiceTrimEndMs)}",
                         style = MaterialTheme.typography.bodySmall,
                         fontFamily = FontFamily.Monospace)
                }
                RangeSlider(
                    value = voiceTrimStartMs..voiceTrimEndMs,
                    onValueChange = { r ->
                        // The same minimum span as the video trim, for the same reason: the
                        // mouth needs a mel window, the encoder needs a frame.
                        onVoiceTrimChange(r.start, maxOf(r.endInclusive, r.start + 333f))
                    },
                    valueRange = 0f..voiceDurationMs.toFloat(),
                    enabled = idle,
                    colors = voiceSliderColors,
                )
                Text(stringResource(R.string.swap_voice_trim_hint),
                     style = MaterialTheme.typography.bodySmall,
                     color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }

        // ---------------------------------------------------------------- inputs
        //
        // SOURCE and TARGET share one 72 dp row (VOICE joins it as a third equal slot
        // while Lip Sync is on), and the inputs are always on screen -- an empty slot is
        // a call to action, and neither input may push the other off the first screen.
        // The source is a face that never changes during a run, so it stays a thumbnail
        // for its whole life; the target tile doubles as the ORIGINAL half of the
        // before/after -- its frame shows the source frame of the swap -- so there is no
        // separate full-width "original" pane below any more.
        //
        // ONE card wraps the whole row so the tiles read as a single input group, with
        // the same surface fill and hairline the section cards use. It is exactly as
        // tall as the 72 dp tiles -- no breathing room top or bottom, the tiles now own
        // the full card edge-to-edge (their own hairlines are gone too). The same 10 dp
        // the row leaves between tiles is mirrored on the inside edges, so the group is
        // framed.
        Box(
            Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(16.dp))
                .background(MaterialTheme.colorScheme.surface)
                .border(1.dp, MaterialTheme.colorScheme.outlineVariant, RoundedCornerShape(16.dp)),
        ) {
            Row(
                Modifier.fillMaxWidth().fillMaxHeight(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                FaceTile(
                    label = stringResource(R.string.swap_source_face),
                    bitmap = sourceThumb,
                    placeholder = stringResource(R.string.swap_source_pick),
                    modifier = Modifier.weight(1f).edgeLine(
                        end = true,
                        color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.31f),
                    ),
                    onClick = if (idle) onPickSource else null,
                    actionIcon = if (hasSource) null else Icons.Default.Add,
                    actions = {
                        // Shoot a face instead of finding one. Stills only: a source is an
                        // identity, and there is no video form of that.
                        if (idle) {
                            IconButton(onCaptureSource, Modifier.size(26.dp)) {
                                Icon(painterResource(R.drawable.ic_photo_camera),
                                     stringResource(R.string.swap_capture_source), Modifier.size(14.dp),
                                     tint = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }
                    },
                    bottomActions = {
                        // Removing the source is not destructive -- it drops a reference to a photo
                        // the user still has -- so unlike the output it does not confirm.
                        if (hasSource && idle) {
                            IconButton(onClearSource, Modifier.size(26.dp)) {
                                Icon(Icons.Default.Delete,
                                     stringResource(R.string.swap_remove_source),
                                     Modifier.size(14.dp),
                                     tint = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }
                    },
                )
                if (opts.lipSync) FaceTile(
                    label = stringResource(R.string.swap_pane_voice),
                    bitmap = null,
                    placeholder = if (hasVoice) (voiceName ?: stringResource(R.string.swap_voice_picked))
                                  else stringResource(R.string.swap_voice_add),
                    modifier = Modifier.weight(1f),
                    // The tile IS the picker, except while a clip is loaded or the mic is
                    // live -- the record button owns the interaction then, and the whole-tile
                    // tap must not fire mid-capture.
                    onClick = if (idle && !recordingVoice) onPickVoice else null,
                    actionIcon = if (hasVoice) null else Icons.Default.Add,
                    actions = {
                        // RECORD. The lip syncer needs a voice that is not the target's own
                        // audio, and the microphone is the one source every user has -- no file
                        // to go find first.
                        if (idle) {
                            IconButton(onToggleRecordVoice, modifier = Modifier.size(26.dp)) {
                                Icon(painterResource(if (recordingVoice) R.drawable.ic_stop
                                                     else R.drawable.ic_mic),
                                     stringResource(if (recordingVoice) R.string.swap_voice_stop
                                                    else R.string.swap_voice_record),
                                     Modifier.size(14.dp),
                                     tint = if (recordingVoice) MaterialTheme.colorScheme.error
                                            else MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }
                    },
                    bottomActions = {
                        if (hasVoice && idle) {
                            IconButton(onClearVoice, modifier = Modifier.size(26.dp)) {
                                Icon(Icons.Default.Delete, stringResource(R.string.swap_remove_voice),
                                     Modifier.size(14.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }
                    },
                )
                FaceTile(
                    // The tile names what it holds: TARGET while asking for one, ORIGINAL once
                    // it is showing the source frame. The timestamp is dropped here -- the
                    // tile's badge is a 9 sp plate, and a clock string does not survive that.
                    label = stringResource(if (hasTarget) R.string.swap_pane_original
                                           else R.string.swap_pane_target),
                    bitmap = preview.original,
                    placeholder = stringResource(when {
                        run.preparing -> R.string.swap_reading_video
                        hasTarget -> R.string.swap_seeking
                        else -> R.string.swap_add_target
                    }),
                    modifier = Modifier.weight(1f).edgeLine(
                        start = true,
                        color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.31f),
                    ),
                    // The tile IS the picker. A separate full-width button said the same thing
                    // twice and cost a row of height the wordmark needed.
                    onClick = if (idle) onPickTarget else null,
                    actionIcon = if (hasTarget) null else Icons.Default.Add,
                    // The still camera keeps the top corner with the other tile actions; the
                    // video camera goes back to the bottom-right corner it started in.
                    actions = {
                        if (!hasTarget) {
                            // CAMERA, beside the gallery pick, shown while the tile is EMPTY -- which
                            // is when someone deciding what to swap needs it. Two buttons because a
                            // still and a clip take different routes through the system camera, and
                            // one button that then asks which is a tap for a question the icons answer.
                            IconButton(onCapturePhoto, enabled = idle, modifier = Modifier.size(26.dp)) {
                                Icon(painterResource(R.drawable.ic_photo_camera),
                                     stringResource(R.string.swap_capture_photo), Modifier.size(14.dp),
                                     tint = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }
                    },
                    bottomActions = {
                        // VIDEO CAMERA, back on its original seat in the bottom-right corner.
                        if (!hasTarget) {
                            IconButton(onCaptureVideo, enabled = idle, modifier = Modifier.size(26.dp)) {
                                Icon(painterResource(R.drawable.ic_videocam),
                                     stringResource(R.string.swap_capture_video), Modifier.size(14.dp),
                                     tint = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }
                        // TRASH, on the same bottom edge as the other tiles' delete.
                        if (hasTarget) {
                            IconButton(onClearTarget, enabled = idle, modifier = Modifier.size(26.dp)) {
                                Icon(Icons.Default.Delete, stringResource(R.string.swap_remove_target),
                                     Modifier.size(14.dp),
                                     tint = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }
                    },
                )
            }
        }
        // ---------------------------------------------------------------- previews
        //
        // Read as a before/after of ONE frame, so the two are always the same size as each
        // other. WHICH WAY they stack follows the footage: a portrait clip in two stacked
        // full-width boxes is mostly empty grey, because ContentScale.Fit letterboxes a
        // 9:16 image into a 16:9 box and throws away about two thirds of the width. Side by
        // side, each pane is half as wide and the image fills it.
// ---------------------------------------------------------------- result
        //
        // The result of the swap gets a full-width pane of its own, sized from the TARGET.
        // The pane fills the available width; its height follows the target's aspect ratio
        // capped so a tall target never eats the screen.
        val maxPaneW = (screenW - 36).dp
        // ⚠ The result pane's height has a CEILING, or a tall target eats the screen:
        // a tall portrait frame scaled to full width would be taller than a phone.
        // The pane used to size itself freely, and the Swap button -- everything below
        // the pane, really -- slid off the first screen; the button was still THERE and
        // still clickable at the edge of the fold, it just could not be seen.
        val maxResultH = (screenH - 460).dp.coerceIn(180.dp, 420.dp)
        val tW = preview.original?.width ?: 0
        val tH = preview.original?.height ?: 0
        val resultH: Dp
        if (tW > 0 && tH > 0) {
            val aspect = tW.toFloat() / tH.toFloat()   // width / height
            var h = maxPaneW / aspect
            if (h > maxResultH) {
                h = maxResultH
            }
            resultH = h
        } else {
            resultH = paneHeight.coerceAtMost(maxResultH)
        }

        // Always shown by default. The placeholder reads as a call to action until the
        // inputs exist, and once they do it is the after half of the before/after.
        //
        // `modelsMissing` keeps the download overlay reachable on a fresh install: it
        // lives on this pane because it is the one that cannot draw without the models.
        PreviewPane(
            label = stringResource(R.string.swap_pane_swapped),
            height = resultH,
            bitmap = preview.swapped,
            placeholder = when {
                modelsMissing -> ""
                // Already a finished, localized sentence from the Activity -- notably
                // the content gate's refusal, which must not be rebuilt here.
                preview.note != null -> preview.note
                preview.busy && !preview.warm ->
                    stringResource(R.string.swap_loading_models)
                preview.busy -> stringResource(R.string.swap_swapping_frame)
                !hasSource -> stringResource(R.string.swap_pick_a_source)
                // No "tap refresh" any more: the preview warms itself as soon as both
                // inputs exist, so this is a transient state rather than an instruction.
                else -> stringResource(R.string.swap_preparing_preview)
            },
            // Full width: the result is meant to fill the screen, not sit at 60%.
            modifier = Modifier.fillMaxWidth(),
            // The download lives here rather than in a bar of its own: this is the pane
            // that cannot draw anything without the models, so it is where their absence
            // is already visible.
            overlay = if (modelsMissing) { { DownloadOverlay(onDownload) } } else null,
            zoom = zoom,
        ) {
            // Spinner WHILE working, save button when there is something to save. Never
            // both: the fixed slot height in PreviewPane keeps either from moving the
            // trim slider and the Swap button down the screen mid-interaction.
            //
            // The save writes the previewed frame straight out of the pane. The output
            // pane has had a Save frame button since the video path existed, but it can
            // only reach frames of a FINISHED run -- so pulling one still out of a clip
            // meant swapping the whole clip first.
            if (!preview.busy && preview.swapped != null) {
                IconButton(onClick = onSavePreviewFrame, enabled = idle) {
                    Icon(
                        IconDownload,
                        stringResource(R.string.out_save_frame),
                        Modifier.size(18.dp),
                    )
                }
            }
            if (preview.busy) {
                CircularProgressIndicator(
                    Modifier.size(18.dp),
                    strokeWidth = 2.dp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        // ---------------------------------------------------------------- batch add
        //
        // 批量添加，位于输出设置上方，与输出设置同为可折叠菜单（默认折叠）。
        // 内部容器采用与"源人脸"输入行相同的卡片样式，承载"添加更多片段"、
        // "每个片段完成后立即保存到相册"以及已加入的片段列表。
        if ((hasTarget && !imageTarget) || batch.isNotEmpty()) {
            SectionCard(
                stringResource(R.string.swap_batch_menu),
                collapsible = true,
                expanded = batchMenuExpanded,
                onToggle = { batchMenuExpanded = !batchMenuExpanded },
            ) {
                Box(
                    Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(16.dp))
                        .background(MaterialTheme.colorScheme.surface)
                        .border(1.dp, MaterialTheme.colorScheme.outlineVariant,
                                RoundedCornerShape(16.dp)),
                ) {
                    Column(Modifier.padding(vertical = 4.dp)) {
                        // 添加更多片段按钮（仅视频目标已加载时显示）
                        if (hasTarget && !imageTarget && idle) {
                            TextButton(onAddToBatch, modifier = Modifier.fillMaxWidth()) {
                                Text(stringResource(if (batch.size > 1) R.string.swap_batch_add_more
                                                    else R.string.swap_batch_add))
                            }
                        }
                        // 自动保存 + 队列列表
                        if (batch.isNotEmpty()) {
                            Row(
                                Modifier
                                    .fillMaxWidth()
                                    .clickable(enabled = idle) { onBatchAutoSave(!batchAutoSave) }
                                    .padding(start = 6.dp, end = 14.dp, top = 2.dp, bottom = 2.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Checkbox(batchAutoSave, { onBatchAutoSave(it) }, enabled = idle)
                                Text(stringResource(R.string.batch_autosave),
                                     style = MaterialTheme.typography.bodySmall)
                            }
                            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                            batch.forEachIndexed { i, item ->
                                if (i > 0) HorizontalDivider(
                                    color = MaterialTheme.colorScheme.outlineVariant)
                                Row(
                                    Modifier.fillMaxWidth()
                                        // A finished row IS the way back to its clip.
                                        .clickable(enabled = idle && item.output != null) {
                                            onOpenBatchOutput(i)
                                        }
                                        .padding(horizontal = 14.dp, vertical = 8.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    if (item.thumb != null) {
                                        Image(
                                            item.thumb!!.asImageBitmap(), null,
                                            Modifier
                                                .size(44.dp, 30.dp)
                                                .clip(RoundedCornerShape(4.dp)),
                                            contentScale = ContentScale.Crop,
                                        )
                                        Spacer(Modifier.width(10.dp))
                                    }
                                    Column(Modifier.weight(1f)) {
                                        Text(item.name, style = MaterialTheme.typography.bodySmall,
                                             maxLines = 1, overflow = TextOverflow.Ellipsis)
                                        if (item.detail != null) Text(
                                            item.detail!!,
                                            style = MaterialTheme.typography.bodySmall,
                                            fontSize = 10.sp,
                                            maxLines = 2,
                                            overflow = TextOverflow.Ellipsis,
                                            color = if (item.state == BatchState.Refused)
                                                        MaterialTheme.colorScheme.onSurfaceVariant
                                                    else MaterialTheme.colorScheme.error,
                                        )
                                    }
                                    Text(
                                        stringResource(when (item.state) {
                                            BatchState.Waiting -> R.string.batch_waiting
                                            BatchState.Running -> R.string.batch_running
                                            BatchState.Done -> R.string.batch_done
                                            BatchState.Refused -> R.string.batch_refused
                                            BatchState.Failed -> R.string.batch_failed
                                            BatchState.Skipped -> R.string.batch_skipped
                                        }),
                                        style = MaterialTheme.typography.bodySmall,
                                        fontSize = 10.sp,
                                        fontFamily = FontFamily.Monospace,
                                        color = when (item.state) {
                                            BatchState.Done -> FfRed
                                            BatchState.Failed -> MaterialTheme.colorScheme.error
                                            else -> MaterialTheme.colorScheme.onSurfaceVariant
                                        },
                                    )
                                    if (idle) {
                                        IconButton({ onRemoveFromBatch(i) },
                                                   modifier = Modifier.size(32.dp)) {
                                            Icon(Icons.Default.Delete,
                                                 stringResource(R.string.batch_remove),
                                                 Modifier.size(16.dp),
                                                 tint = MaterialTheme.colorScheme.onSurfaceVariant)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // ---------------------------------------------------------------- trim
        //
        // Clip and frame rate share ONE foldable card, closed by default. They are
        // per-run tuning knobs, not standing controls, and two standing controls -- the
        // range slider plus the rate steps -- pushed the Swap button off the first
        // screen on every video target. The card keeps the chosen range in its header,
        // 输出设置（首项为片段，然后是输出尺寸与帧率）
        if (durationMs > 0) {
            SectionCard(
                stringResource(R.string.swap_output_settings),
                collapsible = true,
                expanded = trimExpanded,
                onToggle = { trimExpanded = !trimExpanded },
            ) {
                // 片段 — 输出设置首项
                Row(
                    Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        stringResource(R.string.swap_clip_rate),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        "${fmt(trimStartMs)} – ${fmt(trimEndMs)}",
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                    )
                }
                val trimSliderColors = SliderDefaults.colors(
                    thumbColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.69f),
                    activeTrackColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.69f),
                    inactiveTrackColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.69f),
                )
                RangeSlider(
                    value = trimStartMs..trimEndMs,
                    onValueChange = { r ->
                        // Which handle moved: RangeSlider reports the whole range, so the
                        // edge has to be inferred by comparing against what it was. The
                        // previews then follow the handle under the finger rather than
                        // always showing the start frame.
                        val edge = if (r.start != trimStartMs) TrimEdge.Start else TrimEdge.End
                        // Keep at least a third of a second, so the encoder always gets a frame.
                        onTrimChange(r.start, maxOf(r.endInclusive, r.start + 333f), edge)
                    },
                    valueRange = 0f..durationMs.toFloat(),
                    enabled = idle,
                    colors = trimSliderColors,
                )
                // The REAL rate, not a hardcoded 30. The estimate was wrong on every
                // clip that was not 30 fps, and it is the number the ETA is read against.
                val effFps = if (opts.outputFps in 1..inputFps) opts.outputFps else inputFps
                val estFrames = ((trimEndMs - trimStartMs) / 1000f * effFps).roundToInt()
                Text(
                    stringResource(R.string.swap_clip_summary,
                                   estFrames, fmt(durationMs.toFloat()), effFps),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                // Frame rate and output size are the same kind of decision.
                Spacer(Modifier.height(6.dp))
                // ⚠ The SOURCE option is named by its own number, not by the word "same".
                // Sitting in a row that reads 480p / 720p / 1080p, "Same as source" was the
                // one chip that did not say what it would produce -- and the clip's size is
                // already on this screen, so there was nothing to look up. A clip whose
                // short edge is not a familiar number ("606p") still reads honestly, and
                // the hint underneath carries the full WxH either way.
                val srcShort = minOf(targetW, targetH)
                // "Same (960p)": the word says what the choice MEANS and the number says
                // what it produces. The number alone made the source chip look like one
                // more fixed size rather than the leave-it-alone option, which is what it
                // is and what most runs want.
                val srcName = if (srcShort > 0)
                                  stringResource(R.string.swap_size_source_at, srcShort)
                              else stringResource(R.string.swap_size_source)
                // OUTPUT SIZE, on the SHORT edge so the aspect ratio never changes and
                // "480p" means what it means everywhere else. Only sizes BELOW the clip's
                // own are offered, for the same reason the frame rate only offers lower
                // rates: enlarging costs bitrate and adds nothing, because the swapper runs
                // at 256 whatever the frame is.
                //
                // ⚠ It is applied at DECODE, so it makes the RUN faster too -- detector prep
                // and paste-back scale with frame area, and 4K is ~9x the area of 1080p.
                // What it cannot do is make a face sharper; that is pixel boost and the
                // enhancer, and this control must not be mistaken for them.
                val shortEdge = srcShort
                val sizes = listOf(480, 720, 1080).filter { it < shortEdge }
                                .map { it to (it.toString() + "p") } +
                            listOf(0 to srcName)
                if (sizes.size > 1) {
                    OptionSteps(
                        stringResource(R.string.swap_output_size),
                        sizes,
                        if (opts.outputMaxShortEdge in 1 until shortEdge)
                            opts.outputMaxShortEdge else 0,
                        { onOptsChange(opts.copy(outputMaxShortEdge = it)) },
                        hint = if (opts.outputMaxShortEdge in 1 until shortEdge)
                                   stringResource(R.string.swap_size_hint_smaller)
                               else if (targetW > 0 && targetH > 0)
                                   stringResource(R.string.swap_size_hint_source_dims,
                                                  targetW, targetH)
                               else stringResource(R.string.swap_size_hint_source),
                        enabled = idle,
                    )
                }

                // Frame rate. Only rates BELOW the input's are offered: a higher one would
                // duplicate frames, and each duplicate costs a full swap to produce nothing
                // new. Dropping frames is the only direction that saves anything.
                //
                // ⚠ The low stops are the point, and 24/30/60 alone were not enough to be
                // useful. On a 30 fps clip the deepest cut available was 24 -- a 20% saving
                // against the CPU backend, which is an order of magnitude slower than the
                // NPU -- and on a 24 fps clip nothing qualified, so the control hid itself
                // and offered no reduction at all. 5/10/15 are what make it worth having:
                // 30 -> 10 is a third of the frames and close to a third of the time,
                // because VideoSwapper decimates BEFORE the swap rather than after it.
                //
                // Ascending, with "same as source" last: the slider then runs from cheapest
                // on the left to full quality on the right, which is the direction the
                // trade-off reads in.
                val rates = listOf(5, 10, 15, 24, 30, 60).filter { it < inputFps }
                                .map { it to "$it" } +
                            listOf(0 to stringResource(R.string.swap_rate_same, inputFps))
                if (rates.size > 1) {
                    OptionSteps(
                        stringResource(R.string.swap_frame_rate),
                        rates,
                        if (opts.outputFps in 1..inputFps) opts.outputFps else 0,
                        { onOptsChange(opts.copy(outputFps = it)) },
                        hint = if (opts.outputFps == 0 || opts.outputFps >= inputFps)
                                   stringResource(R.string.swap_rate_hint_every)
                               else stringResource(R.string.swap_rate_hint_drop),
                        enabled = idle,
                    )
                }
            }
        }

        // ---------------------------------------------------------------- run
        // One button, two jobs: a separate Cancel would sit dead for the entire time the
        // only thing you can do is start a swap.
        //
        // A still target has no button at all. The pane above IS the output, so a Swap
        // button would offer to compute something the user is already looking at, and the
        // Save button below is the only thing left to do.
        if (!imageTarget) {
            // The primary action carries the brand gradient -- the same sweep as the
            // header, FaceFusion red deepening to its hover state -- and a play mark, so
            // the one thing that starts a run reads as the thing that starts a run. While
            // busy it flips to a bordered cancel, because a solid red button that now says
            // "Cancel" would look like a run that is still inviting to be started.
            val busy = run.busy
            // READY = 两个输入都在、模型齐、Lip Sync 有驱动音。它刻意不含 idle：
            // 添加目标后要复制视频文件并读元数据（preparing，可能耗时数秒），期间
            // 若把按钮压成半透明灰，用户看到"目标已选好按钮却是灰的"会以为坏了。
            val ready = hasSource && hasTarget && !modelsMissing &&
                        (!opts.lipSync || hasVoice)
            val canRun = idle && ready
            // 只有真正缺条件才置灰；preparing 期间按钮保持品牌色，只是暂时不可点。
            val dimmed = !busy && !ready
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(56.dp)
                    .clip(RoundedCornerShape(16.dp))
                    // The button's fill matches the other controls (surface, like the
                    // SectionCards), not the page background. Its identity comes from the
                    // border (outlineVariant) and the text weight, exactly like the
                    // processor chips; only a running swap shows the error border. The
                    // dimmed state keeps the same fill, just muted text.
                    .background(MaterialTheme.colorScheme.surface)
                    .border(
                        1.dp,
                        if (busy) MaterialTheme.colorScheme.error
                        else MaterialTheme.colorScheme.outlineVariant,
                        RoundedCornerShape(16.dp),
                    )
                    .clickable(enabled = busy || canRun) {
                        if (busy) onCancel() else onSwap()
                    },
                contentAlignment = Alignment.Center,
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    if (!busy) {
                        Icon(
                            Icons.Default.PlayArrow,
                            stringResource(R.string.swap_action),
                            Modifier.size(22.dp),
                            // Dim the icon with the text when conditions are truly missing;
                            // otherwise it follows the label colour.
                            tint = if (dimmed) MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.69f)
                                   else MaterialTheme.colorScheme.onBackground,
                        )
                    }
                    Text(
                        stringResource(
                            if (busy) R.string.swap_cancel
                            else if (batch.size > 1) R.string.swap_action_batch
                            else R.string.swap_action,
                            batch.size,
                        ),
                        fontSize = 17.sp,
                        fontWeight = FontWeight.SemiBold,
                        letterSpacing = 0.5.sp,
                        color = when {
                            busy -> MaterialTheme.colorScheme.error
                            // Not clickable: mute the label itself, 31 % lighter than normal.
                            dimmed -> MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.69f)
                            else -> MaterialTheme.colorScheme.onBackground
                        },
                    )
                }
            }
        }

        if (run.busy || run.progress > 0f) {
            LinearProgressIndicator(
                progress = { run.progress.coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth(),
            )
            if (run.framesTotal > 0) {
                val fps = if (run.elapsedS > 0) run.framesDone / run.elapsedS else 0.0
                val eta = if (fps > 0) (run.framesTotal - run.framesDone) / fps else 0.0
                Text(
                    stringResource(R.string.swap_progress, run.framesDone, run.framesTotal,
                                   "%.1f".format(fps), eta.toInt()),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        // Only when there is something to report. It used to carry a standing
        // instruction, which the two empty preview panes above already give.
        if (status.isNotEmpty()) {
            Text(status, style = MaterialTheme.typography.bodyMedium)
            // Only on a failure. A crash leaves no in-app log at all, which is why
            // BugReport also persists uncaught exceptions for the next launch.
            if (statusIsError) {
                TextButton(onShareLog) { Text(stringResource(R.string.swap_share_bug_report)) }
            }
        }

        // ---------------------------------------------------------------- output result
        //
        // 输出结果，位于日志上方，与输出设置同为可折叠菜单（默认折叠）。
        // 标题行带有下载图标，其点击事件和显示状态与"保存到相册"按钮一致。
        if (outputFile != null || hasOutput) {
            SectionCard(
                stringResource(R.string.swap_output_result),
                collapsible = true,
                expanded = outputResultExpanded,
                onToggle = { outputResultExpanded = !outputResultExpanded },
                trailing = {
                    if (hasOutput && !outputAutoSaved) {
                        IconButton(onClick = onSave, enabled = idle, modifier = Modifier.size(26.dp)) {
                            Icon(
                                IconDownload,
                                stringResource(R.string.swap_save_to_gallery),
                                Modifier.size(18.dp),
                            )
                        }
                    }
                },
            ) {
            if (outputFile != null) {
                // SWIPE BETWEEN BATCH RESULTS. The indices of everything finished, and where
                // the pane currently sits in that list.
                val doneIx = batch.indices.filter { batch[it].output != null }
                val cur = doneIx.indexOfFirst { batch[it].output == outputFile }
                var drag by remember(outputFile) { mutableStateOf(0f) }
                Box(
                    Modifier.pointerInput(doneIx.size, cur) {
                        if (doneIx.size < 2 || cur < 0) return@pointerInput
                        // ⚠ HORIZONTAL only, and accumulated to a threshold rather than acted
                        // on per event. detectHorizontalDragGestures ignores a vertical-dominant
                        // drag, so the page still scrolls with a finger on the video -- which
                        // matters, because this pane is most of the screen.
                        detectHorizontalDragGestures(
                            onDragEnd = {
                                val step = if (drag < -60f) 1 else if (drag > 60f) -1 else 0
                                drag = 0f
                                if (step != 0)
                                    doneIx.getOrNull(cur + step)?.let(onOpenBatchOutput)
                            },
                            onDragCancel = { drag = 0f },
                        ) { change, amount -> drag += amount; change.consume() }
                    }
                ) {
                    OutputPane(
                        file = outputFile,
                        height = paneHeight,
                        onSaveFrame = onSaveFrame,
                        partial = outputPartial,
                        enabled = idle,
                    )
                }
                // Says the swipe exists. A gesture with nothing on screen to suggest it is a
                // gesture only its author knows about -- which is what the batch queue itself
                // had just been.
                if (doneIx.size > 1 && cur >= 0) {
                    Text(
                        stringResource(R.string.batch_output_of, cur + 1, doneIx.size),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.fillMaxWidth(),
                        textAlign = TextAlign.Center,
                    )
                }
            }

            if (hasOutput) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    // Auto-save already put this clip in the gallery, so there is nothing to
                    // offer -- just a line saying where it went. Share stays: sending it
                    // somewhere is a different action from keeping it.
                    if (outputAutoSaved) {
                        Text(
                            stringResource(R.string.swap_autosaved_to_gallery),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.weight(1f),
                        )
                    } else {
                        Button(onSave, enabled = idle, modifier = Modifier.weight(1f),
                               shape = RoundedCornerShape(14.dp),
                               colors = ButtonDefaults.buttonColors(
                                   containerColor = MaterialTheme.colorScheme.surface,
                                   contentColor = MaterialTheme.colorScheme.onBackground,
                                   disabledContainerColor = MaterialTheme.colorScheme.surface,
                                   disabledContentColor = MaterialTheme.colorScheme.onSurfaceVariant,
                               ),
                               border = BorderStroke(1.dp,
                                                     MaterialTheme.colorScheme.outlineVariant)) {
                            Text(stringResource(if (saved) R.string.swap_saved_to_gallery
                                                else R.string.swap_save_to_gallery))
                        }
                    }
                    OutlinedButton(onShare, enabled = idle,
                                   shape = RoundedCornerShape(14.dp),
                                   // Same control background as the Save button next to it:
                                   // card-surface in both schemes, not the default accent.
                                   colors = ButtonDefaults.outlinedButtonColors(
                                       containerColor = MaterialTheme.colorScheme.surface,
                                       contentColor = MaterialTheme.colorScheme.onBackground,
                                       disabledContainerColor = MaterialTheme.colorScheme.surface,
                                       disabledContentColor = MaterialTheme.colorScheme.onSurfaceVariant,
                                   ),
                                   border = BorderStroke(1.dp,
                                                         MaterialTheme.colorScheme.outlineVariant)) {
                        Text(stringResource(R.string.swap_share))
                    }
                    // Deleting a render IS destructive -- minutes of NPU time, and the file is
                    // gone from the phone -- so this one asks, unlike the source and target
                    // buttons, which only drop a reference to a file the user still has.
                    //
                    // ⚠ VIDEO ONLY, and that is not an oversight. A still has no output file:
                    // its result is the swapped PANE, regenerated from the source and target
                    // whenever both are present. The button was shown for stills too and did
                    // nothing at all -- discardOutput() deletes outputFile, which is null on
                    // that path -- so it confirmed and then visibly ignored the answer.
                    //
                    // Clearing the pane instead would be worse, not better: the autowarm effect
                    // would redraw it within the same second. The way to get rid of a still's
                    // result is to remove the target, which has its own button on its own pane.
                    if (outputFile != null) {
                        OutlinedButton({ confirmDeleteOutput = true }, enabled = idle,
                                       shape = RoundedCornerShape(14.dp),
                                       // Same control background as the buttons around it.
                                       colors = ButtonDefaults.outlinedButtonColors(
                                           containerColor = MaterialTheme.colorScheme.surface,
                                           contentColor = MaterialTheme.colorScheme.onBackground,
                                           disabledContainerColor = MaterialTheme.colorScheme.surface,
                                           disabledContentColor = MaterialTheme.colorScheme.onSurfaceVariant,
                                       ),
                                       border = BorderStroke(1.dp,
                                                             MaterialTheme.colorScheme.outlineVariant)) {
                            Icon(Icons.Default.Delete, stringResource(R.string.out_delete),
                                 Modifier.size(18.dp))
                        }
                    }
                }
                if (savedPath != null)
                    Text(
                        savedPath,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
            }
            }
        }

        // ---------------------------------------------------------------- log
        if (log.isNotEmpty()) LogBox(log, expanded = logExpanded,
                                     onToggle = { logExpanded = !logExpanded })

        Spacer(Modifier.height(8.dp))
    }

    if (confirmDeleteOutput) {
        AlertDialog(
            onDismissRequest = { confirmDeleteOutput = false },
            title = { Text(stringResource(R.string.out_delete_title)) },
            text = { Text(stringResource(R.string.out_delete_body)) },
            confirmButton = {
                TextButton({ confirmDeleteOutput = false; onDeleteOutput() }) {
                    Text(stringResource(R.string.common_delete))
                }
            },
            dismissButton = {
                TextButton({ confirmDeleteOutput = false }) {
                    Text(stringResource(R.string.common_cancel))
                }
            },
        )
    }

    // ---------------------------------------------------------------- settings sheets
    //
    // What used to be the Advanced accordion, split by the processor each group belongs to
    // and hung off that processor's own gear. Advanced sat below the trim slider and the
    // Swap button, so reaching a mask blur meant scrolling PAST the control that starts the
    // run -- and every group in it was reached the same way regardless of which stage it
    // configured.
    //
    // ⚠ Face Masker and Face Detector are NOT the swapper's own settings: the masker is
    // shared with the lip syncer (one BoxMaskCache serves both, see Pipeline::Impl) and the
    // detector feeds every stage. They live behind the swapper's gear because face_swapper
    // is the one processor that is always on, so its gear is the one that can always be
    // reached -- not because they belong to it. Anything added here that a second stage
    // also reads deserves the same note.
    val sheet = settingsFor
    if (sheet != null) {
        AlertDialog(
            onDismissRequest = { settingsFor = null },
            confirmButton = {
                TextButton({ settingsFor = null }) { Text(stringResource(R.string.swap_close)) }
            },
            title = {
                Text(stringResource(when (sheet) {
                    "enhancer" -> R.string.swap_proc_enhancer
                    "lipsync"  -> R.string.swap_proc_lip_syncer
                    else       -> R.string.swap_proc_swapper
                }))
            },
            text = {
                // Scrollable: the swapper sheet holds three expandable cards, and all three
                // open at once is taller than a phone in landscape.
                Column(
                    Modifier.verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    when (sheet) {
                        "enhancer" -> {
                            OptionSlider(
                                stringResource(R.string.opt_blend), opts.enhanceBlend,
                                { onOptsChange(opts.copy(enhanceBlend = it)) },
                                hint = when {
                                    opts.enhanceBlend >= 0.95f ->
                                        stringResource(R.string.opt_blend_hint_full)
                                    opts.enhanceBlend <= 0.05f ->
                                        stringResource(R.string.opt_blend_hint_none)
                                    else -> stringResource(R.string.opt_blend_hint_mixed)
                                },
                            )
                            // It runs on the swapper's own crop: gpen_bfr_256 and
                            // hyperswap_1a_256 declare the same template and size, so no
                            // second alignment is involved.
                            Text(stringResource(R.string.opt_enhancer_note, opts.pixelBoostLabel),
                                 style = MaterialTheme.typography.bodySmall, fontSize = 11.sp)
                        }
                        "lipsync" -> {
                            OptionSlider(
                                stringResource(R.string.opt_weight), opts.lipSyncWeight,
                                { onOptsChange(opts.copy(lipSyncWeight = it)) },
                                hint = stringResource(R.string.opt_lip_sync_weight_hint),
                            )
                            // The Voice picker deliberately stays on the main screen: it is
                            // a REQUIRED input that gates the Swap button, not a knob, and
                            // a required input behind a gear is a required input nobody
                            // finds.
                        }
                        else -> {
                            FaceSwapperCard(opts, onOptsChange, openCard == "swapper",
                                            { onToggleCard("swapper") },
                                            inswapperAvailable = hasInswapper)
                            FaceMaskerCard(opts, onOptsChange, openCard == "masker",
                                           { onToggleCard("masker") })
                            FaceDetectorCard(opts, onOptsChange, openCard == "detector",
                                             { onToggleCard("detector") })
                            if (opts != SwapOptions()) {
                                TextButton(
                                    onClick = { onOptsChange(SwapOptions()) },
                                    modifier = Modifier.align(Alignment.End),
                                ) { Text(stringResource(R.string.swap_reset_defaults)) }
                            }
                        }
                    }
                }
            },
        )
    }
}

/**
 * The model download, drawn over a preview pane.
 *
 * Only ever composed when the files are actually missing, so there is no button sitting
 * around inviting a 275 MB transfer nobody needs.
 *
 * Shared with [LiveScreen] rather than private to this file: Live is a tab, so it can be
 * the first screen a fresh install sees, and it needs the same offer. It briefly had a
 * plain Button of its own instead -- same onDownload, but none of the progress, the byte
 * counter, the error or the retry, so the two screens disagreed about what a download
 * looks like for no reason beyond where the composable happened to live.
 */
@Composable
fun DownloadOverlay(onDownload: () -> Unit) {
    Surface(color = MaterialTheme.colorScheme.surface.copy(alpha = 0.92f)) {
        Column(
            Modifier.fillMaxSize().padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            when {
                ModelDownload.running -> {
                    Text(ModelDownload.currentName,
                         style = MaterialTheme.typography.bodyMedium,
                         fontFamily = FontFamily.Monospace)
                    Spacer(Modifier.height(8.dp))
                    LinearProgressIndicator(
                        progress = { ModelDownload.progress },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Spacer(Modifier.height(8.dp))
                    Text(
                        stringResource(R.string.dl_progress,
                            ModelDownload.doneBytes / 1048576,
                            ModelDownload.totalBytes / 1048576,
                            ModelDownload.fileIndex, ModelDownload.fileCount,
                        ),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                else -> {
                    Text(stringResource(R.string.dl_models_required),
                         style = MaterialTheme.typography.titleSmall)
                    Spacer(Modifier.height(4.dp))
                    Text(
                        ModelDownload.error ?: stringResource(R.string.dl_not_on_device),
                        style = MaterialTheme.typography.bodySmall,
                        textAlign = TextAlign.Center,
                        color = if (ModelDownload.error != null)
                            MaterialTheme.colorScheme.error
                        else MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(Modifier.height(12.dp))
                    Button(onDownload, shape = RoundedCornerShape(14.dp)) {
                        Text(stringResource(if (ModelDownload.error != null) R.string.dl_retry
                                            else R.string.dl_download))
                    }
                }
            }
        }
    }
}

/**
 * A single hairline on one edge of a tile — the divider that separates a tile from its
 * neighbour without drawing a full box around it. Drawn over the tile's content so it
 * survives the tile's own surface fill.
 */
private fun Modifier.edgeLine(
    start: Boolean = false,
    end: Boolean = false,
    color: Color,
    stroke: Dp = 1.dp,
): Modifier = drawWithContent {
    drawContent()
    val w = stroke.toPx()
    if (start) {
        drawLine(color, Offset(w / 2f, 0f), Offset(w / 2f, size.height), w)
    }
    if (end) {
        drawLine(color, Offset(size.width - w / 2f, 0f), Offset(size.width - w / 2f, size.height), w)
    }
}
