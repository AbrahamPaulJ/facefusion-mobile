package com.facefusion.mobile.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.facefusion.mobile.FaceDetectorCard
import com.facefusion.mobile.FaceEnhancerCard
import com.facefusion.mobile.FaceMaskerCard
import com.facefusion.mobile.FaceSwapperCard
import com.facefusion.mobile.ModelDownload
import com.facefusion.mobile.OptionSegments
import com.facefusion.mobile.SwapOptions
import java.io.File
import kotlin.math.roundToInt

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
    fmt: (Float) -> String,
    preview: PreviewUi,
    run: RunUi,
    status: String,
    log: String,
    opts: SwapOptions,
    onOptsChange: (SwapOptions) -> Unit,
    hasInswapper: Boolean,
    hasEnhancer: Boolean,
    openCard: String,
    onToggleCard: (String) -> Unit,
    advancedOpen: Boolean,
    onToggleAdvanced: () -> Unit,
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
    onClearTarget: () -> Unit,
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

    // Both panes share one height, chosen so the pair fits on screen together with the
    // controls around them. The reserve is the wordmark, source button, captions, trim,
    // Swap and the nav bar; whatever is left is split in two.
    val screenH = LocalConfiguration.current.screenHeightDp
    // Side-by-side panes are half as wide, so they can afford to be taller: the pair costs
    // ONE pane's height instead of two, which is the whole reason portrait gets this layout.
    val paneHeight = if (targetAspect < 1f) (screenH - 400).coerceIn(200, 460).dp
                     else ((screenH - 470) / 2).coerceIn(140, 320).dp

    // ONE instance for every pane, which is what makes them zoom together (item 4).
    val zoom = remember { ZoomState() }
    Column(
        modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        // ---------------------------------------------------------------- inputs
        Row(
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            if (sourceThumb != null) {
                Image(
                    sourceThumb.asImageBitmap(), "Source face",
                    Modifier.size(52.dp).clip(RoundedCornerShape(10.dp)),
                    contentScale = ContentScale.Crop,
                )
            }
            OutlinedButton(onPickSource, enabled = idle, modifier = Modifier.weight(1f),
                           shape = RoundedCornerShape(14.dp)) {
                Text(if (hasSource) "Source face selected" else "Pick source face")
            }
        }
        // ---------------------------------------------------------------- previews
        //
        // Read as a before/after of ONE frame, so the two are always the same size as each
        // other. WHICH WAY they stack follows the footage: a portrait clip in two stacked
        // full-width boxes is mostly empty grey, because ContentScale.Fit letterboxes a
        // 9:16 image into a 16:9 box and throws away about two thirds of the width. Side by
        // side, each pane is half as wide and the image fills it.
        val portrait = targetAspect < 1f
        val panes: @Composable (Modifier) -> Unit = { paneModifier ->
            PreviewPane(
                label = if (preview.timeLabel.isEmpty()) "Original"
                        else "Original  ${preview.timeLabel}",
                height = paneHeight,
                bitmap = preview.original,
                placeholder = when {
                    run.preparing -> "Reading video..."
                    hasTarget -> "Seeking..."
                    else -> "Add a target"
                },
                modifier = paneModifier,
                // The pane IS the picker. A separate full-width button said the same thing
                // twice and cost a row of height the wordmark needed.
                onClick = if (idle) onPickTarget else null,
                actionIcon = if (hasTarget) null else Icons.Default.Add,
                zoom = zoom,
            ) {
                if (hasTarget) {
                    // Icons rather than the word "Change": two actions fit where one word
                    // did, and the pane itself is already the picker, so the word was
                    // saying a third time what the tap and the + icon already say.
                    IconButton(onPickTarget, enabled = idle, modifier = Modifier.size(36.dp)) {
                        Icon(Icons.Default.Add, "Choose another target",
                             Modifier.size(20.dp),
                             tint = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    IconButton(onClearTarget, enabled = idle, modifier = Modifier.size(36.dp)) {
                        Icon(Icons.Default.Delete, "Remove target",
                             Modifier.size(20.dp),
                             tint = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }

            // Hidden until there IS a target: an empty pane telling you to pick one is
            // the same instruction the pane above already gives, in the same words.
            //
            // `|| modelsMissing` because the download overlay lives on this pane -- it is
            // the one that cannot draw without the models -- so hiding it unconditionally
            // would leave a fresh install with no way to fetch them.
            if (hasTarget || modelsMissing) PreviewPane(
                label = "Swapped",
                height = paneHeight,
                bitmap = preview.swapped,
                placeholder = when {
                    modelsMissing -> ""
                    preview.note != null -> preview.note
                    preview.busy && !preview.warm -> "Loading models, this takes a few seconds..."
                    preview.busy -> "Swapping this frame..."
                    !hasSource -> "Pick a source face"
                    // No "tap refresh" any more: the preview warms itself as soon as both
                    // inputs exist, so this is a transient state rather than an instruction.
                    else -> "Preparing preview..."
                },
                modifier = paneModifier,
                // The download lives here rather than in a bar of its own: this is the pane
                // that cannot draw anything without the models, so it is where their absence
                // is already visible.
                overlay = if (modelsMissing) { { DownloadOverlay(onDownload) } } else null,
                zoom = zoom,
            ) {
                // Spinner only. The refresh button is gone -- see item 1 -- and the fixed
                // slot height in PreviewPane keeps this from moving anything below it.
                if (preview.busy) {
                    CircularProgressIndicator(
                        Modifier.size(18.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        if (portrait) {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                panes(Modifier.weight(1f))
            }
        } else {
            panes(Modifier)
        }

        // ---------------------------------------------------------------- trim
        if (durationMs > 0) {
            Column {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Caption("Clip", Modifier.weight(1f))
                    Text(
                        "${fmt(trimStartMs)} – ${fmt(trimEndMs)}",
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                    )
                }
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
                )
                // The REAL rate, not a hardcoded 30. The estimate was wrong on every
                // clip that was not 30 fps, and it is the number the ETA is read against.
                val effFps = if (opts.outputFps in 1..inputFps) opts.outputFps else inputFps
                val estFrames = ((trimEndMs - trimStartMs) / 1000f * effFps).roundToInt()
                Text(
                    "$estFrames frames of ${fmt(durationMs.toFloat())}   ${effFps} fps",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                // Frame rate. Only rates at or below the input's are offered: a higher one
                // would duplicate frames, and each duplicate costs a full swap to produce
                // nothing new. Dropping frames is the only direction that saves anything.
                val rates = listOf(0 to "Same ($inputFps)") +
                    listOf(24, 30, 60).filter { it < inputFps }.map { it to "$it" }
                if (rates.size > 1) {
                    Spacer(Modifier.height(6.dp))
                    OptionSegments(
                        "Frame rate",
                        rates,
                        if (opts.outputFps in 1..inputFps) opts.outputFps else 0,
                        { onOptsChange(opts.copy(outputFps = it)) },
                        hint = if (opts.outputFps == 0 || opts.outputFps >= inputFps)
                                   "every frame of the source"
                               else "drops frames before the swap, so it is faster too",
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
        if (!imageTarget) Button(
            onClick = if (run.busy) onCancel else onSwap,
            enabled = run.busy || (idle && hasSource && hasTarget && !modelsMissing),
            modifier = Modifier.fillMaxWidth().height(52.dp),
            // 14.dp everywhere: the stadium default made the two primary buttons the only
            // fully-round things on a screen of 14.dp panes and cards.
            shape = RoundedCornerShape(14.dp),
        ) { Text(if (run.busy) "Cancel" else "Swap", fontSize = 16.sp) }

        if (run.busy || run.progress > 0f) {
            LinearProgressIndicator(
                progress = { run.progress.coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth(),
            )
            if (run.framesTotal > 0) {
                val fps = if (run.elapsedS > 0) run.framesDone / run.elapsedS else 0.0
                val eta = if (fps > 0) (run.framesTotal - run.framesDone) / fps else 0.0
                Text(
                    "Frame ${run.framesDone} / ${run.framesTotal}   " +
                        "%.1f fps   ETA %ds".format(fps, eta.toInt()),
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
            if (status.startsWith("Failed")) {
                TextButton(onShareLog) { Text("Share bug report") }
            }
        }

        // ---------------------------------------------------------------- output
        //
        // The result was previously invisible in the app: Save and Share, and no way to see
        // what you were about to save. A video gets a player with a scrub bar; an image
        // result is a still, which is all there is to show.
        if (outputFile != null) {
            OutputPane(
                file = outputFile,
                height = paneHeight,
                onSaveFrame = onSaveFrame,
                partial = outputPartial,
                enabled = idle,
            )
        }

        if (hasOutput) {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Button(onSave, enabled = idle, modifier = Modifier.weight(1f),
                       shape = RoundedCornerShape(14.dp)) {
                    Text(if (saved) "Saved to gallery" else "Save to gallery")
                }
                OutlinedButton(onShare, enabled = idle,
                               shape = RoundedCornerShape(14.dp)) { Text("Share") }
            }
            if (savedPath != null)
                Text(
                    savedPath,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
        }

        // ---------------------------------------------------------------- advanced
        //
        // One accordion around the three option groups. They used to sit between the trim
        // slider and the Swap button, i.e. across the primary path, for settings almost
        // every run leaves alone.
        Accordion(
            "Advanced",
            if (opts == SwapOptions()) "FaceFusion defaults"
            else "weight %.2f   %s   blur %.2f".format(opts.weight, opts.pixelBoostLabel, opts.maskBlur),
            advancedOpen,
            onToggleAdvanced,
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Spacer(Modifier.height(2.dp))
                // Enhancer first: it is the newest control and the one being reached
                // for, and it is also the only one that can be absent -- so when it IS
                // there it should not be buried under three cards that are always there.
                if (hasEnhancer) {
                    FaceEnhancerCard(opts, onOptsChange, openCard == "enhancer",
                                     { onToggleCard("enhancer") })
                }
                FaceSwapperCard(opts, onOptsChange, openCard == "swapper",
                                { onToggleCard("swapper") }, inswapperAvailable = hasInswapper)
                FaceMaskerCard(opts, onOptsChange, openCard == "masker", { onToggleCard("masker") })
                FaceDetectorCard(opts, onOptsChange, openCard == "detector",
                                 { onToggleCard("detector") })
                if (opts != SwapOptions()) {
                    TextButton(
                        onClick = { onOptsChange(SwapOptions()) },
                        modifier = Modifier.align(Alignment.End),
                    ) { Text("Reset to defaults") }
                }
            }
        }

        // ---------------------------------------------------------------- log
        if (log.isNotEmpty()) LogBox(log)

        Spacer(Modifier.height(8.dp))
    }
}

/**
 * The model download, drawn over the swapped preview.
 *
 * Only ever composed when the files are actually missing, so there is no button sitting
 * around inviting a 275 MB transfer nobody needs.
 */
@Composable
private fun DownloadOverlay(onDownload: () -> Unit) {
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
                        "%d / %d MB   file %d of %d".format(
                            ModelDownload.doneBytes / 1048576,
                            ModelDownload.totalBytes / 1048576,
                            ModelDownload.fileIndex, ModelDownload.fileCount,
                        ),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                else -> {
                    Text("Models required",
                         style = MaterialTheme.typography.titleSmall)
                    Spacer(Modifier.height(4.dp))
                    Text(
                        ModelDownload.error ?: "The face models are not on this device yet.",
                        style = MaterialTheme.typography.bodySmall,
                        textAlign = TextAlign.Center,
                        color = if (ModelDownload.error != null)
                            MaterialTheme.colorScheme.error
                        else MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(Modifier.height(12.dp))
                    Button(onDownload, shape = RoundedCornerShape(14.dp)) {
                        Text(if (ModelDownload.error != null) "Retry download" else "Download models")
                    }
                }
            }
        }
    }
}
