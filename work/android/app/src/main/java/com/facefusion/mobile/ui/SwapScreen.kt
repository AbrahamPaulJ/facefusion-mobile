package com.facefusion.mobile.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
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
import com.facefusion.mobile.SwapOptions
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
    durationMs: Long,
    trimStartMs: Float,
    trimEndMs: Float,
    onTrimChange: (Float, Float) -> Unit,
    fmt: (Float) -> String,
    preview: PreviewUi,
    onRefreshPreview: () -> Unit,
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
    hasOutput: Boolean,
    saved: Boolean,
    savedPath: String?,
    onPickSource: () -> Unit,
    onPickTarget: () -> Unit,
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
    val paneHeight = ((screenH - 470) / 2).coerceIn(140, 320).dp
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
        // Both full width and stacked, so the pair reads as a before/after of ONE frame.
        // The originals used to be a 56 dp thumbnail and a leftover-space preview, at which
        // size a face swap cannot actually be judged.
        PreviewPane(
            label = if (preview.timeLabel.isEmpty()) "Original" else "Original  ${preview.timeLabel}",
            height = paneHeight,
            bitmap = preview.original,
            placeholder = when {
                run.preparing -> "Reading video..."
                hasTarget -> "Seeking..."
                else -> "Add a target video"
            },
            // The pane IS the picker. A separate full-width button said the same thing
            // twice and cost a row of height the wordmark needed.
            onClick = if (idle) onPickTarget else null,
            actionIcon = if (hasTarget) null else Icons.Default.Add,
        ) {
            if (hasTarget)
                TextButton(onPickTarget, enabled = idle) { Text("Change") }
        }

        PreviewPane(
            label = "Swapped",
            height = paneHeight,
            bitmap = preview.swapped,
            placeholder = when {
                modelsMissing -> ""
                preview.note != null -> preview.note
                preview.busy && !preview.warm -> "Loading models, this takes a few seconds..."
                preview.busy -> "Swapping this frame..."
                !hasSource || !hasTarget -> "Pick a source face and a target video"
                !preview.warm -> "Tap refresh to preview this frame"
                else -> "Tap refresh"
            },
            // The download lives here rather than in a bar of its own: this is the pane
            // that cannot draw anything without the models, so it is where their absence
            // is already visible.
            overlay = if (modelsMissing) { { DownloadOverlay(onDownload) } } else null,
        ) {
            if (preview.busy) {
                CircularProgressIndicator(
                    Modifier.size(18.dp),
                    strokeWidth = 2.dp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                IconButton(
                    onClick = onRefreshPreview,
                    enabled = idle && hasSource && hasTarget && !modelsMissing,
                    modifier = Modifier.size(28.dp),
                ) {
                    Icon(
                        Icons.Default.Refresh, "Refresh preview",
                        Modifier.size(18.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
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
                        // Keep at least a third of a second, so the encoder always gets a frame.
                        onTrimChange(r.start, maxOf(r.endInclusive, r.start + 333f))
                    },
                    valueRange = 0f..durationMs.toFloat(),
                    enabled = idle,
                )
                val estFrames = ((trimEndMs - trimStartMs) / 1000f * 30f).roundToInt()
                Text(
                    "$estFrames frames of ${fmt(durationMs.toFloat())}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        // ---------------------------------------------------------------- run
        // One button, two jobs: a separate Cancel would sit dead for the entire time the
        // only thing you can do is start a swap.
        Button(
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
                FaceSwapperCard(opts, onOptsChange, openCard == "swapper",
                                { onToggleCard("swapper") }, inswapperAvailable = hasInswapper)
                FaceMaskerCard(opts, onOptsChange, openCard == "masker", { onToggleCard("masker") })
                FaceDetectorCard(opts, onOptsChange, openCard == "detector",
                                 { onToggleCard("detector") })
                // Only when gpen_<tier>.bin is actually on the device.
                if (hasEnhancer) {
                    FaceEnhancerCard(opts, onOptsChange, openCard == "enhancer",
                                     { onToggleCard("enhancer") })
                }
                if (opts != SwapOptions()) {
                    TextButton(
                        onClick = { onOptsChange(SwapOptions()) },
                        modifier = Modifier.align(Alignment.End),
                    ) { Text("Reset to FaceFusion defaults") }
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
