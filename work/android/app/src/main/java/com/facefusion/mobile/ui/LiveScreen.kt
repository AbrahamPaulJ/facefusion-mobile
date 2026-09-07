package com.facefusion.mobile.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.facefusion.mobile.R
import kotlinx.coroutines.delay

/**
 * The front camera, swapped, live.
 *
 * Deliberately not a third copy of the Swap screen: no trim, no frame rate, no output file,
 * no per-processor sheets. What is here is what a live feed can actually act on -- a source
 * face, a start/stop, and the frame rate it is achieving.
 *
 * The source is picked through the SAME [PreviewPane] the Swap screen uses, at the top, for
 * the same reason it sits at the top there: it is an input, it is chosen by tapping its own
 * frame, and a second way of picking the same thing is a second thing to learn.
 */
@Composable
fun LiveScreen(
    sourceThumb: Bitmap?,
    sourceCount: Int = 0,
    activeSource: Int = 0,
    onSelectSource: (Int) -> Unit = {},
    onPickSource: () -> Unit,
    onClearSource: () -> Unit,
    onCaptureSource: () -> Unit,
    frame: Bitmap?,
    running: Boolean,
    onToggleRun: () -> Unit,
    fps: Double,
    faces: Int,
    useMySettings: Boolean,
    onUseMySettings: (Boolean) -> Unit,
    note: String?,
    modelsReady: Boolean,
    /** Start the model download. Live is reachable before any model exists. */
    onDownload: () -> Unit,
    /** Which lens is bound. Decides the MIRROR, and nothing else on this screen. */
    frontCamera: Boolean = true,
    /** Flip the lens. Stops and restarts the pump when it is running. */
    onSwitchCamera: () -> Unit = {},
    /** Whether a recording is in flight -- roadmap 13b. */
    recording: Boolean = false,
    microphone: Boolean = false,
    finalizing: Boolean = false,
    onMicrophoneChange: (Boolean) -> Unit = {},
    largestOnly: Boolean = false,
    onLargestOnlyChange: (Boolean) -> Unit = {},
    swapEnabled: Boolean = true,
    onToggleSwapEnabled: () -> Unit = {},
    /** Assign-per-person mode: OFF is default behaviour, ON lets each face keep a source. */
    assignMode: Boolean = false,
    onToggleAssignMode: () -> Unit = {},
    /** A tap on the feed, as DISPLAY bitmap coordinates (mirror and crop already undone). */
    onAssignFace: (Float, Float) -> Unit = { _, _ -> },
    /**
     * The last assigned face, as liveFrame returned it: x0, y0, x1, y1, source -- in
     * DISPLAY bitmap coordinates -- plus a nonce that changes with it so the fade can
     * retime, and how many people are assigned (for the Clear affordance).
     */
    assignBox: FloatArray? = null,
    assignNonce: Int = 0,
    assignCount: Int = 0,
    /**
     * The SELECTED person (assign mode): x0, y0, x1, y1, source -- DISPLAY bitmap
     * coordinates, polled every shot so the highlight follows them. The selected person
     * follows the source chip until an empty tap deselects them (they keep their last
     * source). Null when nobody is selected.
     */
    selectionBox: FloatArray? = null,
    onClearAssignments: () -> Unit = {},
    /** Start or finish recording the feed. Only meaningful while it is running. */
    onToggleRecord: () -> Unit = {},
) {
    // SCROLLS. Without this the controls below the feed are simply clipped: the first build
    // put the settings switch behind the navigation bar, where the only clue it existed was
    // a few pixels of its track poking out under the Start button.
    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text(stringResource(R.string.live_title),
             style = MaterialTheme.typography.titleMedium)

        // ---------------------------------------------------------------- source
        //
        // Above the feed, and the same pane the Swap screen uses: full width always, and
        // collapsed in HEIGHT once filled -- the source is one face that never changes
        // during a run, so the pane it fills need not be tall. Not narrow, though: the
        // caption row has to hold "SOURCE FACE" beside a camera and a delete button.
        //
        // ⚠ Not tappable while running. setSource re-detects and re-embeds, and doing that
        // under the pump would change identity halfway through a frame the camera is still
        // filling.
        val sourceBox = 104.dp
        Box(Modifier.fillMaxWidth()) {
            PreviewPane(
                label = stringResource(R.string.swap_source_face),
                height = if (sourceThumb != null) sourceBox else 220.dp,
                bitmap = sourceThumb,
                placeholder = stringResource(R.string.swap_source_pick),
                onClick = if (!running) onPickSource else null,
                actionIcon = if (sourceThumb != null) null else Icons.Default.Add,
                zoom = null,
            ) {
                if (!running) {
                    IconButton(onCaptureSource, Modifier.size(28.dp)) {
                        Icon(painterResource(R.drawable.ic_photo_camera),
                             stringResource(R.string.swap_capture_source), Modifier.size(16.dp))
                    }
                }
                if (sourceThumb != null && !running) {
                    IconButton(onClearSource, Modifier.size(28.dp)) {
                        Icon(Icons.Default.Delete,
                             stringResource(R.string.swap_remove_source),
                             Modifier.size(16.dp))
                    }
                }
            }
        }
        Text(
            if (running) stringResource(R.string.live_stop_to_change)
            else stringResource(R.string.live_source_hint),
            style = MaterialTheme.typography.bodySmall, fontSize = 11.sp,
        )

        if (sourceCount > 0) {
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.fillMaxWidth()) {
                repeat(sourceCount) { index ->
                    FilterChip(
                        selected = index == activeSource,
                        onClick = { onSelectSource(index) },
                        label = { Text("Source ${index + 1}") },
                        // The whole point of the multi-source mode: switch ON THE FLY,
                        // including while a recording is in flight -- the native side
                        // reads the active slot per frame, so the file simply changes
                        // face at the switch. Only finalization is locked.
                        enabled = !finalizing,
                    )
                }
            }
        }

        // ---------------------------------------------------------------- the feed
        //
        // ⚠ The pane is CAPPED at 3:4, not given the frame's own ratio.
        //
        // The camera hands back 9:16 (0.5625). A pane of that shape is taller than any
        // phone camera app's viewfinder -- it ran past the fold and pushed Start off the
        // screen -- and it was reported as simply "too long". Capping at 3:4 and cropping
        // to fill is what a camera app does: the face is centred, so what leaves the frame
        // is the ceiling and the floor.
        //
        // Crop, not Fit, for the same reason: Fit inside a wider box would letterbox the
        // 9:16 image into grey side bars, which trades one ugly shape for another.
        val raw = frame?.let { it.width.toFloat() / it.height } ?: (3f / 4f)
        // Frame dimensions, captured ONCE per composition as the gesture keys -- see the
        // pointerInput below. Constant within a session (the pump reuses the resolution),
        // which is exactly why they are safe as keys where the Bitmap reference is not.
        val fw = (frame?.width ?: 0).toFloat()
        val fh = (frame?.height ?: 0).toFloat()
        // The confirmation box around a just-assigned face, faded out after a beat. The
        // nonce from the caller retimes it: a new assignment restarts the fade.
        var assignFade by remember { mutableStateOf(false) }
        LaunchedEffect(assignNonce) {
            if (assignNonce > 0) { assignFade = true; delay(1500); assignFade = false }
        }
        BoxWithConstraints(
            Modifier
                .fillMaxWidth()
                .aspectRatio(raw.coerceIn(3f / 4f, 4f / 3f))
                .clip(RoundedCornerShape(12.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant)
                // ASSIGN MODE: the feed is a touch surface. The tap is mapped from the
                // pane (which CROPS the frame and, on the front lens, draws it MIRRORED)
                // into DISPLAY bitmap coordinates -- what the pipeline sees -- so the
                // native side can resolve it against the pre-swap detections.
                // ⚠ NOT keyed on `frame`: that Bitmap reference changes every shot, so
                // keying on it tore the gesture detector down and rebuilt it every frame,
                // eating the occasional tap in the middle of a recomposition. The frame
                // DIMENSIONS are constant for a session, so they are the key: the detector
                // restarts once when live starts (0 -> real size) and not again, and the
                // mapping math below only needs them.
                .pointerInput(assignMode, running, frontCamera, fw, fh) {
                    if (assignMode && running && frame != null) {
                        detectTapGestures { off ->
                            val bw = size.width.toFloat(); val bh = size.height.toFloat()
                            val s = maxOf(bw / fw, bh / fh)
                            val ox = (bw - fw * s) / 2f; val oy = (bh - fh * s) / 2f
                            var bx = (off.x - ox) / s
                            val by = (off.y - oy) / s
                            if (frontCamera) bx = fw - bx
                            onAssignFace(bx.coerceIn(0f, fw), by.coerceIn(0f, fh))
                        }
                    }
                },
            contentAlignment = Alignment.Center,
        ) {
            if (frame != null) {
                Image(
                    bitmap = frame.asImageBitmap(),
                    contentDescription = null,
                    contentScale = ContentScale.Crop,
                    // ⚠ THE MIRROR LIVES HERE AND NOWHERE ELSE. The pipeline sees the true
                    // image so the detector gets a face the right way round; only what is
                    // drawn is flipped, which is what every selfie camera does and what
                    // makes moving left move left.
                    //
                    // ⚠ FRONT ONLY. The back camera is not a mirror -- it points at what
                    // the user is looking at, and flipping it puts text backwards and
                    // moves the world the wrong way. Mirroring it would not touch the
                    // swap, which is what makes the mistake hard to read: the pipeline
                    // gets the true image either way, so the bug would look like the
                    // model failing when it is only the view.
                    modifier = Modifier
                        .fillMaxSize()
                        .graphicsLayer(scaleX = if (frontCamera) -1f else 1f),
                )
            } else {
                Text(
                    stringResource(when {
                        !modelsReady -> R.string.live_models_missing
                        sourceThumb == null -> R.string.live_pick_source
                        !running -> R.string.live_ready
                        else -> R.string.live_starting
                    }),
                    style = MaterialTheme.typography.bodyMedium,
                )
            }

            // The SAME overlay the Swap screen draws over its swapped pane -- progress,
            // byte counter, error and retry included. Live is a tab, so a fresh install can
            // land here first: it used to say "Models not installed" over a Start button
            // that could never enable, with the only way out on a screen it did not mention.
            if (!modelsReady) DownloadOverlay(onDownload)

            // THE LENS, top-left, opposite the frame rate. A chip rather than an icon
            // because material-icons-extended is not a dependency here and, more usefully,
            // because a word says which camera is live -- an icon only says that it can be
            // changed. Drawn whether or not the pump is running, since the choice is worth
            // making before pressing Start.
            Surface(
                onClick = onSwitchCamera,
                color = MaterialTheme.colorScheme.scrim.copy(alpha = 0.55f),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.align(Alignment.TopStart).padding(8.dp),
            ) {
                Text(
                    stringResource(if (frontCamera) R.string.live_lens_front
                                   else R.string.live_lens_back),
                    color = MaterialTheme.colorScheme.inverseOnSurface,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 11.sp,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                )
            }

            // The one thing that does belong over the picture: whether this is being
            // recorded. It is the state a user must be able to check without looking away
            // from what they are pointing the camera at.
            if (recording) {
                Surface(
                    color = FfRed,
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.align(Alignment.BottomEnd).padding(8.dp),
                ) {
                    Text(
                        stringResource(R.string.live_rec_badge),
                        color = Color.White,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 11.sp,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                    )
                }
            }

            // The confirmation around the face just assigned: box plus "Source N", mapped
            // from DISPLAY bitmap coordinates through the same crop+mirror the image above
            // draws with, so the outline sits on the person the user tapped.
            if (assignFade && assignBox != null && assignBox.size >= 5 && frame != null) {
                val b = assignBox
                Canvas(Modifier.fillMaxSize()) {
                    val fw = frame.width.toFloat(); val fh = frame.height.toFloat()
                    val bw = size.width.toFloat(); val bh = size.height.toFloat()
                    val s = maxOf(bw / fw, bh / fh)
                    val ox = (bw - fw * s) / 2f; val oy = (bh - fh * s) / 2f
                    val l = ox + (if (frontCamera) fw - b[2] else b[0]) * s
                    val r = ox + (if (frontCamera) fw - b[0] else b[2]) * s
                    val t = oy + b[1] * s
                    val bo = oy + b[3] * s
                    drawRect(FfRed, topLeft = Offset(l, t),
                             size = androidx.compose.ui.geometry.Size(r - l, bo - t),
                             style = androidx.compose.ui.graphics.drawscope.Stroke(3.dp.toPx()))
                    val label = "Source ${b[4].toInt() + 1}"
                    val paint = android.graphics.Paint().apply {
                        color = android.graphics.Color.RED
                        textSize = 13.dp.toPx()
                        isAntiAlias = true
                        typeface = android.graphics.Typeface.DEFAULT_BOLD
                    }
                    drawContext.canvas.nativeCanvas.drawText(
                        label, l + 6.dp.toPx(),
                        if (t - 6.dp.toPx() > 0f) t - 6.dp.toPx() else t + 14.dp.toPx(), paint)
                }
            }

            // The SELECTED person's persistent highlight -- white, so it reads as the
            // active target against the red one-shot confirmation above. Follows the
            // person frame to frame; the label shows the source they currently have
            // (which a selected person updates when the chip changes). Same crop+mirror
            // mapping as the confirmation box.
            if (selectionBox != null && selectionBox.size >= 5 && frame != null) {
                val b = selectionBox
                Canvas(Modifier.fillMaxSize()) {
                    val fw = frame.width.toFloat(); val fh = frame.height.toFloat()
                    val bw = size.width.toFloat(); val bh = size.height.toFloat()
                    val s = maxOf(bw / fw, bh / fh)
                    val ox = (bw - fw * s) / 2f; val oy = (bh - fh * s) / 2f
                    val l = ox + (if (frontCamera) fw - b[2] else b[0]) * s
                    val r = ox + (if (frontCamera) fw - b[0] else b[2]) * s
                    val t = oy + b[1] * s
                    val bo = oy + b[3] * s
                    drawRect(Color.White, topLeft = Offset(l, t),
                             size = androidx.compose.ui.geometry.Size(r - l, bo - t),
                             style = androidx.compose.ui.graphics.drawscope.Stroke(2.dp.toPx()))
                    val label = "Source ${b[4].toInt() + 1}"
                    val paint = android.graphics.Paint().apply {
                        color = android.graphics.Color.WHITE
                        textSize = 13.dp.toPx()
                        isAntiAlias = true
                        typeface = android.graphics.Typeface.DEFAULT_BOLD
                    }
                    drawContext.canvas.nativeCanvas.drawText(
                        label, l + 6.dp.toPx(),
                        if (t - 6.dp.toPx() > 0f) t - 6.dp.toPx() else t + 14.dp.toPx(), paint)
                }
            }

            // Frame rate over the feed, where it is read while looking at the result rather
            // than after it. Only while running: a stale rate on a stopped feed is a lie.
            if (running) {
                Surface(
                    color = MaterialTheme.colorScheme.scrim.copy(alpha = 0.55f),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.align(Alignment.TopEnd).padding(8.dp),
                ) {
                    Text(
                        stringResource(R.string.live_stat, "%.1f".format(fps),
                            if (faces > 0) stringResource(R.string.live_faces, faces)
                            else stringResource(R.string.live_no_face)),
                        color = MaterialTheme.colorScheme.inverseOnSurface,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 11.sp,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                    )
                }
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = onToggleRun,
                enabled = modelsReady && sourceThumb != null,
                modifier = Modifier.weight(1f),
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                    contentColor = MaterialTheme.colorScheme.onBackground,
                    disabledContainerColor = MaterialTheme.colorScheme.surface,
                    disabledContentColor = MaterialTheme.colorScheme.onSurfaceVariant,
                ),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
            ) { Text(stringResource(if (running) R.string.live_stop else R.string.live_start)) }
            // RECORD, beside Start rather than over the feed: it writes a file, which is
            // the kind of thing that belongs with the other button that commits something,
            // not floating over the picture as an ornament. Only enabled while the pump is
            // running -- arming a recorder before the camera produces a zero-frame file.
            OutlinedButton(
                onClick = onToggleRecord,
                enabled = running && !finalizing,
                modifier = Modifier.weight(1f),
            ) {
                Text(stringResource(if (recording) R.string.live_rec_stop
                                    else R.string.live_rec_start),
                     color = if (recording) FfRed else Color.Unspecified)
            }
        }

        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(stringResource(R.string.live_mic), style = MaterialTheme.typography.bodyMedium)
                Text(stringResource(if (microphone) R.string.live_mic_on else R.string.live_mic_off),
                     style = MaterialTheme.typography.bodySmall)
            }
            Switch(checked = microphone, onCheckedChange = onMicrophoneChange,
                   enabled = !recording && !finalizing)
        }

        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Face swap", style = MaterialTheme.typography.bodyMedium)
                Text(if (swapEnabled) "Swapping active" else "Swapping disabled",
                     style = MaterialTheme.typography.bodySmall)
            }
            Switch(checked = swapEnabled,
                   onCheckedChange = { onToggleSwapEnabled() },
                   // Also while recording: disabling the swap mid-file is the other half
                   // of the on-the-fly mode, and the recorded feed simply keeps the
                   // unswapped frames for as long as it is off.
                   enabled = !finalizing)
        }

        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Target faces", style = MaterialTheme.typography.bodyMedium)
                // Mutually exclusive with assign per person (both choose which face gets
                // which source). While assign is on the selector is pinned to "all
                // faces", so the switch is locked and says so.
                Text(stringResource(
                    if (assignMode) R.string.live_target_locked
                    else if (largestOnly) R.string.live_target_one
                    else R.string.live_target_all),
                     style = MaterialTheme.typography.bodySmall)
            }
            Switch(checked = !largestOnly,
                   onCheckedChange = { onLargestOnlyChange(!it) },
                   enabled = !assignMode && !recording && !finalizing)
        }

        // How assign mode works: select the source FIRST, then tap the person -- the
        // order the tap captures. A help dialog is the one place the flow can be stated
        // without cluttering the row; it is available whether or not the mode is on.
        var showAssignHelp by rememberSaveable { mutableStateOf(false) }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(stringResource(R.string.live_assign_title),
                     style = MaterialTheme.typography.bodyMedium)
                Text(stringResource(if (assignMode) R.string.live_assign_on
                                    else R.string.live_assign_off),
                     style = MaterialTheme.typography.bodySmall, fontSize = 11.sp)
            }
            IconButton(
                onClick = { showAssignHelp = true },
                modifier = Modifier.size(32.dp),
            ) {
                Icon(
                    Icons.Filled.Info,
                    contentDescription = stringResource(R.string.live_assign_help),
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.size(18.dp),
                )
            }
            // Clear lives with the switch: turning the mode off means default behaviour
            // (each face takes the selected source) for the whole session, so the
            // assignments are only meaningful -- and only shown -- while it is on.
            if (assignMode && assignCount > 0) {
                TextButton(onClick = onClearAssignments) {
                    Text(stringResource(R.string.live_assign_clear))
                }
            }
            Switch(checked = assignMode,
                   onCheckedChange = { onToggleAssignMode() },
                   // A face must be selectable before it can be assigned, so the mode
                   // cannot be turned on mid-recording either -- the chips are locked
                   // for the same reason.
                   enabled = running && !recording && !finalizing && sourceCount > 0)
        }
        if (showAssignHelp) {
            AlertDialog(
                onDismissRequest = { showAssignHelp = false },
                icon = { Icon(Icons.Filled.Info, contentDescription = null) },
                title = { Text(stringResource(R.string.live_assign_help_title)) },
                text = { Text(stringResource(R.string.live_assign_help_body)) },
                confirmButton = {
                    TextButton(onClick = { showAssignHelp = false }) {
                        Text(stringResource(R.string.live_assign_help_gotit))
                    }
                },
            )
        }

        // ---------------------------------------------------------------- fast mode
        //
        // The SAME switch as before, stated the way round it is actually used. It used to
        // read "Use my Swap settings", off by default -- so the recommended configuration
        // was the negative of an option, and the thing being turned off had no name. Now
        // the preset has the name, it is on by default, and the experimental path is the
        // one that asks before it is taken.
        var confirmSlow by rememberSaveable { mutableStateOf(false) }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(stringResource(R.string.live_fast_mode),
                     style = MaterialTheme.typography.bodyMedium)
                Text(
                    stringResource(if (useMySettings) R.string.live_fast_off
                                   else R.string.live_fast_on),
                    style = MaterialTheme.typography.bodySmall, fontSize = 11.sp,
                )
            }
            Switch(
                // Inverted: fast mode ON is useMySettings OFF.
                checked = !useMySettings,
                onCheckedChange = { wantFast ->
                    // Turning it ON needs no ceremony -- it is the safe direction, and the
                    // configuration everything about this tab was measured on. Turning it
                    // OFF is the one that can take the feed to single figures, so that is
                    // the one that explains itself first.
                    if (wantFast) onUseMySettings(false) else confirmSlow = true
                },
                enabled = !running,
            )
        }

        if (confirmSlow) {
            AlertDialog(
                onDismissRequest = { confirmSlow = false },
                title = { Text(stringResource(R.string.live_fast_confirm_title)) },
                text = { Text(stringResource(R.string.live_fast_confirm_body)) },
                confirmButton = {
                    TextButton({ confirmSlow = false; onUseMySettings(true) }) {
                        Text(stringResource(R.string.live_fast_confirm_ok))
                    }
                },
                dismissButton = {
                    TextButton({ confirmSlow = false }) {
                        Text(stringResource(R.string.common_cancel))
                    }
                },
            )
        }

        if (note != null)
            Text(note, style = MaterialTheme.typography.bodySmall,
                 color = MaterialTheme.colorScheme.error)

        Spacer(Modifier.height(8.dp))
    }
}
