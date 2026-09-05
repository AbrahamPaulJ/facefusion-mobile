package com.facefusion.mobile.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.facefusion.mobile.R

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
        // Above the feed, and the same pane the Swap screen uses: full width while empty,
        // because an empty pane is a call to action, and collapsed to a small square once
        // filled, because the source is one face that never changes during a run.
        //
        // ⚠ Not tappable while running. setSource re-detects and re-embeds, and doing that
        // under the pump would change identity halfway through a frame the camera is still
        // filling.
        val sourceBox = 104.dp
        Box(if (sourceThumb != null) Modifier.width(sourceBox) else Modifier.fillMaxWidth()) {
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
        Box(
            Modifier
                .fillMaxWidth()
                .aspectRatio(raw.coerceIn(3f / 4f, 4f / 3f))
                .clip(RoundedCornerShape(12.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant),
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
                    modifier = Modifier.fillMaxSize().graphicsLayer(scaleX = -1f),
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

        Button(
            onClick = onToggleRun,
            enabled = modelsReady && sourceThumb != null,
            modifier = Modifier.fillMaxWidth(),
        ) { Text(stringResource(if (running) R.string.live_stop else R.string.live_start)) }

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
