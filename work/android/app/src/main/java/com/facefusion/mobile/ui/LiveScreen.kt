package com.facefusion.mobile.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * The front camera, swapped, live. Dev builds only.
 *
 * Deliberately not a third copy of the Swap screen: no trim, no frame rate, no output file,
 * no Advanced. What is here is what a live feed can actually act on -- a source face, a
 * start/stop, and the frame rate it is achieving.
 */
@Composable
fun LiveScreen(
    sourceThumb: Bitmap?,
    onPickSource: () -> Unit,
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
    // put the "Use my Swap settings" switch behind the navigation bar, where the only clue
    // it existed was a few pixels of its track poking out under the Start button.
    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("Live", style = MaterialTheme.typography.titleMedium)

        // The feed. 4:5 rather than the sensor's own ratio so the pane is a stable size
        // whatever resolution CameraX actually granted -- the image letterboxes inside it.
        Box(
            Modifier
                .fillMaxWidth()
                // The FRAME's own ratio once one has arrived, so the feed fills the pane
                // instead of sitting in grey bands. 3:4 until then, which is roughly what
                // a front camera returns and stops the pane resizing under the first frame.
                .aspectRatio(frame?.let { it.width.toFloat() / it.height } ?: (3f / 4f))
                .clip(RoundedCornerShape(12.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant),
            contentAlignment = Alignment.Center,
        ) {
            if (frame != null) {
                Image(
                    bitmap = frame.asImageBitmap(),
                    contentDescription = null,
                    contentScale = ContentScale.Fit,
                    // ⚠ THE MIRROR LIVES HERE AND NOWHERE ELSE. The pipeline sees the true
                    // image so the detector gets a face the right way round; only what is
                    // drawn is flipped, which is what every selfie camera does and what
                    // makes moving left move left.
                    modifier = Modifier.fillMaxSize().graphicsLayer(scaleX = -1f),
                )
            } else {
                Text(
                    when {
                        !modelsReady -> "Models not installed"
                        sourceThumb == null -> "Pick a source face"
                        !running -> "Ready"
                        else -> "Starting camera..."
                    },
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
                        "%.1f fps  %s".format(fps, if (faces > 0) "$faces face" else "no face"),
                        color = MaterialTheme.colorScheme.inverseOnSurface,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 11.sp,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                    )
                }
            }
        }

        // Source face, and the run control beside it.
        Row(verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Box(
                Modifier
                    .size(72.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant),
                contentAlignment = Alignment.Center,
            ) {
                if (sourceThumb != null)
                    Image(sourceThumb.asImageBitmap(), null,
                          Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
                else Text("+", style = MaterialTheme.typography.titleLarge)
            }
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                OutlinedButton(onClick = onPickSource, enabled = !running) {
                    Text(if (sourceThumb == null) "Pick source face" else "Change source")
                }
                Text(
                    // The source cannot change mid-run: setSource re-detects and re-embeds,
                    // and doing that under the pump would swap identity halfway through a
                    // frame the camera is still filling.
                    if (running) "Stop to change the source"
                    else "The face to swap ONTO the live feed",
                    style = MaterialTheme.typography.bodySmall, fontSize = 11.sp,
                )
            }
        }

        Button(
            onClick = onToggleRun,
            enabled = modelsReady && sourceThumb != null,
            modifier = Modifier.fillMaxWidth(),
        ) { Text(if (running) "Stop" else "Start") }

        // The override. Off means the forced preset: tracking on, enhancer off, boost 1x,
        // no lip sync -- so the frame rate is predictable and nobody arrives at 3 fps by
        // way of the enhancer at 1024 without knowing they asked for it.
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Use my Swap settings", style = MaterialTheme.typography.bodyMedium)
                Text(
                    if (useMySettings) "Whatever Advanced is set to, including the enhancer"
                    else "Fast preset: tracking on, enhancer off, 256, no lip sync",
                    style = MaterialTheme.typography.bodySmall, fontSize = 11.sp,
                )
            }
            Switch(checked = useMySettings, onCheckedChange = onUseMySettings, enabled = !running)
        }

        if (note != null)
            Text(note, style = MaterialTheme.typography.bodySmall,
                 color = MaterialTheme.colorScheme.error)

        Spacer(Modifier.height(8.dp))
    }
}
