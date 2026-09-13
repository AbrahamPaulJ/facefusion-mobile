package com.facefusion.mobile.ui

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.content.pm.ActivityInfo
import android.graphics.Bitmap
import android.view.ViewGroup
import android.view.WindowManager
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.path
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.compose.ui.window.DialogWindowProvider
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.facefusion.mobile.R
import kotlinx.coroutines.delay

/**
 * Material's `pause`, `fullscreen_exit` and `screen_rotation`, declared here rather than
 * depended on -- the same trade [IconDownload] documents. `material-icons-core` carries a
 * fixed short list and none of the three is on it; the rest live in
 * `material-icons-extended`, thousands of vectors for three. Google's own path data,
 * unchanged, on the standard 24 dp viewport.
 */
val IconPause: ImageVector = ImageVector.Builder(
    name = "pause",
    defaultWidth = 24.dp, defaultHeight = 24.dp,
    viewportWidth = 24f, viewportHeight = 24f,
).apply {
    path(fill = SolidColor(Color.Black)) {
        moveTo(6f, 19f); horizontalLineToRelative(4f); verticalLineTo(5f)
        horizontalLineTo(6f); verticalLineToRelative(14f); close()
        moveTo(14f, 5f); verticalLineToRelative(14f); horizontalLineToRelative(4f)
        verticalLineTo(5f); horizontalLineToRelative(-4f); close()
    }
}.build()

val IconRotate: ImageVector = ImageVector.Builder(
    name = "screen_rotation",
    defaultWidth = 24.dp, defaultHeight = 24.dp,
    viewportWidth = 24f, viewportHeight = 24f,
).apply {
    path(fill = SolidColor(Color.Black)) {
        moveTo(16.48f, 2.52f)
        curveToRelative(3.27f, 1.55f, 5.61f, 4.72f, 5.97f, 8.48f)
        horizontalLineToRelative(1.5f)
        curveTo(23.44f, 4.84f, 18.29f, 0f, 12f, 0f)
        lineToRelative(-0.66f, 0.03f)
        lineToRelative(3.81f, 3.81f)
        lineToRelative(1.33f, -1.32f)
        close()
        moveTo(10.23f, 1.75f)
        curveToRelative(-0.59f, -0.59f, -1.54f, -0.59f, -2.12f, 0f)
        lineTo(1.75f, 8.11f)
        curveToRelative(-0.59f, 0.59f, -0.59f, 1.54f, 0f, 2.12f)
        lineToRelative(12.02f, 12.02f)
        curveToRelative(0.59f, 0.59f, 1.54f, 0.59f, 2.12f, 0f)
        lineToRelative(6.36f, -6.36f)
        curveToRelative(0.59f, -0.59f, 0.59f, -1.54f, 0f, -2.12f)
        lineTo(10.23f, 1.75f)
        close()
        moveTo(14.83f, 21.19f)
        lineTo(2.81f, 9.17f)
        lineToRelative(6.36f, -6.36f)
        lineToRelative(12.02f, 12.02f)
        lineToRelative(-6.36f, 6.36f)
        close()
        moveTo(7.52f, 21.48f)
        curveTo(4.25f, 19.94f, 1.91f, 16.76f, 1.55f, 13f)
        horizontalLineTo(0.05f)
        curveTo(0.56f, 19.16f, 5.71f, 24f, 12f, 24f)
        lineToRelative(0.66f, -0.03f)
        lineToRelative(-3.81f, -3.81f)
        lineToRelative(-1.33f, 1.32f)
        close()
    }
}.build()

/**
 * The target clip, swapped as it plays, on the whole screen.
 *
 * ⚠ It shows SWAPPED frames that no run produced and no file holds, which makes it a
 * processing path like any other. It shipped dev-only until `MainActivity.startPlayer`
 * grew the check every other path has; that check, not this file, is what puts it on both
 * lines. This file stays pure UI on purpose -- it draws, it does not decide.
 *
 * ⚠ The gate class's NAME is deliberately not spelled anywhere in this file. The dev
 * line is verified by grepping this tree for that name and expecting nothing back, and
 * the grep reads comments as well as code -- so a file that is identical on both lines
 * must never be the thing that fails it. The first draft of this very paragraph did,
 * twice, which is how the rule earned the warning.
 *
 * The controls hide themselves after a few seconds of playback and come back on a tap,
 * which is the one convention every video player on the phone already shares. They do NOT
 * hide while paused: a paused player with no controls looks broken.
 */
@Composable
fun LivePlayerOverlay(
    frame: Bitmap?,
    positionMs: Int,
    durationMs: Int,
    playing: Boolean,
    fps: Double,
    faces: Int,
    dropped: Int,
    /** A finished sentence from the Activity, or null. Shown over the picture. */
    note: String?,
    /** A seek is being resolved: the picture is the old one and the sound is off. */
    seeking: Boolean = false,
    onPlayPause: () -> Unit,
    onSeek: (Int) -> Unit,
    onClose: () -> Unit,
) {
    var showControls by remember { mutableStateOf(true) }
    var forceLandscape by remember { mutableStateOf(false) }
    // While a finger is on the bar the position comes from the FINGER, not from the pump.
    // Without this the thumb fights the playhead and jumps back under the drag.
    var scrubMs by remember { mutableStateOf<Int?>(null) }

    val activity = LocalContext.current.findActivity()

    // ⚠ Restored on the way out, ALWAYS. Leaving the Activity pinned to landscape after the
    // player closes is the kind of state that outlives the feature that set it.
    DisposableEffect(forceLandscape, activity) {
        activity?.requestedOrientation =
            if (forceLandscape) ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
            else ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
        onDispose {
            activity?.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
        }
    }

    // ⚠ The countdown does not start until a frame exists. Opening the player pays for an
    // init and a model load, and hiding the controls during that wait left a black screen
    // with nothing on it at all -- including no way out but the system back gesture.
    LaunchedEffect(showControls, playing, seeking, frame != null) {
        if (showControls && playing && !seeking && frame != null) {
            delay(4500); showControls = false
        }
    }

    Dialog(
        onDismissRequest = onClose,
        // usePlatformDefaultWidth = false is what makes a Dialog able to be the whole
        // screen at all; without it the window is inset to a tablet-ish dialog width and
        // "fullscreen" is a black card in the middle of one.
        properties = DialogProperties(usePlatformDefaultWidth = false),
    ) {
        val view = LocalView.current
        val dialogWindow = (view.parent as? DialogWindowProvider)?.window
        LaunchedEffect(dialogWindow) {
            dialogWindow?.let { w ->
                w.setLayout(ViewGroup.LayoutParams.MATCH_PARENT,
                            ViewGroup.LayoutParams.MATCH_PARENT)
                // The phone must not sleep in the middle of a clip.
                w.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                // Edge to edge, then hide the bars. Set on the DIALOG's window, not the
                // Activity's: a dialog gets its own, and decorating the Activity's leaves
                // the status bar sitting on top of the video.
                WindowCompat.setDecorFitsSystemWindows(w, false)
                WindowInsetsControllerCompat(w, w.decorView).apply {
                    hide(WindowInsetsCompat.Type.systemBars())
                    systemBarsBehavior =
                        WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
                }
            }
        }

        // Computed INSIDE the Dialog: a dialog carries its own window and its own insets,
        // and reading them from the Activity's would describe a different window.
        val edges = playerEdgeInsets()

        Box(
            Modifier
                .fillMaxSize()
                .background(Color.Black)
                .pointerInput(Unit) {
                    detectTapGestures { showControls = !showControls }
                },
            contentAlignment = Alignment.Center,
        ) {
            // ContentScale.Fit, never Crop: this is a preview of what the run would
            // produce, and cropping it would hide exactly the edges a face can sit on.
            if (frame != null) Image(
                bitmap = frame.asImageBitmap(),
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit,
            ) else Text(
                stringResource(R.string.player_starting),
                color = Color.White.copy(alpha = 0.7f),
                style = MaterialTheme.typography.bodyMedium,
            )

            // ⚠ Shown from the REAL state, never on a timer. Resolving a seek costs
            // whatever the clip's keyframe spacing costs -- nothing at all on a short GOP,
            // a few hundred ms on a long one -- and a spinner on a fixed delay would be
            // lying in both directions at once.
            if (seeking) CircularProgressIndicator(
                color = Color.White,
                strokeWidth = 3.dp,
                modifier = Modifier.align(Alignment.Center).size(44.dp),
            )

            if (note != null) Text(
                note,
                color = Color.White,
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier
                    .align(Alignment.Center)
                    .padding(24.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .background(Color.Black.copy(alpha = 0.66f))
                    .padding(horizontal = 14.dp, vertical = 10.dp),
            )

            // ---- top row: the measurement, and the way out
            AnimatedVisibility(
                visible = showControls,
                enter = fadeIn(), exit = fadeOut(),
                modifier = Modifier.align(Alignment.TopCenter),
            ) {
                Row(
                    Modifier
                        .fillMaxWidth()
                        .background(Color.Black.copy(alpha = 0.40f))
                        .padding(edges.topBar())
                        .padding(vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    // What this screen exists to show: whether the pipeline is keeping up.
                    // `dropped` is the honest half of it -- fps alone looks healthy while
                    // half the clip is being thrown away to achieve it.
                    Text(
                        "%.1f fps  %d face%s  %d dropped".format(
                            fps, faces, if (faces == 1) "" else "s", dropped),
                        color = Color.White.copy(alpha = 0.85f),
                        fontFamily = FontFamily.Monospace,
                        fontSize = 12.sp,
                        modifier = Modifier.weight(1f).padding(start = 8.dp),
                    )
                    IconButton(onClick = { forceLandscape = !forceLandscape }) {
                        Icon(
                            IconRotate,
                            stringResource(R.string.player_landscape),
                            tint = if (forceLandscape) MaterialTheme.colorScheme.primary
                                   else Color.White,
                        )
                    }
                    IconButton(onClick = onClose) {
                        Icon(Icons.Default.Close, stringResource(R.string.player_close),
                             tint = Color.White)
                    }
                }
            }

            // ---- transport
            AnimatedVisibility(
                visible = showControls,
                enter = fadeIn(), exit = fadeOut(),
                modifier = Modifier.align(Alignment.BottomCenter),
            ) {
                Column(
                    Modifier
                        .fillMaxWidth()
                        // The SCRIM reaches the physical edge -- it is what makes white
                        // controls legible over a bright frame -- and the padding is applied
                        // INSIDE it, so the bar looks full-bleed while nothing touchable is.
                        .background(Color.Black.copy(alpha = 0.55f))
                        .padding(edges.bottomBar())
                        .padding(vertical = 6.dp),
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        IconButton(onClick = onPlayPause) {
                            Icon(
                                if (playing) IconPause else Icons.Default.PlayArrow,
                                stringResource(if (playing) R.string.out_pause
                                               else R.string.out_play),
                                tint = Color.White,
                            )
                        }
                        Text(
                            clock(scrubMs ?: positionMs),
                            color = Color.White,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 12.sp,
                        )
                        Slider(
                            value = (scrubMs ?: positionMs).toFloat()
                                .coerceIn(0f, durationMs.coerceAtLeast(1).toFloat()),
                            onValueChange = { scrubMs = it.toInt() },
                            // Committed on RELEASE, not per pixel: a seek re-positions the
                            // extractor and flushes the codec, and doing that for every
                            // sample of a drag is how a scrub bar becomes unusable.
                            onValueChangeFinished = {
                                scrubMs?.let(onSeek)
                                scrubMs = null
                            },
                            valueRange = 0f..durationMs.coerceAtLeast(1).toFloat(),
                            modifier = Modifier.weight(1f).padding(horizontal = 8.dp),
                        )
                        Text(
                            clock(durationMs),
                            color = Color.White.copy(alpha = 0.7f),
                            fontFamily = FontFamily.Monospace,
                            fontSize = 12.sp,
                            modifier = Modifier.padding(end = 8.dp),
                        )
                    }
                }
            }
        }
    }
}

/** Floors, so an edge that reports nothing still gets a margin a thumb can live with. */
private val kEdgeMin: Dp = 16.dp
private val kEdgeBottomMin: Dp = 32.dp

/**
 * How far the controls must stay from the physical edges of the screen.
 *
 * ⚠ `safeDrawingPadding()` ON ITS OWN IS ZERO HERE, and that is the whole bug it replaces.
 * This player HIDES the system bars, and a hidden bar reports no inset -- so the transport
 * was laid out flush against the bottom of the panel, underneath the gesture handle, where
 * it is both hard to see against the video and impossible to drag without the system
 * claiming the touch. It read as "the seek bar does not show" rather than as a margin bug,
 * which is exactly how a zero inset fails.
 *
 * What survives hiding the bars is the display CUTOUT and the gesture regions, so both are
 * unioned in; then every edge takes a floor as well, because the union is still zero on a
 * phone with no cutout and three-button navigation. Landscape is the case that needs all
 * three: the bar moves to a side, the cutout comes with it, and the player has a button
 * that puts it there deliberately.
 */
@Composable
private fun playerEdgeInsets(): PlayerEdges {
    val dir = LocalLayoutDirection.current
    val p = WindowInsets.safeDrawing
        .union(WindowInsets.systemGestures)
        .union(WindowInsets.displayCutout)
        .asPaddingValues()
    return PlayerEdges(
        start = maxOf(p.calculateStartPadding(dir), kEdgeMin),
        end = maxOf(p.calculateEndPadding(dir), kEdgeMin),
        top = maxOf(p.calculateTopPadding(), kEdgeMin),
        bottom = maxOf(p.calculateBottomPadding(), kEdgeBottomMin),
    )
}

/**
 * ⚠ Each bar pads only the edges it actually touches. A single PaddingValues applied to
 * both would give the TOP row the bottom bar's 32 dp underneath it -- a gap where nothing
 * is, pushing the readout down over the picture for a navigation bar at the other end of
 * the screen.
 */
private data class PlayerEdges(val start: Dp, val end: Dp, val top: Dp, val bottom: Dp) {
    fun topBar() = PaddingValues(start = start, end = end, top = top)
    fun bottomBar() = PaddingValues(start = start, end = end, bottom = bottom)
}

private fun clock(ms: Int): String {
    val s = (ms / 1000).coerceAtLeast(0)
    return "%d:%02d".format(s / 60, s % 60)
}

/**
 * The Activity behind a Compose context.
 *
 * `LocalContext` inside a Dialog is not necessarily the Activity -- it can be a
 * ContextWrapper around it -- so the chain is walked rather than cast. A failed cast here
 * would have been a crash on exactly one device configuration.
 */
private fun Context.findActivity(): Activity? {
    var c: Context? = this
    while (c is ContextWrapper) {
        if (c is Activity) return c
        c = c.baseContext
    }
    return null
}
