package com.facefusion.mobile.ui

import android.graphics.Bitmap
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import android.view.ViewGroup
import android.view.WindowManager
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.calculateEndPadding
import androidx.compose.foundation.layout.displayCutout
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.systemGestures
import androidx.compose.foundation.layout.union
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.compose.ui.window.DialogWindowProvider
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.Image
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.facefusion.mobile.R

/**
 * The live feed, filling the screen, with the controls out of the way.
 *
 * Deliberately a SEPARATE surface rather than the Live column re-laid-out. The column is
 * a scroll of a dozen controls around one pane; hiding all of them to make the pane big
 * means every one of those controls grows a condition, and the pane itself grows two
 * modifier chains that have to stay in step. This draws the same Bitmap the pane draws
 * and nothing else.
 *
 * ⚠ It does NOT own the camera. The pump, the pipeline and the recorder all live in
 * [com.facefusion.mobile.LiveEngine] and keep running exactly as they were; this is a
 * second view of the frames they were already producing, so entering and leaving costs
 * nothing and cannot interrupt a recording.
 *
 * ⚠ Tap-to-assign keeps working, through the same [liveTapToFrame] the inline pane uses.
 * That is the whole reason the mapping was lifted out of the pane: this surface is a
 * different size and a different aspect ratio, and a second copy of the crop arithmetic
 * would have been correct here only until one of the two was next corrected.
 */
@Composable
fun LiveFullscreen(
    frame: Bitmap?,
    mirror: Boolean,
    assignMode: Boolean,
    running: Boolean,
    onAssignFace: (Float, Float) -> Unit,
    onExit: () -> Unit,
) {
    val fw = (frame?.width ?: 0).toFloat()
    val fh = (frame?.height ?: 0).toFloat()

    // A Dialog, the same way the live player does it -- so this does not care where in
    // the tree it is rendered and survives a tab switch underneath it.
    Dialog(
        onDismissRequest = onExit,
        // Without this the window is inset to a dialog width and "fullscreen" is a black
        // card in the middle of one.
        properties = DialogProperties(usePlatformDefaultWidth = false),
    ) {
    // Back leaves fullscreen rather than the screen behind it. Anything else would end a
    // running session with a gesture the user meant as "make this smaller again".
    BackHandler(enabled = true, onBack = onExit)

    // ⚠ THE WINDOW, set up the way the live player's is. Without this the dialog is a
    // window that merely covers the screen: the system bars still sit over it, insets
    // inside it read as nothing, and the exit control lands UNDER the navigation bar and
    // hard against the physical edge -- which is exactly how it shipped in 0.9.31.
    // MATCH_PARENT makes it the whole display, decorFitsSystemWindows(false) makes the
    // insets real, and hiding the bars means there is no navigation bar to dodge.
    // Set on the DIALOG's window: a dialog gets its own, and decorating the Activity's
    // leaves the bars over this one.
    val view = LocalView.current
    val dialogWindow = (view.parent as? DialogWindowProvider)?.window
    LaunchedEffect(dialogWindow) {
        dialogWindow?.let { w ->
            w.setLayout(ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT)
            // A live feed is something you watch; the screen must not time out under it.
            w.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            WindowCompat.setDecorFitsSystemWindows(w, false)
            WindowInsetsControllerCompat(w, w.decorView).apply {
                hide(WindowInsetsCompat.Type.systemBars())
                systemBarsBehavior =
                    WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            }
        }
    }

    // Read INSIDE the dialog: it carries its own window and its own insets, and the
    // Activity's describe a different one. safeDrawing alone is not enough -- a swipe
    // brings the bars back transiently, and the gesture strip and any cutout are places
    // a control must not be even while they are hidden.
    val dir = LocalLayoutDirection.current
    val edges = WindowInsets.safeDrawing
        .union(WindowInsets.systemGestures)
        .union(WindowInsets.displayCutout)
        .asPaddingValues()
    val edgeEnd = maxOf(edges.calculateEndPadding(dir), 20.dp)
    val edgeBottom = maxOf(edges.calculateBottomPadding(), 28.dp)
    Box(
        Modifier
            .fillMaxSize()
            .background(Color.Black)
            // Same keys as the inline pane, and for the same reason: the Bitmap reference
            // changes every shot, so keying on it would rebuild the detector once per
            // frame and eat taps mid-recomposition. The DIMENSIONS are constant for a
            // session.
            .pointerInput(assignMode, running, mirror, fw, fh) {
                if (assignMode && running && fw > 0f && fh > 0f) {
                    detectTapGestures { off ->
                        val (bx, by) = liveTapToFrame(
                            off.x, off.y,
                            size.width.toFloat(), size.height.toFloat(),
                            fw, fh, mirror)
                        onAssignFace(bx, by)
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
                modifier = Modifier
                    .fillMaxSize()
                    // ⚠ The mirror is drawn here and nowhere else, exactly as the inline
                    // pane does it: the pipeline and the recorder keep the true image.
                    .graphicsLayer(scaleX = if (mirror) -1f else 1f),
            )
        } else {
            Text(stringResource(R.string.live_starting),
                 color = MaterialTheme.colorScheme.onSurface)
        }

        // The one way out, kept clear of every edge the system might want back.
        IconButton(
            onClick = onExit,
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(end = edgeEnd, bottom = edgeBottom),
            colors = IconButtonDefaults.filledTonalIconButtonColors(),
        ) {
            Icon(
                painterResource(R.drawable.ic_fullscreen_exit),
                contentDescription = stringResource(R.string.live_fullscreen_exit),
                modifier = Modifier.size(20.dp),
            )
        }
    }
    }
}
