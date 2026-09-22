package com.facefusion.mobile.ui

import android.graphics.Bitmap
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
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

        IconButton(
            onClick = onExit,
            modifier = Modifier.align(Alignment.BottomEnd).padding(16.dp),
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
