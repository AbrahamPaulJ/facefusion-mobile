package com.facefusion.mobile.ui

import android.graphics.Bitmap
import android.widget.VideoView
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.path
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.facefusion.mobile.R
import kotlinx.coroutines.delay
import java.io.File
import androidx.compose.ui.res.stringResource
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom

/**
 * FaceFusion's mark, on a light plate.
 *
 * The plate is not decoration. The artwork is a BLACK ring with the face carved out of it,
 * so on a near-black background the ring disappears and all that is left is a face floating
 * in space. Insetting it inside a white circle reproduces exactly what the launcher icon
 * does, and for the same reason.
 */
@Composable
fun AppMark(size: Dp = 30.dp, modifier: Modifier = Modifier) {
    Box(
        modifier.size(size).clip(CircleShape).background(Color.White),
        contentAlignment = Alignment.Center,
    ) {
        // ⚠ drawable/ff_mark, NOT mipmap/ic_launcher. On API 26+ the launcher name resolves
        // to the adaptive-icon XML, and painterResource cannot load an AdaptiveIconDrawable
        // -- it throws "Only VectorDrawables and rasterized asset types are supported" at
        // first composition, which crashes the app on launch rather than at build time.
        Image(
            painterResource(R.drawable.ff_mark), null,
            Modifier.fillMaxSize().padding(1.5.dp),
        )
    }
}

/**
 * The wordmark.
 *
 * Two weights on one word rather than a display font, because no font may be bundled here
 * (see [Theme.kt]). The weight break at FACE|FUSION is what carries the identity; without
 * it this is just a heading in caps.
 */
@Composable
fun Wordmark(modifier: Modifier = Modifier, color: Color = MaterialTheme.colorScheme.onBackground) {
    Text(
        buildAnnotatedString {
            withStyle(SpanStyle(fontWeight = FontWeight.Light)) { append("FACE") }
            withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append("FUSION") }
        },
        style = WordmarkStyle,
        color = color,
        modifier = modifier,
    )
}

/**
 * Material's `file_download`, declared here rather than depended on.
 *
 * `material-icons-core` — the artifact material3 already brings in — carries a fixed short
 * list, and a download glyph is not on it. The rest live in `material-icons-extended`,
 * which is thousands of vectors pulled in for one, so the path is written out instead. It
 * is Google's own `file_download` path data, unchanged, on the standard 24 dp viewport.
 */
val IconDownload: ImageVector = ImageVector.Builder(
    name = "file_download",
    defaultWidth = 24.dp, defaultHeight = 24.dp,
    viewportWidth = 24f, viewportHeight = 24f,
).apply {
    // Black, so Icon()'s tint is what actually colours it — the convention every Material
    // icon follows. A themed colour baked in here would ignore the caller.
    path(fill = SolidColor(Color.Black)) {
        moveTo(19f, 9f); horizontalLineToRelative(-4f); verticalLineTo(3f)
        horizontalLineTo(9f); verticalLineToRelative(6f); horizontalLineTo(5f)
        lineToRelative(7f, 7f); lineToRelative(7f, -7f); close()
        moveTo(5f, 18f); verticalLineToRelative(2f); horizontalLineToRelative(14f)
        verticalLineToRelative(-2f); horizontalLineTo(5f); close()
    }
}.build()

/** A small all-caps caption. Used for the pane labels and settings section headers. */
@Composable
fun Caption(text: String, modifier: Modifier = Modifier, singleLine: Boolean = false) {
    Text(
        text.uppercase(),
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        letterSpacing = 1.5.sp,
        fontSize = 11.sp,
        // ⚠ A PANE's caption must never wrap. It shares its row with that pane's buttons,
        // so a second line makes the row taller -- and since ORIGINAL carries three icons
        // and SWAPPED carries one, only ORIGINAL grew, leaving two panes that are supposed
        // to be the same size sitting at different heights. It showed up on PORTRAIT
        // targets, where the two panes go side by side and each has half the width to fit
        // "ORIGINAL AT 0:03" and three buttons into.
        maxLines = if (singleLine) 1 else Int.MAX_VALUE,
        overflow = if (singleLine) TextOverflow.Ellipsis else TextOverflow.Clip,
        modifier = modifier,
    )
}

/**
 * A labelled group card, the visual "room" the Swap screen is divided into.
 *
 * The screen used to be one long scroll of chips, panes and buttons with nothing saying
 * which things belonged together. This is the divider: one card per group (the processor
 * stages, the workbench, the output), with the brand-red square as its mark — the same
 * mark the wordmark's two-weight break carries — and a trailing slot for a live status
 * readout such as which stages are armed.
 *
 * Deliberately `surface`, not `surfaceVariant`: the card is the step the panes sit ON,
 * and the panes inside it are already the recessed step. Nesting both the other way made
 * the card read as a hole in the page.
 */
@Composable
fun SectionCard(
    title: String,
    modifier: Modifier = Modifier,
    trailing: @Composable RowScope.() -> Unit = {},
    // Collapsible behaviour, opt-in. When [onToggle] is non-null the whole title row
    // becomes the tap target and [content] animates in/out under it; the trailing arrow
    // flips with the state so the affordance reads without a tap.
    //
    // ⚠ These sit BEFORE [content] deliberately: a trailing lambda at the call site binds
    // to the LAST parameter, so content must stay last or the block goes to onToggle.
    collapsible: Boolean = false,
    expanded: Boolean = true,
    onToggle: (() -> Unit)? = null,
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(
        modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(MaterialTheme.colorScheme.surface)
            .border(1.dp, MaterialTheme.colorScheme.outlineVariant, RoundedCornerShape(20.dp))
            .padding(horizontal = 14.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.then(
                if (onToggle != null) Modifier
                    // Slightly larger than the visual title so the whole band is easy to
                    // hit, but not full-width-bleed: the card's own padding already frames
                    // it, and an 100%-wide target would make accidental collapses common.
                    .clip(RoundedCornerShape(10.dp))
                    .clickable(onClick = onToggle)
                    .padding(vertical = 2.dp)
                else Modifier
            ),
        ) {
            Box(
                Modifier
                    .size(10.dp)
                    .clip(RoundedCornerShape(3.dp))
                    .background(MaterialTheme.colorScheme.primary),
            )
            Spacer(Modifier.width(8.dp))
            Caption(title, Modifier.weight(1f))
            trailing()
            if (collapsible) {
                Spacer(Modifier.width(6.dp))
                Icon(
                    Icons.Default.KeyboardArrowDown,
                    stringResource(if (expanded) R.string.common_collapse
                                   else R.string.common_expand),
                    Modifier
                        .size(20.dp)
                        .rotate(if (expanded) 0f else 180f),
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        // Plain `if`, deliberately NOT AnimatedVisibility.
        //
        // ⚠ The animated version caused a real overlap: inside the page's verticalScroll,
        // expanding a card that holds sliders (the trim card's RangeSlider carries a 48 dp
        // touch target) could leave the drawn content overlapping the elements below it --
        // the "expanded but contents overlap and can't be used" report. A conditional
        // render has no transition in which measured height and drawn height disagree,
        // so the overlap cannot happen. The chevron above still rotates, which keeps the
        // fold affordance's motion.
        if (expanded) content()
    }
}

/**
 * Pan and zoom shared by several panes.
 *
 * Hoisted into a state object rather than remembered inside each pane, because the point is
 * that the panes move TOGETHER: pinching the original has to zoom the swapped one to the
 * same place, or a before/after comparison at 4x is worthless. One instance, passed to
 * every pane, is what makes that true by construction instead of by synchronisation.
 */
@Stable
class ZoomState {
    var scale by mutableStateOf(1f)
        private set
    var offset by mutableStateOf(Offset.Zero)
        private set

    val zoomed get() = scale > 1.01f

    fun reset() {
        scale = 1f
        offset = Offset.Zero
    }

    /**
     * Fold one gesture into the state, clamped so the image cannot be lost.
     *
     * The offset limit is (scale - 1) * size / 2 per axis: at scale 1 that is zero, so the
     * image cannot be panned at all while it fits, and at higher scales it is exactly the
     * amount of image hidden outside the box. Without it a fling leaves an empty pane and
     * no way back except double-tap.
     */
    fun transform(zoomChange: Float, panChange: Offset, boxW: Float, boxH: Float) {
        val next = (scale * zoomChange).coerceIn(1f, 6f)
        val maxX = (next - 1f) * boxW / 2f
        val maxY = (next - 1f) * boxH / 2f
        // Pan is scaled by the zoom change too, so the point under the fingers stays put.
        val moved = offset * (next / scale) + panChange
        scale = next
        offset = Offset(moved.x.coerceIn(-maxX, maxX), moved.y.coerceIn(-maxY, maxY))
    }
}

@Composable
fun PreviewPane(
    label: String,
    /**
     * The box height, IDENTICAL for every pane on screen.
     *
     * Deliberately not the source's aspect ratio. Sizing each pane to its content meant a
     * portrait clip produced a box taller than the screen, so the two were never visible at
     * once -- and a before/after you have to scroll between is not a comparison. A fixed
     * shared box plus ContentScale.Fit letterboxes the image instead, which keeps them on
     * screen and keeps them the same size as each other. Which WAY they are stacked is the
     * caller's decision (see PreviewPair).
     */
    height: Dp,
    bitmap: Bitmap?,
    placeholder: String,
    modifier: Modifier = Modifier,
    /** Makes the whole pane tappable. Used so the target is picked by tapping its own frame. */
    onClick: (() -> Unit)? = null,
    /** Shown large and centred while [bitmap] is null, as the pane's call to action. */
    actionIcon: ImageVector? = null,
    /** Drawn ON TOP of the pane, whatever it contains. Used for the model download. */
    overlay: (@Composable BoxScope.() -> Unit)? = null,
    /**
     * Shared pan/zoom. Pass the SAME instance to every pane that should move together;
     * null disables gestures entirely (the output pane, which owns its own surface).
     */
    zoom: ZoomState? = null,
    /**
     * Face boxes to outline, in [bitmap]'s OWN pixel coordinates: five floats per face --
     * x0, y0, x1, y1, score -- exactly as `NativePipe.detectFaces` returns them.
     *
     * In image space rather than pane space on purpose: the pane knows its own letterbox
     * and its own zoom, and the detector does not. Converting here is four lines; converting
     * at the call site would mean teaching MainActivity about ContentScale.Fit.
     */
    faceBoxes: FloatArray? = null,
    /**
     * The box of the face currently chosen as the reference, drawn differently from the
     * rest. Four floats, same coordinates as [faceBoxes].
     */
    referenceBox: FloatArray? = null,
    /**
     * A tap landed inside one of [faceBoxes]; the arguments are the tap in the BITMAP's own
     * pixel coordinates.
     *
     * ⚠ Only consulted when there are boxes to hit. A tap that misses every box still runs
     * [onClick] -- the pane is the target picker first and a face picker second, and
     * swallowing misses would break the way every other pane on the screen behaves.
     */
    onPickFace: ((Float, Float) -> Unit)? = null,
    /**
     * When set, the image box is THIS wide instead of filling the column, and is centred
     * inside the (still full-width) container. Used by the swapped result pane: its box
     * then matches the image's own aspect ratio rather than stretching full width and
     * letterboxing a portrait result into grey side bars.
     */
    contentWidth: Dp? = null,
    trailing: @Composable RowScope.() -> Unit = {},
) {
    // ONE container around the caption row AND the image, rather than a caption floating
    // above a rounded box. The label and its buttons sat flush against the pane's outer
    // edge while the image below was clipped to a 14 dp radius, so neither lined up with
    // the other and the text read as belonging to the page rather than to the pane under
    // it -- most visible on ORIGINAL and SWAPPED, which sit side by side.
    //
    // The nesting is the theme's own, not a new colour: `surface` is the card step above
    // the background and `surfaceVariant` the recessed step above that, so the image still
    // reads as a panel set INTO the container. The radii are concentric -- inner = outer
    // minus the padding -- which is what stops the corners looking doubled.
    Column(
        modifier
            .clip(RoundedCornerShape(18.dp))
            .background(MaterialTheme.colorScheme.surface)
            .padding(horizontal = 6.dp, vertical = 5.dp)
    ) {
        Row(
            Modifier.fillMaxWidth().padding(start = 4.dp, bottom = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // A badge rather than a bare caption: the label names the pane, and a small
            // recessed plate is how the name stops reading as a stray line of page text
            // and starts reading as part of the pane. Same surfaceVariant as the image
            // well, so the two line up like a label on a folder.
            Text(
                label.uppercase(),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                letterSpacing = 1.2.sp,
                fontSize = 10.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier
                    .weight(1f, fill = false)
                    .clip(RoundedCornerShape(6.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant)
                    .padding(horizontal = 8.dp, vertical = 3.dp),
            )
            // Fixed HEIGHT, whatever the slot holds; width still wraps.
            //
            // It used to size to its content, and content alternated between an 18 dp
            // spinner and a 28 dp IconButton -- so starting a preview changed the label
            // row's height and shoved the trim slider and the Swap button down the screen
            // mid-interaction. Reserving the larger height makes the swap invisible.
            //
            // Height only: the shift was vertical, and pinning the width too would clip the
            // "Change" button the original pane carries.
            Box(
                // 40 dp, not 28: a Material TextButton has a 40 dp minimum height, so
                // reserving an IconButton's 28 clipped "Change" and "Save frame" to their
                // top halves. Reserve the tallest thing the slot can hold.
                Modifier.height(40.dp),
                contentAlignment = Alignment.Center,
            ) {
                Row(verticalAlignment = Alignment.CenterVertically, content = trailing)
            }
        }
        Box(
            Modifier
                .then(if (contentWidth != null)
                          Modifier.align(Alignment.CenterHorizontally).width(contentWidth)
                      else Modifier.fillMaxWidth())
                .height(height)
                // 12, not 14: concentric with the container 18 minus its 6 dp padding.
                .clip(RoundedCornerShape(12.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant)
                .border(1.dp, MaterialTheme.colorScheme.outlineVariant,
                        RoundedCornerShape(12.dp))
                .then(
                    // Gestures only once there is an image: with none, the pane is purely a
                    // picker and `clickable` keeps its ripple and accessibility semantics.
                    if (zoom != null && bitmap != null) {
                        Modifier
                            .pointerInput(zoom) {
                                // ⚠ NOT detectTransformGestures, which is what this was.
                                // That helper treats a ONE-FINGER drag as a pan and
                                // consumes it, so the parent verticalScroll never saw the
                                // gesture: a finger that happened to land on a pane could
                                // not scroll the page, and two of the four things on the
                                // Swap screen are panes.
                                //
                                // Consume only when the gesture is really ours:
                                //   * two or more pointers -- a pinch, which is the only
                                //     way to zoom;
                                //   * or one pointer while ALREADY zoomed, which is a pan
                                //     of an image bigger than its box. Double-tap resets,
                                //     so there is always a way back to scrolling.
                                // One finger at scale 1 is left entirely alone and falls
                                // through to the scroll.
                                awaitEachGesture {
                                    awaitFirstDown(requireUnconsumed = false)
                                    do {
                                        val ev = awaitPointerEvent()
                                        val mine = ev.changes.size > 1 || zoom.zoomed
                                        if (mine) {
                                            val z = ev.calculateZoom()
                                            val pan = ev.calculatePan()
                                            if (z != 1f || pan != Offset.Zero) {
                                                zoom.transform(z, pan, size.width.toFloat(),
                                                               size.height.toFloat())
                                                ev.changes.forEach { it.consume() }
                                            }
                                        }
                                    } while (ev.changes.any { it.pressed })
                                }
                            }
                            .pointerInput(zoom, onClick, onPickFace, faceBoxes) {
                                detectTapGestures(
                                    // The only way back from a deep zoom, and the
                                    // conventional one.
                                    onDoubleTap = { zoom.reset() },
                                    onTap = { tap ->
                                        // PANE COORDINATES BACK TO IMAGE COORDINATES, in
                                        // the reverse order they were applied: undo the
                                        // zoom layer (scale about the box's centre, then
                                        // translate), then undo ContentScale.Fit's
                                        // letterbox. Getting this backwards does not throw
                                        // -- it picks a face a few dozen pixels from the
                                        // one under the finger, which is indistinguishable
                                        // from a bad detector.
                                        var picked = false
                                        val boxes = faceBoxes
                                        if (onPickFace != null && boxes != null &&
                                            boxes.size >= 5) {
                                            val bw = size.width.toFloat()
                                            val bh = size.height.toFloat()
                                            val iw = bitmap.width.toFloat()
                                            val ih = bitmap.height.toFloat()
                                            val z = zoom.scale
                                            val ux = bw / 2f + (tap.x - bw / 2f -
                                                                zoom.offset.x) / z
                                            val uy = bh / 2f + (tap.y - bh / 2f -
                                                                zoom.offset.y) / z
                                            val k = minOf(bw / iw, bh / ih)
                                            val ix = (ux - (bw - iw * k) / 2f) / k
                                            val iy = (uy - (bh - ih * k) / 2f) / k
                                            for (i in 0 until boxes.size / 5) {
                                                val b = i * 5
                                                if (ix >= boxes[b] && ix <= boxes[b + 2] &&
                                                    iy >= boxes[b + 1] && iy <= boxes[b + 3]) {
                                                    onPickFace(ix, iy)
                                                    picked = true
                                                    break
                                                }
                                            }
                                        }
                                        if (!picked) onClick?.invoke()
                                    },
                                )
                            }
                    } else if (onClick != null) {
                        Modifier.clickable(onClick = onClick)
                    } else {
                        Modifier
                    }
                ),
            contentAlignment = Alignment.Center,
        ) {
            if (bitmap != null) {
                Image(
                    bitmap.asImageBitmap(), label,
                    Modifier
                        .fillMaxSize()
                        .then(
                            if (zoom != null) Modifier.graphicsLayer {
                                scaleX = zoom.scale
                                scaleY = zoom.scale
                                translationX = zoom.offset.x
                                translationY = zoom.offset.y
                            } else Modifier
                        ),
                    contentScale = ContentScale.Fit,
                )
                if (faceBoxes != null && faceBoxes.size >= 5) {
                    // The SAME graphicsLayer as the Image above, so the outlines pan and
                    // zoom with what they are outlining rather than sliding off it.
                    Canvas(
                        Modifier
                            .fillMaxSize()
                            .then(
                                if (zoom != null) Modifier.graphicsLayer {
                                    scaleX = zoom.scale
                                    scaleY = zoom.scale
                                    translationX = zoom.offset.x
                                    translationY = zoom.offset.y
                                } else Modifier
                            )
                    ) {
                        // ContentScale.Fit, recomputed rather than guessed: uniform scale
                        // to the smaller ratio, then centred. Getting this wrong does not
                        // fail loudly -- it draws rectangles that are slightly off the
                        // faces, which reads as a bad detector.
                        val iw = bitmap.width.toFloat()
                        val ih = bitmap.height.toFloat()
                        if (iw > 0f && ih > 0f) {
                            val k = minOf(size.width / iw, size.height / ih)
                            val ox = (size.width - iw * k) / 2f
                            val oy = (size.height - ih * k) / 2f
                            // 2 dp at scale 1, thinned as the pane zooms in so the stroke
                            // stays the same width on screen instead of growing into a slab.
                            val w = 2.dp.toPx() / (zoom?.scale ?: 1f)
                            for (i in 0 until faceBoxes.size / 5) {
                                val b = i * 5
                                // The CHOSEN face is drawn twice as thick and opaque; the
                                // others are dimmed. A selector that picked the wrong
                                // neighbour and one that picked nothing look identical
                                // otherwise, and they need different reactions.
                                val chosen = referenceBox != null &&
                                    referenceBox.size >= 4 &&
                                    kotlin.math.abs(referenceBox[0] - faceBoxes[b]) < 1f &&
                                    kotlin.math.abs(referenceBox[1] - faceBoxes[b + 1]) < 1f
                                drawRect(
                                    color = if (chosen) FfRed
                                            else FfRed.copy(alpha =
                                                if (referenceBox != null) 0.35f else 1f),
                                    topLeft = Offset(ox + faceBoxes[b] * k,
                                                     oy + faceBoxes[b + 1] * k),
                                    size = Size((faceBoxes[b + 2] - faceBoxes[b]) * k,
                                                (faceBoxes[b + 3] - faceBoxes[b + 1]) * k),
                                    style = Stroke(width = if (chosen) w * 2f else w),
                                )
                            }
                        }
                    }
                }
            } else {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    if (actionIcon != null) {
                        Icon(
                            actionIcon, null,
                            Modifier.size(48.dp),
                            // The pane's call to action gets the brand tint, not the flat
                            // grey: an empty pane is a button asking to be tapped, and the
                            // accent is how a button is recognised.
                            tint = MaterialTheme.colorScheme.primary.copy(alpha = 0.75f),
                        )
                    }
                    Text(
                        placeholder,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(horizontal = 24.dp),
                    )
                }
            }
            overlay?.invoke(this)
        }
    }
}

/**
 * A compact 72 dp tile for the workbench row — source face on the left, target on the
 * right.
 *
 * They used to be full-height panes; a portrait clip pushed them off the first screen and
 * the panes competed with the stage chips for what a fresh install sees. A tile keeps both
 * inputs permanently on screen: 72 dp tall whatever the state, width wrapping to whatever
 * the row gives it, the label as a translucent plate ON the picture so the whole height is
 * image, and the pick/remove actions tucked into the top corner.
 */
@Composable
fun FaceTile(
    label: String,
    bitmap: Bitmap?,
    placeholder: String,
    modifier: Modifier = Modifier,
    /** Makes the whole tile tappable — the target is picked by tapping its own frame. */
    onClick: (() -> Unit)? = null,
    /** Shown large and centred while [bitmap] is null, as the tile's call to action. */
    actionIcon: ImageVector? = null,
    /** Small actions (camera, change, delete…) over the tile's top-right corner. */
    actions: @Composable () -> Unit = {},
    /** Actions that keep the original bottom-right corner (the video camera). */
    bottomActions: @Composable () -> Unit = {},
) {
    Box(
        modifier
            .height(72.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(MaterialTheme.colorScheme.surface)
            .then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier),
        contentAlignment = Alignment.Center,
    ) {
        if (bitmap != null) {
            // A 64 dp square thumbnail pinned to the tile's LEADING edge, not a full-bleed
            // image: the tile keeps its own frame and the thumbnail reads as a compact
            // preview -- content first, then the arrow pointing at what it becomes.
            Image(
                bitmap.asImageBitmap(), label,
                Modifier
                    .align(Alignment.CenterStart)
                    .size(64.dp)
                    .clip(RoundedCornerShape(12.dp)),
                contentScale = ContentScale.Crop,
            )
        } else {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                if (actionIcon != null) {
                    Icon(
                        actionIcon, null,
                        Modifier.size(26.dp),
                        // Same brand-tinted call-to-action as the full panes use, scaled
                        // down to fit the tile's smaller plate.
                        tint = MaterialTheme.colorScheme.primary.copy(alpha = 0.75f),
                    )
                }
                Text(
                    placeholder,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.padding(horizontal = 6.dp),
                )
            }
        }
        // The name rides ON the picture so the tile's whole height is image; a caption row
        // of its own would push the pair past the 72 dp the workbench row is budgeted for.
        Text(
            label.uppercase(),
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            letterSpacing = 1.2.sp,
            fontSize = 9.sp,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(5.dp)
                // The label sits directly on the image with no plate behind it.
                .padding(horizontal = 6.dp, vertical = 2.dp),
        )
        Row(
            Modifier.align(Alignment.TopEnd).padding(4.dp),
            horizontalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            actions()
        }
        Row(
            Modifier.align(Alignment.BottomEnd).padding(4.dp),
            horizontalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            bottomActions()
        }
    }
}

/**
 * The finished video, playable in place, with a scrub bar and a Save frame button.
 *
 * Framework [VideoView] rather than media3/ExoPlayer. One pane does not justify a player
 * dependency in an APK whose whole design is about not carrying libraries it can do
 * without -- there is no OpenCV and no ONNX Runtime on device for the same reason.
 *
 * The scrub bar is what makes Save frame worth having: it is how you find the frame you
 * want before you save it, and seeking a local MP4 is cheap.
 */
@Composable
fun OutputPane(
    file: File,
    height: Dp,
    /** Given the position in milliseconds currently on screen. */
    onSaveFrame: (Int) -> Unit,
    modifier: Modifier = Modifier,
    partial: Boolean = false,
    enabled: Boolean = true,
    /** When set, the video box is THIS wide instead of filling the column, and is
     *  centred inside the (still full-width) pane -- so a portrait clip is not
     *  letterboxed into side bars. Mirrors PreviewPane.contentWidth. */
    contentWidth: Dp? = null,
) {
    // Keyed on the file: a second run replaces the video, and stale position/duration from
    // the previous one would put the scrub bar somewhere that no longer exists.
    var player by remember(file) { mutableStateOf<VideoView?>(null) }
    var durationMs by remember(file) { mutableStateOf(0) }
    var positionMs by remember(file) { mutableStateOf(0) }
    var playing by remember(file) { mutableStateOf(false) }

    // Only while playing. VideoView has no position callback, so the bar has to be polled,
    // and polling a paused video is pure battery.
    LaunchedEffect(playing, file) {
        while (playing) {
            positionMs = player?.currentPosition ?: 0
            delay(200)
        }
    }

    Column(modifier.fillMaxWidth()) {
        Row(
            Modifier.fillMaxWidth().padding(bottom = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Caption(stringResource(if (partial) R.string.out_output_partial
                                  else R.string.out_output), Modifier.weight(1f))
            Box(Modifier.height(40.dp), contentAlignment = Alignment.Center) {
                TextButton(
                    onClick = { onSaveFrame(positionMs) },
                    enabled = enabled,
                ) { Text(stringResource(R.string.out_save_frame)) }
            }
        }
        Box(
            Modifier
                .then(if (contentWidth != null)
                          Modifier.align(Alignment.CenterHorizontally).width(contentWidth)
                      else Modifier.fillMaxWidth())
                .height(height)
                .clip(RoundedCornerShape(12.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant)
                .border(1.dp, MaterialTheme.colorScheme.outlineVariant,
                        RoundedCornerShape(12.dp)),
            contentAlignment = Alignment.Center,
        ) {
            // ⚠ KEYED ON THE PATH, and this is not decoration. AndroidView's `factory`
            // runs ONCE; nothing re-points a VideoView when the composable's `file`
            // changes, so swapping the file left the previous video loaded and playing.
            //
            // It went unnoticed because a single run never exercises it: between two runs
            // `outputFile` is null while discardOutput deletes the old file, the pane
            // leaves composition, and the next run builds a fresh view. Batch swipe is the
            // first path that goes from one file straight to another, and it showed clip
            // one's frames under clip two's name.
            //
            // key() rather than an `update` block: VideoView is a stateful legacy view with
            // a prepared/playing lifecycle, and re-targeting one mid-playback is a longer
            // list of things to get right than throwing it away. Every piece of state
            // around it is already `remember(file)`, so it resets with the view.
            key(file.absolutePath) {
            AndroidView(
                factory = { ctx ->
                    VideoView(ctx).apply {
                        setVideoPath(file.absolutePath)
                        setOnPreparedListener { mp ->
                            durationMs = mp.duration
                            // Seek off zero so the pane shows the first frame instead of
                            // black while paused.
                            seekTo(1)
                        }
                        setOnCompletionListener { playing = false }
                        player = this
                    }
                },
                modifier = Modifier.fillMaxSize(),
            )
            }
        }
        Row(
            Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            TextButton(
                onClick = {
                    val v = player ?: return@TextButton
                    if (playing) v.pause() else v.start()
                    playing = !playing
                },
                enabled = enabled,
            ) { Text(stringResource(if (playing) R.string.out_pause else R.string.out_play)) }

            // The slider reads too dark / high-contrast against the theme, so every part
            // (thumb, active and inactive track) has its opacity cut by 31%: 69% alpha
            // keeps the exact same hue with a much softer contrast.
            val slimColors = SliderDefaults.colors(
                thumbColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.69f),
                activeTrackColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.69f),
                inactiveTrackColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.69f),
            )
            Slider(
                value = positionMs.toFloat().coerceIn(0f, durationMs.toFloat()),
                onValueChange = {
                    positionMs = it.toInt()
                    player?.seekTo(it.toInt())
                },
                valueRange = 0f..durationMs.coerceAtLeast(1).toFloat(),
                enabled = enabled && durationMs > 0,
                modifier = Modifier.weight(1f),
                colors = slimColors,
            )
            Text(
                "%d.%02ds".format(positionMs / 1000, (positionMs % 1000) / 10),
                style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(start = 6.dp),
            )
        }
    }
}

/**
 * A collapsible section with a rotating chevron.
 *
 * Replaces the "edit"/"hide" text button the option cards used to carry: a chevron is the
 * conventional affordance for this and, unlike a word, it shows the CURRENT state and the
 * direction of travel at once.
 */
@Composable
fun Accordion(
    title: String,
    summary: String?,
    expanded: Boolean,
    onToggle: () -> Unit,
    modifier: Modifier = Modifier,
    content: @Composable ColumnScope.() -> Unit,
) {
    val angle by animateFloatAsState(if (expanded) 180f else 0f, label = "chevron")
    Card(modifier.fillMaxWidth()) {
        Column(Modifier.padding(horizontal = 14.dp, vertical = 10.dp)) {
            Row(
                Modifier.fillMaxWidth().clickable(onClick = onToggle),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(Modifier.weight(1f)) {
                    Text(title, style = MaterialTheme.typography.titleSmall)
                    if (!summary.isNullOrEmpty())
                        Text(
                            summary,
                            style = MaterialTheme.typography.bodySmall,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 11.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                }
                Icon(
                    Icons.Default.KeyboardArrowDown,
                    stringResource(if (expanded) R.string.out_collapse else R.string.out_expand),
                    Modifier.rotate(angle),
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            // Same plain conditional as SectionCard: a transition here can draw the
            // content over the elements below it inside a scrolling container.
            if (expanded) {
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Spacer(Modifier.height(4.dp))
                    content()
                }
            }
        }
    }
}

/**
 * The log, in a panel that scrolls on its own.
 *
 * The whole row is the toggle: tap "Log" (or its chevron) to fold the 170 dp panel away
 * and tap again to bring it back. Default COLLAPSED -- the log is debug output, not a
 * thing every visit needs open, and a standing 170 dp panel pushed the controls below it
 * down the page.
 *
 * The height of the open panel is FIXED and must stay so: this sits inside the page's
 * vertical scroll, and a scrollable of unbounded height nested in another one cannot
 * measure.
 */
@Composable
fun LogBox(
    text: String,
    expanded: Boolean,
    onToggle: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val scroll = rememberScrollState()
    // Follow the tail, which is the only part anyone reads while a run is going.
    LaunchedEffect(text, expanded) {
        if (expanded) scroll.animateScrollTo(scroll.maxValue)
    }
    Column(modifier.fillMaxWidth()) {
        Row(
            Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(10.dp))
                .clickable(onClick = onToggle)
                .padding(vertical = 2.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Caption(stringResource(R.string.out_log), Modifier.weight(1f))
            Icon(
                Icons.Default.KeyboardArrowDown,
                stringResource(if (expanded) R.string.common_collapse
                               else R.string.common_expand),
                Modifier
                    .size(20.dp)
                    .rotate(if (expanded) 0f else 180f),
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        // Plain `if` for the same reason as SectionCard: AnimatedVisibility in this
        // verticalScroll could draw the panel over what followed it mid-transition.
        if (expanded) {
            Surface(
                Modifier.fillMaxWidth().height(170.dp).padding(top = 4.dp),
                shape = RoundedCornerShape(12.dp),
                color = MaterialTheme.colorScheme.surfaceVariant,
            ) {
                Text(
                    text,
                    Modifier.verticalScroll(scroll).padding(12.dp),
                    fontFamily = FontFamily.Monospace,
                    fontSize = 10.sp,
                    lineHeight = 14.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}
