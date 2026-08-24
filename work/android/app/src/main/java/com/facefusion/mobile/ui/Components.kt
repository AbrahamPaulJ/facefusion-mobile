package com.facefusion.mobile.ui

import android.graphics.Bitmap
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.facefusion.mobile.R

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
fun Wordmark(modifier: Modifier = Modifier) {
    Text(
        buildAnnotatedString {
            withStyle(SpanStyle(fontWeight = FontWeight.Light)) { append("FACE") }
            withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append("FUSION") }
        },
        style = WordmarkStyle,
        color = MaterialTheme.colorScheme.onBackground,
        modifier = modifier,
    )
}

/** A small all-caps caption. Used for the pane labels and settings section headers. */
@Composable
fun Caption(text: String, modifier: Modifier = Modifier) {
    Text(
        text.uppercase(),
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        letterSpacing = 1.5.sp,
        fontSize = 11.sp,
        modifier = modifier,
    )
}

/**
 * One full-width preview pane.
 *
 * Full width is the point: the previous UI showed the result in whatever space was left
 * over, and a face swap cannot be judged at that size. The box reserves its space via the
 * source's own aspect ratio so the layout does not jump when the first frame arrives.
 */
@Composable
fun PreviewPane(
    label: String,
    /**
     * The box height, IDENTICAL for both panes.
     *
     * Deliberately not the source's aspect ratio. Sizing each pane to its content meant a
     * portrait clip produced a box taller than the screen, so the two were never visible at
     * once -- and a before/after you have to scroll between is not a comparison. A fixed
     * shared box plus ContentScale.Fit letterboxes the image instead, which keeps both on
     * screen and keeps them the same size as each other.
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
    trailing: @Composable RowScope.() -> Unit = {},
) {
    Column(modifier.fillMaxWidth()) {
        Row(
            Modifier.fillMaxWidth().padding(bottom = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Caption(label, Modifier.weight(1f))
            trailing()
        }
        Box(
            Modifier
                .fillMaxWidth()
                .height(height)
                .clip(RoundedCornerShape(14.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant)
                .then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier),
            contentAlignment = Alignment.Center,
        ) {
            if (bitmap != null) {
                Image(
                    bitmap.asImageBitmap(), label,
                    Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit,
                )
            } else {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    if (actionIcon != null) {
                        Icon(
                            actionIcon, null,
                            Modifier.size(40.dp),
                            tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Text(
                        placeholder,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(horizontal = 24.dp),
                    )
                }
            }
            overlay?.invoke(this)
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
                    if (expanded) "Collapse" else "Expand",
                    Modifier.rotate(angle),
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            AnimatedVisibility(expanded) {
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
 * The height is FIXED and must stay so: this sits inside the page's vertical scroll, and a
 * scrollable of unbounded height nested in another one cannot measure.
 */
@Composable
fun LogBox(text: String, modifier: Modifier = Modifier) {
    val scroll = rememberScrollState()
    // Follow the tail, which is the only part anyone reads while a run is going.
    LaunchedEffect(text) { scroll.animateScrollTo(scroll.maxValue) }
    Column(modifier.fillMaxWidth()) {
        Caption("Log", Modifier.padding(bottom = 4.dp))
        Surface(
            Modifier.fillMaxWidth().height(170.dp),
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
