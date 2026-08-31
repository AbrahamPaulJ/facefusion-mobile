package com.facefusion.mobile.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

/**
 * The app's colour and type.
 *
 * The accent is **upstream FaceFusion's own red**, sampled from its web UI: `#EF4444` for
 * a control that is ON, `#DC2626` a step darker for the mark inside it. This replaces a
 * near-white primary that was chosen here on the argument that a neutral accent "cannot
 * clash with whatever is in the user's photo". That argument was not wrong, but it was
 * answering the wrong question: the accent's job is to say WHICH APP THIS IS, and taking
 * it from the project this one ports settles that better than avoiding the issue did.
 *
 * Both schemes exist and the phone chooses. Two things that follow from that and are easy
 * to miss:
 *
 *   * Nothing outside this file names a colour. Every screen reads MaterialTheme, so the
 *     light scheme needed no changes anywhere else -- verified by grep, not by hope.
 *   * The WINDOW is separate. `res/values/themes.xml` paints the frame before the first
 *     composition, so it needs its own light/night pair, and the system-bar icons have to
 *     flip with it or they vanish into their own background.
 *
 * The greys are Tailwind's zinc, which is what upstream's UI uses: the light surfaces
 * measure #FAFAFA, #F4F4F5 and #E4E4E7 in a screenshot of it.
 */

/** ON. Upstream's red-500. */
val FfRed = Color(0xFFEF4444)

/** The mark inside an ON control. Upstream's red-600, one step down so it reads on the red. */
val FfRedDeep = Color(0xFFDC2626)

private val FfLight = lightColorScheme(
    primary = FfRed,
    onPrimary = Color(0xFFFFFFFF),
    primaryContainer = FfRed,
    onPrimaryContainer = Color(0xFFFFFFFF),

    secondary = Color(0xFF52525B),
    onSecondary = Color(0xFFFFFFFF),
    secondaryContainer = Color(0xFFF4F4F5),
    onSecondaryContainer = Color(0xFF27272A),

    background = Color(0xFFFFFFFF),
    onBackground = Color(0xFF18181B),

    surface = Color(0xFFFAFAFA),
    onSurface = Color(0xFF18181B),
    surfaceVariant = Color(0xFFF4F4F5),
    onSurfaceVariant = Color(0xFF52525B),

    outline = Color(0xFFD4D4D8),
    outlineVariant = Color(0xFFE4E4E7),

    error = Color(0xFFDC2626),
    onError = Color(0xFFFFFFFF),
    errorContainer = Color(0xFFFEE2E2),
    onErrorContainer = Color(0xFF7F1D1D),
)

private val FfDark = darkColorScheme(
    primary = FfRed,
    onPrimary = Color(0xFFFFFFFF),
    primaryContainer = FfRed,
    onPrimaryContainer = Color(0xFFFFFFFF),

    secondary = Color(0xFF9AA0B4),
    onSecondary = Color(0xFF16161B),
    secondaryContainer = Color(0xFF23232C),
    onSecondaryContainer = Color(0xFFD5D8E3),

    background = Color(0xFF0D0D10),
    onBackground = Color(0xFFE6E4EC),

    // Cards sit one step above the background; the variant is the step above that, used
    // for preview placeholders and the log box so they read as recessed panels.
    surface = Color(0xFF16161B),
    onSurface = Color(0xFFE6E4EC),
    surfaceVariant = Color(0xFF212129),
    onSurfaceVariant = Color(0xFFA8ADBD),

    outline = Color(0xFF3A3A46),
    outlineVariant = Color(0xFF26262F),

    error = Color(0xFFFF6B6B),
    onError = Color(0xFF1A0A0A),
    errorContainer = Color(0xFF3A1D1D),
    onErrorContainer = Color(0xFFFFD5D5),
)

/**
 * Type.
 *
 * No font file is bundled -- the only faces available offline here are Qualcomm's, which
 * are not ours to ship. So the wordmark is built from weight and letter-spacing on the
 * platform sans instead of a display face. Dropping a licensed .ttf into `res/font/` later
 * changes only [FfTypography] and [WordmarkStyle].
 */
private val FfTypography = Typography(
    titleMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 16.sp,
        letterSpacing = 0.15.sp,
    ),
    labelLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 14.sp,
        letterSpacing = 0.4.sp,
    ),
)

/** The base of the FACEFUSION wordmark; the two weights are applied per-span at the call site. */
val WordmarkStyle = TextStyle(
    fontFamily = FontFamily.SansSerif,
    fontSize = 26.sp,
    // Wide tracking is what makes an all-caps wordmark read as a mark rather than as a
    // shouted sentence.
    letterSpacing = 4.sp,
)

@Composable
fun FaceFusionTheme(content: @Composable () -> Unit) =
    MaterialTheme(
        colorScheme = if (isSystemInDarkTheme()) FfDark else FfLight,
        typography = FfTypography,
        content = content,
    )
