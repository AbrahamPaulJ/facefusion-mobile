package com.facefusion.mobile.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

/**
 * The app's colour and type.
 *
 * Previously this was bare `darkColorScheme()`, i.e. Material 3's stock baseline purple.
 * That palette is the one every unstyled Compose app ships with, which is most of why the
 * UI read as a debug harness rather than a product.
 *
 * FaceFusion's own mark is black-and-white, so the scheme is monochrome-forward: the
 * primary is a near-white that puts DARK text on a LIGHT button, which is both the highest
 * contrast available on a near-black ground and the only accent choice that cannot clash
 * with whatever is in the user's photo -- every screen here is dominated by two large
 * images we do not control.
 */
private val FfColors = darkColorScheme(
    // The call to action. Near-white rather than tinted: see above.
    primary = Color(0xFFECEAF0),
    onPrimary = Color(0xFF101014),
    primaryContainer = Color(0xFF2A2A34),
    onPrimaryContainer = Color(0xFFECEAF0),

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
    MaterialTheme(colorScheme = FfColors, typography = FfTypography, content = content)
