package com.facefusion.mobile.ui

import android.content.Context
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
 * The accent is **neutral monochrome** — no brand hue at all. The requested palette is a
 * light `#F7F8FA` page and a dark `#121212` page with `#FFFFFF` / `#1E1E1E` cards, so the
 * accent resolves to the text colour itself: near-black `#1D1D1D` on the light scheme,
 * near-white `#E4E4E7` on the dark one. A control that is ON is therefore a solid dark
 * (light scheme) or solid light (dark scheme) chip with a contrasting label, and it holds
 * a high-contrast relationship with both schemes without introducing a second hue.
 *
 * The user can pin light or dark in Settings, and the choice survives a restart
 * ([ThemePrefs]); until one is made the phone chooses ([isSystemInDarkTheme]). Two things
 * that follow from that and are easy to miss:
 *
 *   * Nothing outside this file names a colour. Every screen reads MaterialTheme, so the
 *     light scheme needed no changes anywhere else -- verified by grep, not by hope.
 *   * The WINDOW is separate. `res/values/themes.xml` paints the frame before the first
 *     composition, so it needs its own light/night pair, and the system-bar icons have to
 *     flip with it or they vanish into their own background.
 *
 * The two schemes follow the requested palette exactly: dark is `#121212` page on
 * `#1E1E1E` cards with `#E4E4E7` text; light is `#F7F8FA` page on `#FFFFFF` cards with
 * `#1D1D1D` text. Cards and recessed panels share the card colour (controls sit on the
 * card colour too), so the hierarchy is carried by borders and by the contrast of the
 * monochrome accent rather than by a second grey step.
 */

/** ON state / recording / batch-failure red (upstream's red-500), used by UI that is
 *  red only while something is actively wrong or recording. */
val FfRed = Color(0xFFEF4444)

/** The mark inside an ON control. Upstream's red-600, one step down so it reads on the red. */
val FfRedDeep = Color(0xFFDC2626)

private val FfLight = lightColorScheme(
    primary = Color(0xFF1D1D1D),
    onPrimary = Color(0xFFFFFFFF),
    primaryContainer = Color(0xFFE4E4E7),
    onPrimaryContainer = Color(0xFF1D1D1D),

    secondary = Color(0xFF52525B),
    onSecondary = Color(0xFFFFFFFF),
    secondaryContainer = Color(0xFFF2F3F5),
    onSecondaryContainer = Color(0xFF1D1D1D),

    // #F7F8FA page, #FFFFFF cards/controls, #1D1D1D text -- the light half of the
    // requested palette. Secondary text steps one grey down so captions and notes stay
    // readable without shouting.
    background = Color(0xFFF7F8FA),
    onBackground = Color(0xFF1D1D1D),

    // Material3 1.3 reads the surfaceContainer* family for real component backgrounds
    // (NavigationBar, Card, top app bars, menus). Leaving them unset falls back to the
    // baseline purple-grey palette and the light/dark switch looks broken -- cards and
    // the bottom bar stay M3-default instead of tracking the requested scheme. They all
    // resolve to the card colour here, exactly like surface.
    surface = Color(0xFFFFFFFF),
    onSurface = Color(0xFF1D1D1D),
    surfaceContainerLowest = Color(0xFFFFFFFF),
    surfaceContainerLow = Color(0xFFFFFFFF),
    surfaceContainer = Color(0xFFFFFFFF),
    surfaceContainerHigh = Color(0xFFFFFFFF),
    surfaceContainerHighest = Color(0xFFF4F5F7),
    surfaceBright = Color(0xFFFFFFFF),
    surfaceDim = Color(0xFFE9EAEC),
    surfaceTint = Color(0xFF1D1D1D),
    surfaceVariant = Color(0xFFFFFFFF),
    onSurfaceVariant = Color(0xFF6E6E76),

    outline = Color(0xFFD4D4D8),
    outlineVariant = Color(0xFFE4E4E7),

    error = Color(0xFFDC2626),
    onError = Color(0xFFFFFFFF),
    errorContainer = Color(0xFFFEE2E2),
    onErrorContainer = Color(0xFF7F1D1D),
)

private val FfDark = darkColorScheme(
    primary = Color(0xFFE4E4E7),
    onPrimary = Color(0xFF16161B),
    primaryContainer = Color(0xFF3A3A40),
    onPrimaryContainer = Color(0xFFE4E4E7),

    secondary = Color(0xFF9AA0B4),
    onSecondary = Color(0xFF16161B),
    secondaryContainer = Color(0xFF26262E),
    onSecondaryContainer = Color(0xFFE4E4E7),

    // #121212 page, #1E1E1E cards/controls, #E4E4E7 text -- the dark half of the
    // requested palette. Secondary text steps one grey lighter so captions and notes stay
    // readable without glowing.
    background = Color(0xFF121212),
    onBackground = Color(0xFFE4E4E7),

    surface = Color(0xFF1E1E1E),
    onSurface = Color(0xFFE4E4E7),
    surfaceContainerLowest = Color(0xFF161616),
    surfaceContainerLow = Color(0xFF1E1E1E),
    surfaceContainer = Color(0xFF1E1E1E),
    surfaceContainerHigh = Color(0xFF1E1E1E),
    surfaceContainerHighest = Color(0xFF262626),
    surfaceBright = Color(0xFF2A2A2A),
    surfaceDim = Color(0xFF121212),
    surfaceTint = Color(0xFFE4E4E7),
    surfaceVariant = Color(0xFF1E1E1E),
    onSurfaceVariant = Color(0xFF9CA3AF),

    outline = Color(0xFF33333B),
    outlineVariant = Color(0xFF2A2A32),

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

/**
 * The app's theme wrapper.
 *
 * @param darkTheme the user's manual choice, or null to follow the system. The Settings
 *   switch writes a real Boolean; a fresh install passes null until it is touched.
 */
@Composable
fun FaceFusionTheme(
    darkTheme: Boolean? = null,
    content: @Composable () -> Unit,
) {
    val dark = darkTheme ?: isSystemInDarkTheme()
    MaterialTheme(
        colorScheme = if (dark) FfDark else FfLight,
        typography = FfTypography,
        content = content,
    )
}

/**
 * Where the manual light/dark choice lives, when one has been made.
 *
 * `null` means "never touched -- follow the system", which is what a fresh install gets.
 * The Settings switch writes a real Boolean the first time it is used; there is no way
 * back to "follow the system" once a choice exists, and that is deliberate: a switch that
 * could silently stop applying what the user asked for is a switch that lies.
 */
object ThemePrefs {
    private const val FILE = "theme_prefs"
    private const val K_DARK = "dark_mode"

    fun load(context: Context): Boolean? {
        val p = context.getSharedPreferences(FILE, Context.MODE_PRIVATE)
        return if (p.contains(K_DARK)) p.getBoolean(K_DARK, false) else null
    }

    fun save(context: Context, dark: Boolean) {
        context.getSharedPreferences(FILE, Context.MODE_PRIVATE)
            .edit().putBoolean(K_DARK, dark).apply()
    }
}
