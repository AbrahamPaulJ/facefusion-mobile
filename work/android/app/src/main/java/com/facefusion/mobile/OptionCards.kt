package com.facefusion.mobile

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.facefusion.mobile.ui.Accordion
import kotlin.math.roundToInt

/**
 * FaceFusion's option groups, laid out for a phone.
 *
 * Upstream's UI is a Gradio two-column desktop page: a tall left rail of always-open
 * accordions beside a preview. That does not survive the transfer -- on a 6" screen an
 * always-open rail buries the swap button under two screens of sliders. So the *grouping*
 * and every name, range and step are upstream's, while the presentation is not:
 *
 *  * one collapsed card per group, so the primary flow (source, target, trim, swap) stays
 *    on one screen and options are opt-in;
 *  * each card header shows its current values, so a collapsed card still answers "what is
 *    this set to" without a tap;
 *  * a slider row is label + value on one line with the track beneath it, because a
 *    label-track-value row leaves the track too narrow to hit accurately with a thumb;
 *  * mask padding is one slider for all four edges by default, with the per-edge sliders
 *    behind a toggle. Upstream shows a 4-handle range slider, which is not usable here.
 */

@Composable
fun OptionCard(
    title: String,
    summary: String,
    expanded: Boolean,
    onToggle: () -> Unit,
    content: @Composable ColumnScope.() -> Unit,
) = Accordion(title, summary, expanded, onToggle, content = content)

/** A labelled slider. Value sits beside the label; the track gets the full width. */
@Composable
fun OptionSlider(
    label: String,
    value: Float,
    onChange: (Float) -> Unit,
    range: ClosedFloatingPointRange<Float> = 0f..1f,
    step: Float = 0.05f,
    format: (Float) -> String = { "%.2f".format(it) },
    hint: String? = null,
    enabled: Boolean = true,
) {
    // Compose counts the INTERIOR stops, so a 0..1 slider in 0.05 steps has 19, not 21.
    val steps = ((range.endInclusive - range.start) / step).roundToInt() - 1
    Column(Modifier.padding(top = 6.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(label, style = MaterialTheme.typography.bodyMedium,
                 modifier = Modifier.weight(1f))
            Text(format(value), style = MaterialTheme.typography.bodyMedium,
                 fontFamily = FontFamily.Monospace)
        }
        Slider(
            value = value,
            onValueChange = onChange,
            valueRange = range,
            steps = if (steps > 0) steps else 0,
            enabled = enabled,
            modifier = Modifier.fillMaxWidth().height(28.dp),
        )
        if (hint != null)
            Text(hint, style = MaterialTheme.typography.bodySmall, fontSize = 11.sp)
    }
}

/** A horizontal segmented picker. Better than a dropdown for two to four short options. */
@Composable
fun <T> OptionSegments(
    label: String,
    options: List<Pair<T, String>>,
    selected: T,
    onSelect: (T) -> Unit,
    hint: String? = null,
) {
    Column(Modifier.padding(top = 8.dp)) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth().padding(top = 4.dp)) {
            options.forEachIndexed { i, (value, text) ->
                SegmentedButton(
                    selected = value == selected,
                    onClick = { onSelect(value) },
                    shape = SegmentedButtonDefaults.itemShape(i, options.size),
                ) { Text(text, maxLines = 1, textAlign = TextAlign.Center, fontSize = 13.sp) }
            }
        }
        if (hint != null)
            Text(hint, style = MaterialTheme.typography.bodySmall, fontSize = 11.sp,
                 modifier = Modifier.padding(top = 2.dp))
    }
}

// ---------------------------------------------------------------- the groups

@Composable
fun FaceSwapperCard(
    opts: SwapOptions,
    onChange: (SwapOptions) -> Unit,
    expanded: Boolean,
    onToggle: () -> Unit,
    inswapperAvailable: Boolean,
) {
    // The value the user is actually here for goes in the summary.
    val summary = "weight %.2f   %s   %s"
        .format(opts.weight, opts.pixelBoostLabel, opts.swapper)
    OptionCard("Face Swapper", summary, expanded, onToggle) {
        OptionSlider(
            "Weight", opts.weight, { onChange(opts.copy(weight = it)) },
            hint = when {
                opts.weight > 0.55f -> "source amplified, target identity subtracted"
                opts.weight < 0.45f -> "some of the target's own identity kept"
                else -> "the source embedding, unmodified (default)"
            },
        )
        OptionSegments(
            "Pixel boost",
            listOf(1 to "256", 2 to "512", 3 to "768", 4 to "1024"),
            opts.pixelBoost,
            { onChange(opts.copy(pixelBoost = it)) },
            hint = if (opts.pixelBoost == 1) "the model's native size"
                   else "${opts.invocationsPerFace}x the swapper cost per face — " +
                        "fine for stills, slow for video",
        )
        if (inswapperAvailable) {
            OptionSegments(
                "Model",
                listOf("hyperswap" to "hyperswap", "inswapper" to "inswapper"),
                opts.swapper,
                { onChange(opts.copy(swapper = it)) },
                hint = "hyperswap is 256², inswapper 128² and slower",
            )
        }
    }
}

@Composable
fun FaceMaskerCard(
    opts: SwapOptions,
    onChange: (SwapOptions) -> Unit,
    expanded: Boolean,
    onToggle: () -> Unit,
) {
    var perEdge by rememberSaveable { mutableStateOf(opts.maskPadding.distinct().size > 1) }
    val pad = opts.maskPadding
    val padLabel = if (pad.distinct().size == 1) "${pad[0]}" else pad.joinToString("/")
    OptionCard("Face Masker", "blur %.2f   padding %s".format(opts.maskBlur, padLabel),
               expanded, onToggle) {
        OptionSlider("Blur", opts.maskBlur, { onChange(opts.copy(maskBlur = it)) },
                     hint = "how soft the edge of the swapped region is")

        if (!perEdge) {
            OptionSlider(
                "Padding", pad[0].toFloat(),
                { onChange(opts.copy(maskPadding = List(4) { _ -> it.roundToInt() })) },
                range = 0f..100f, step = 1f, format = { "${it.roundToInt()}" },
                hint = "shrinks the swapped region, letting more of the original show",
            )
        } else {
            listOf("Top", "Right", "Bottom", "Left").forEachIndexed { i, name ->
                OptionSlider(
                    name, pad[i].toFloat(),
                    { v ->
                        onChange(opts.copy(maskPadding =
                            pad.toMutableList().also { it[i] = v.roundToInt() }))
                    },
                    range = 0f..100f, step = 1f, format = { "${it.roundToInt()}" },
                )
            }
        }
        Row(Modifier.fillMaxWidth().padding(top = 4.dp),
            verticalAlignment = Alignment.CenterVertically) {
            Text("Per-edge padding", style = MaterialTheme.typography.bodyMedium,
                 modifier = Modifier.weight(1f))
            Switch(checked = perEdge, onCheckedChange = { on ->
                perEdge = on
                // Collapsing four values into one has to pick a survivor; the largest is
                // the safe one, since padding only ever removes area.
                if (!on) onChange(opts.copy(maskPadding = List(4) { _ -> pad.max() }))
            })
        }
    }
}

@Composable
fun FaceDetectorCard(
    opts: SwapOptions,
    onChange: (SwapOptions) -> Unit,
    expanded: Boolean,
    onToggle: () -> Unit,
) {
    OptionCard(
        "Face Detector",
        "score %.2f   landmarker %.2f   %s"
            .format(opts.detectorScore, opts.landmarkerScore,
                    if (opts.largestOnly) "largest face" else "every face"),
        expanded, onToggle,
    ) {
        OptionSlider("Detector score", opts.detectorScore,
                     { onChange(opts.copy(detectorScore = it)) },
                     hint = "raise it to ignore uncertain detections")
        OptionSlider("Landmarker score", opts.landmarkerScore,
                     { onChange(opts.copy(landmarkerScore = it)) },
                     hint = "below it, the detector's 5 points are used unrefined")
        OptionSegments(
            "Faces",
            listOf(false to "Every face", true to "Largest only"),
            opts.largestOnly,
            { onChange(opts.copy(largestOnly = it)) },
            hint = "swap every face in the frame, or only the largest one",
        )
        // detector size (640) and the swapper's 256² input are absent on purpose: both are
        // baked into the context binary at conversion, so they are a rebuild, not a knob.
        Text("detector size is fixed at 640 and the swapper at 256 — both are compiled " +
             "into the NPU binaries",
             style = MaterialTheme.typography.bodySmall, fontSize = 11.sp,
             modifier = Modifier.padding(top = 8.dp))
    }
}

/**
 * `--face-enhancer`, gpen_bfr_256.
 *
 * Composed ONLY when the context binary is on the device -- same rule the inswapper choice
 * follows. A switch for a model that is not there turns a missing file into a failed run at
 * the one moment the user is least able to do anything about it.
 *
 * Off by default: it is 8.57 GMAC per face on top of the swapper's 31.93, and on video that
 * is a cost the user should opt into rather than discover.
 */
@Composable
fun FaceEnhancerCard(
    opts: SwapOptions,
    onChange: (SwapOptions) -> Unit,
    expanded: Boolean,
    onToggle: () -> Unit,
) {
    OptionCard(
        "Face Enhancer",
        if (opts.faceEnhance) "gpen_bfr_256   blend %.2f".format(opts.enhanceBlend) else "off",
        expanded, onToggle,
    ) {
        OptionSegments(
            "Enhancer",
            listOf(false to "Off", true to "On"),
            opts.faceEnhance,
            { onChange(opts.copy(faceEnhance = it)) },
            hint = "restores detail the swapper loses — about +4 ms per face",
        )
        if (opts.faceEnhance) {
            OptionSlider(
                "Blend", opts.enhanceBlend, { onChange(opts.copy(enhanceBlend = it)) },
                hint = when {
                    opts.enhanceBlend >= 0.95f -> "the enhancer's output alone"
                    opts.enhanceBlend <= 0.05f -> "no effect — the swap is left untouched"
                    else -> "mixed with the unenhanced swap (upstream default 0.80)"
                },
            )
            // It runs on the swapper's own crop: gpen_bfr_256 and hyperswap_1a_256 declare
            // the same template and size, so no second alignment is involved.
            Text("runs on the swapped face at ${opts.pixelBoostLabel}, before it is pasted back",
                 style = MaterialTheme.typography.bodySmall, fontSize = 11.sp,
                 modifier = Modifier.padding(top = 8.dp))
        }
    }
}
