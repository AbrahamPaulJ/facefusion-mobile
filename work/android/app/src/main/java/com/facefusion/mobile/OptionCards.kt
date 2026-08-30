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
import androidx.compose.ui.res.stringResource

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
    // The model name and the boost are identifiers; only the word "weight" is a word.
    val summary = stringResource(R.string.opt_swapper_summary,
                                 "%.2f".format(opts.weight), opts.pixelBoostLabel, opts.swapper)
    OptionCard(stringResource(R.string.opt_face_swapper), summary, expanded, onToggle) {
        OptionSlider(
            stringResource(R.string.opt_weight), opts.weight,
            { onChange(opts.copy(weight = it)) },
            hint = when {
                opts.weight > 0.55f -> stringResource(R.string.opt_weight_hint_high)
                opts.weight < 0.45f -> stringResource(R.string.opt_weight_hint_low)
                else -> stringResource(R.string.opt_weight_hint_default)
            },
        )
        OptionSegments(
            stringResource(R.string.opt_pixel_boost),
            listOf(1 to "256", 2 to "512", 3 to "768", 4 to "1024"),
            opts.pixelBoost,
            { onChange(opts.copy(pixelBoost = it)) },
            hint = if (opts.pixelBoost == 1)
                       stringResource(R.string.opt_pixel_boost_native)
                   else stringResource(R.string.opt_pixel_boost_cost,
                                       opts.invocationsPerFace),
        )
        if (inswapperAvailable) {
            OptionSegments(
                stringResource(R.string.opt_model),
                listOf("hyperswap" to "hyperswap", "inswapper" to "inswapper"),
                opts.swapper,
                { onChange(opts.copy(swapper = it)) },
                hint = stringResource(R.string.opt_model_hint),
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
    OptionCard(stringResource(R.string.opt_face_masker),
               stringResource(R.string.opt_masker_summary,
                              "%.2f".format(opts.maskBlur), padLabel),
               expanded, onToggle) {
        OptionSlider(stringResource(R.string.opt_blur), opts.maskBlur,
                     { onChange(opts.copy(maskBlur = it)) },
                     hint = stringResource(R.string.opt_blur_hint))

        if (!perEdge) {
            OptionSlider(
                stringResource(R.string.opt_padding), pad[0].toFloat(),
                { onChange(opts.copy(maskPadding = List(4) { _ -> it.roundToInt() })) },
                range = 0f..100f, step = 1f, format = { "${it.roundToInt()}" },
                hint = stringResource(R.string.opt_padding_hint),
            )
        } else {
            listOf(R.string.opt_edge_top, R.string.opt_edge_right,
                   R.string.opt_edge_bottom, R.string.opt_edge_left)
                .map { stringResource(it) }.forEachIndexed { i, name ->
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
            Text(stringResource(R.string.opt_padding_per_edge),
                 style = MaterialTheme.typography.bodyMedium,
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
        stringResource(R.string.opt_face_detector),
        stringResource(R.string.opt_detector_summary,
                       "%.2f".format(opts.detectorScore),
                       "%.2f".format(opts.landmarkerScore),
                       if (opts.largestOnly) stringResource(R.string.opt_faces_largest_short)
                       else stringResource(R.string.opt_faces_every_short)),
        expanded, onToggle,
    ) {
        OptionSlider(stringResource(R.string.opt_find_faces), opts.detectorScore,
                     { onChange(opts.copy(detectorScore = it)) },
                     hint = stringResource(R.string.opt_find_faces_hint))
        // Was "below it, the detector's 5 points are used unrefined", which describes the
        // implementation to someone who already knows it and nothing to anyone else. What
        // the user can actually decide is how well the swap should line up with the face
        // underneath, so that is what the words are about.
        OptionSlider(stringResource(R.string.opt_face_alignment), opts.landmarkerScore,
                     { onChange(opts.copy(landmarkerScore = it)) },
                     hint = stringResource(R.string.opt_face_alignment_hint))
        OptionSegments(
            stringResource(R.string.opt_faces),
            listOf(false to stringResource(R.string.opt_faces_every),
                   true to stringResource(R.string.opt_faces_largest)),
            opts.largestOnly,
            { onChange(opts.copy(largestOnly = it)) },
            hint = stringResource(R.string.opt_faces_hint),
        )
        // detector size (640) and the swapper's 256² input are absent on purpose: both are
        // baked into the context binary at conversion, so they are a rebuild, not a knob.
        Text(stringResource(R.string.opt_detector_note),
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
        stringResource(R.string.opt_face_enhancer),
        if (opts.faceEnhance)
            stringResource(R.string.opt_enhancer_summary, "%.2f".format(opts.enhanceBlend))
        else stringResource(R.string.opt_enhancer_off_summary),
        expanded, onToggle,
    ) {
        // The on/off lives in the Processors row on the Swap screen now. Two controls for
        // one boolean is two places to look and one of them to be surprised by.
        if (opts.faceEnhance) {
            OptionSlider(
                stringResource(R.string.opt_blend), opts.enhanceBlend,
                { onChange(opts.copy(enhanceBlend = it)) },
                hint = when {
                    opts.enhanceBlend >= 0.95f -> stringResource(R.string.opt_blend_hint_full)
                    opts.enhanceBlend <= 0.05f -> stringResource(R.string.opt_blend_hint_none)
                    else -> stringResource(R.string.opt_blend_hint_mixed)
                },
            )
            // It runs on the swapper's own crop: gpen_bfr_256 and hyperswap_1a_256 declare
            // the same template and size, so no second alignment is involved.
            Text(stringResource(R.string.opt_enhancer_note, opts.pixelBoostLabel),
                 style = MaterialTheme.typography.bodySmall, fontSize = 11.sp,
                 modifier = Modifier.padding(top = 8.dp))
        }
    }
}
