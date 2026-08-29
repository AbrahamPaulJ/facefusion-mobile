package com.facefusion.mobile.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** One context binary on disk, or one that should be and is not. */
data class ModelRow(
    val label: String,
    val fileName: String,
    val bytes: Long,
    val present: Boolean,
    /** Required models block a run when absent; optional ones only remove a feature. */
    val required: Boolean,
)

/** What the HTP said about itself, already parsed out of `NativePipe.probeDeviceInfo`. */
data class DeviceUi(
    val ok: Boolean = false,
    val arch: Int = 0,
    val vtcmMb: Int = 0,
    val soc: Int = 0,
    val tier: String = "",
    /** "yes" | "no" | "unknown" -- and "unknown" must never be shown as "no". */
    val fp16: String = "unknown",
)

private fun mb(bytes: Long) = "%.1f MB".format(bytes / 1048576.0)

@Composable
fun SettingsScreen(
    models: List<ModelRow>,
    modelDirPath: String,
    device: DeviceUi,
    onDeleteModel: (ModelRow) -> Unit,
    modifier: Modifier = Modifier,
) {
    var confirming by remember { mutableStateOf<ModelRow?>(null) }

    Column(
        modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        // ---------------------------------------------------------------- models
        Caption("Models")
        val onDisk = models.filter { it.present }.sumOf { it.bytes }
        Text(
            "${models.count { it.present }} of ${models.size} present, ${mb(onDisk)} on disk",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(vertical = 4.dp)) {
                models.forEachIndexed { i, m ->
                    if (i > 0) HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                    Row(
                        Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(m.label, style = MaterialTheme.typography.bodyMedium)
                            Text(
                                if (m.present) "${m.fileName}   ${mb(m.bytes)}"
                                else m.fileName + (if (m.required) "   missing" else "   not installed"),
                                style = MaterialTheme.typography.bodySmall,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 10.sp,
                                color = if (m.present || !m.required)
                                    MaterialTheme.colorScheme.onSurfaceVariant
                                else MaterialTheme.colorScheme.error,
                            )
                        }
                        if (m.present) {
                            IconButton({ confirming = m }, Modifier.size(32.dp)) {
                                Icon(
                                    Icons.Default.Delete, "Delete ${m.label}",
                                    Modifier.size(18.dp),
                                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                    }
                }
            }
        }

        Spacer(Modifier.height(6.dp))

        // ---------------------------------------------------------------- device
        Caption("This device")
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                if (!device.ok) {
                    Text(
                        "The HTP could not be measured. The app falls back to the most " +
                            "permissive build, so this is not necessarily a failure to run.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                } else {
                    InfoRow("Hexagon arch", "v${device.arch}")
                    InfoRow("VTCM", "${device.vtcmMb} MB")
                    InfoRow("SoC model", "${device.soc}")
                }
                InfoRow(
                    "fp16 execution", when (device.fp16) {
                        "yes" -> "supported"
                        "no" -> "not supported"
                        // The control canary failed. Reporting this as "no" would push a
                        // working device onto the slower compatibility build.
                        else -> "could not be measured"
                    }
                )
                InfoRow("Context binaries", if (device.tier.isEmpty()) "-" else device.tier)
            }
        }

        Caption("Supported devices")
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(vertical = 4.dp)) {
                TIERS.forEachIndexed { i, (tier, chips) ->
                    if (i > 0) HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                    val active = tier == device.tier
                    Row(
                        Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 9.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            tier,
                            style = MaterialTheme.typography.bodyMedium,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = if (active) FontWeight.Bold else FontWeight.Normal,
                            color = if (active) MaterialTheme.colorScheme.onSurface
                            else MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.width(46.dp),
                        )
                        Text(
                            chips,
                            style = MaterialTheme.typography.bodySmall,
                            fontSize = 11.sp,
                            color = if (active) MaterialTheme.colorScheme.onSurface
                            else MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.weight(1f),
                        )
                        if (active)
                            Text(
                                "in use",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.primary,
                            )
                    }
                }
            }
        }
        Spacer(Modifier.height(6.dp))

        // ---------------------------------------------------------------- about
        Caption("About")
        val uris = LocalUriHandler.current
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    "This app is an offline port of FaceFusion's default face-swap path to " +
                        "the Qualcomm Hexagon NPU. The pipeline, the models, the option " +
                        "names, defaults and ranges are all FaceFusion's.",
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    "Created by Henry Ruhs.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                OutlinedButton(
                    { uris.openUri("https://github.com/AbrahamPaulJ/facefusion-mobile") },
                    Modifier.fillMaxWidth(),
                ) { Text("This project on GitHub") }
                OutlinedButton(
                    { uris.openUri("https://github.com/facefusion/facefusion") },
                    Modifier.fillMaxWidth(),
                ) { Text("FaceFusion (original) on GitHub") }
                OutlinedButton(
                    { uris.openUri("https://github.com/facefusion/facefusion/blob/master/LICENSE.md") },
                    Modifier.fillMaxWidth(),
                ) { Text("Upstream licence (OpenRAIL-AS)") }
                Text(
                    "Model licences differ and are not all permissive: yoloface_8n is " +
                        "GPL-3.0, arcface_w600k_r50 and inswapper_128 are non-commercial, " +
                        "and hyperswap_1a_256 is ResearchRAIL.",
                    style = MaterialTheme.typography.bodySmall,
                    fontSize = 11.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        Spacer(Modifier.height(16.dp))
    }

    confirming?.let { m ->
        AlertDialog(
            onDismissRequest = { confirming = null },
            title = { Text("Delete ${m.label}?") },
            text = {
                Text(
                    "${m.fileName} is ${mb(m.bytes)}. Deleting it frees that space; the app " +
                        "can download it again when you next swap." +
                        if (m.required) "\n\nSwapping will not work until it is back." else ""
                )
            },
            confirmButton = {
                TextButton({ onDeleteModel(m); confirming = null }) {
                    Text("Delete", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = { TextButton({ confirming = null }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(
            label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f),
        )
        Text(value, style = MaterialTheme.typography.bodyMedium, fontFamily = FontFamily.Monospace)
    }
}

/**
 * The tier map, matching `ffqnn::pickTier`.
 *
 * ⚠ If pickTier changes, this table has to change with it. The CLI's `--probe` asserts the
 * mapping against a 9-case synthetic table; this is only its human-readable form.
 */
private val TIERS = listOf(
    "v68" to "Snapdragon 888 and older (v68), 8 Gen 1 (v69), or any part with under 8 MB VTCM",
    "v73" to "8 Gen 2 (v73), 8 Gen 3 (v75), and v79 parts other than SM8750",
    "v79" to "Snapdragon 8 Elite (SM8750)",
    "v81" to "Snapdragon 8 Elite Gen 5 (v81) and newer",
)
