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
import com.facefusion.mobile.ApiService
import com.facefusion.mobile.ModelDownload
import androidx.compose.ui.res.stringResource
import com.facefusion.mobile.R
import androidx.compose.runtime.saveable.rememberSaveable

/** One context binary on disk, or one that should be and is not. */
data class ModelRow(
    val label: String,
    val fileName: String,
    val bytes: Long,
    val present: Boolean,
    /** Required models block a run when absent; optional ones only remove a feature. */
    val required: Boolean,
    /**
     * The manifest publishes this file, so an absent one is worth showing WITH a way to
     * get it. An optional model that is not hosted stays hidden when absent: a download
     * button for a file nobody serves is a promise the app cannot keep.
     */
    val downloadable: Boolean = false,
    /**
     * On disk, but not the file the manifest now publishes.
     *
     * Compared by LENGTH, which is exactly what `ModelDownload.missing` re-fetches on, so a
     * row can never say "update available" for something the downloader would then decline
     * to fetch. Without this the app had no notion that a model it already holds could be
     * superseded -- `ModelPaths.missing` is a `canRead` test -- so when the enhancer was
     * republished 9.5x faster, every existing install kept the slow one and was never told.
     */
    val outdated: Boolean = false,
    /**
     * EVERY file behind this row, not just the one it is named after.
     *
     * A QNN model is one context binary and this is a single name; an ncnn model is a
     * `.param`/`.bin` PAIR shown as one row, because a 30 KB param beside a 400 MB bin is
     * not a row of its own. Delete then has to take both -- deleting only [fileName] left
     * the 402 MB hyperswap weights on the device while the inventory reported the model
     * gone, which is the worst of both.
     */
    val files: List<String> = listOf(fileName),
)

/**
 * One model SET, as the inventory shows it.
 *
 * There is more than one on a device that has run both runtimes -- a QNN tier is context
 * binaries named after the arch, the ncnn set is param/bin pairs named after the ONNX graph
 * -- and only one of them is in use at a time. The other used to be invisible: ~600 MB of
 * weights the active build never looks at, with no screen on which they existed and no way
 * to delete them short of switching the runtime back.
 *
 * @param active the set this phone is actually running. Shown open and offering downloads;
 *   the others collapse and only offer deletion, because fetching a set you are not using
 *   is not something to invite.
 */
data class ModelSection(
    val title: String,
    val summary: String,
    val rows: List<ModelRow>,
    val active: Boolean,
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
    /**
     * Which runtime actually came up: "qnn", "ncnn", or "none".
     *
     * Reported SEPARATELY from [ok], which describes the HTP probe. On a part with no
     * Hexagon that probe is meant to fail, and a panel that could only say "the HTP could
     * not be measured" would leave the one user who needs this screen unable to see what
     * their phone is running.
     */
    val backend: String = "",
    /** ncnn only: a usable Vulkan device is present, so the GPU path is available. */
    val gpu: Boolean = false,
)

private fun mb(bytes: Long) = "%.1f MB".format(bytes / 1048576.0)

@Composable
fun SettingsScreen(
    sections: List<ModelSection>,
    modelDirPath: String,
    device: DeviceUi,
    onDeleteModel: (ModelRow) -> Unit,
    /** Fetch whatever the manifest has that this device does not. */
    onDownloadModel: () -> Unit,
    /** Start or stop the HTTP server. [lan] binds every interface instead of loopback. */
    onApiToggle: (on: Boolean, lan: Boolean) -> Unit,
    /**
     * Set the LAN preference. Independent of whether the server is running -- routing this
     * through [onApiToggle] is what used to make the switch inert while it was stopped.
     */
    onApiLan: (Boolean) -> Unit,
    /** Assemble the report and hand it to a share target. */
    onShareBugReport: () -> Unit,
    /** "" | "qnn" | "ncnn" -- which runtime the user has pinned, "" being automatic. */
    forcedBackend: String = "",
    /**
     * Pin the runtime. **null when this build has no second backend**, which is what hides
     * the control entirely rather than drawing one that cannot do anything.
     */
    onForceBackend: ((String) -> Unit)? = null,
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
        //
        // One block per model SET, not one list. A device that has run both runtimes
        // holds two, only one of which is in use -- and the other used to be invisible:
        // ~600 MB of ncnn weights that the NPU build never looks at, with no screen on
        // which they existed and no way to remove them short of switching runtime back.
        //
        // The active set is open and offers downloads. The rest are collapsed and offer
        // only deletion, which is the action they are there for.
        val modelCard: @Composable (List<ModelRow>, Boolean) -> Unit = { rows, active ->
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(vertical = 4.dp)) {
                    rows.forEachIndexed { i, m ->
                        if (i > 0) HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                        Row(
                            Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 10.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Column(Modifier.weight(1f)) {
                                Text(m.label, style = MaterialTheme.typography.bodyMedium)
                                Text(
                                    // ONE phrase for one state.  "missing" for required and
                                    // "not installed" for optional read as two different
                                    // STATES when they are the same state at two severities --
                                    // and the severity was already carried by the colour
                                    // below.  The suffix says it in words instead, because
                                    // colour alone is not something every reader gets.
                                    // The FILENAME is never translated -- it is the name
                                    // on disk and in the manifest.
                                    if (m.outdated)
                                        m.fileName + "   " + mb(m.bytes) + "   " +
                                            stringResource(R.string.set_update_available)
                                    else if (m.present) m.fileName + "   " + mb(m.bytes)
                                    else m.fileName + "   " +
                                         stringResource(R.string.set_not_installed) +
                                         (if (m.required)
                                              " " + stringResource(R.string.set_required_suffix)
                                          else ""),
                                    style = MaterialTheme.typography.bodySmall,
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 10.sp,
                                    color = if (m.outdated) MaterialTheme.colorScheme.primary
                                    else if (m.present || !m.required)
                                        MaterialTheme.colorScheme.onSurfaceVariant
                                    else MaterialTheme.colorScheme.error,
                                )
                            }
                            // ONE kind of control for both directions. A trash icon on one
                            // row and a "Download" link on the next made two halves of the
                            // same decision look like different kinds of thing, and the icon
                            // was the destructive one -- the half that should read loudest.
                            if (m.outdated) {
                                // Still ONE control, still pointing the way the row needs to
                                // go. Delete is not offered here: the file works, it is merely
                                // superseded, and the useful action is to replace it. Deleting
                                // it first would reach the same place through a broken app.
                                TextButton(onDownloadModel, enabled = !ModelDownload.running) {
                                    Text(stringResource(if (ModelDownload.running)
                                                           R.string.set_updating
                                                       else R.string.set_update))
                                }
                            } else if (m.present) {
                                TextButton({ confirming = m }, enabled = !ModelDownload.running) {
                                    Text(stringResource(R.string.set_delete),
                                         color = MaterialTheme.colorScheme.error)
                                }
                            } else if (m.downloadable && active) {
                                TextButton(onDownloadModel, enabled = !ModelDownload.running) {
                                    Text(stringResource(if (ModelDownload.running)
                                                           R.string.set_downloading
                                                       else R.string.set_download))
                                }
                            }
                        }
                    }
                }
            }
        }

        sections.forEach { section ->
            val onDisk = section.rows.filter { it.present }.sumOf { it.bytes }
            val summary = stringResource(R.string.set_models_summary,
                                         section.rows.count { it.present },
                                         section.rows.size, mb(onDisk)) +
                (section.rows.count { it.outdated }
                    .let { if (it > 0) stringResource(R.string.set_models_updates, it) else "" })
            if (section.active) {
                Caption(section.title)
                Text(
                    summary,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                modelCard(section.rows, true)
            } else {
                // Collapsed by default. It is storage the user is not using, so it
                // should be findable rather than prominent.
                var open by rememberSaveable(section.title) { mutableStateOf(false) }
                Accordion(section.title, summary, open, { open = !open }) {
                    Text(
                        stringResource(R.string.set_models_inactive_note),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(Modifier.height(8.dp))
                    modelCard(section.rows, false)
                }
            }
            Spacer(Modifier.height(6.dp))
        }

        Spacer(Modifier.height(6.dp))

        // ---------------------------------------------------------------- device
        Caption(stringResource(R.string.set_this_device))
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                // The RUNTIME first, because on a non-Qualcomm phone it is the only row on
                // this card about the machine the user is actually holding. Everything
                // below it describes a Hexagon NPU, which such a device does not have.
                InfoRow(
                    stringResource(R.string.set_runtime), stringResource(when (device.backend) {
                        "qnn" -> R.string.set_runtime_npu
                        "ncnn" -> if (device.gpu) R.string.set_runtime_gpu_cpu
                                  else R.string.set_runtime_cpu
                        "none" -> R.string.set_runtime_none
                        else -> R.string.set_unknown_dash
                    })
                )
                if (device.backend == "ncnn") {
                    // Say the two things a preview user needs before they reach for a
                    // stopwatch or file a report, and say them here rather than in a
                    // release note nobody has open.
                    Text(
                        stringResource(R.string.set_ncnn_note),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                } else if (!device.ok) {
                    Text(
                        stringResource(R.string.set_htp_unmeasured),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                } else {
                    InfoRow(stringResource(R.string.set_hexagon_arch), "v${device.arch}")
                    InfoRow(stringResource(R.string.set_vtcm), "${device.vtcmMb} MB")
                    InfoRow(stringResource(R.string.set_soc_model), "${device.soc}")
                }
                // Both rows below are QNN facts. On ncnn there is no context binary and no
                // fp16 canary was ever run -- printing "could not be measured" there would
                // report a failure where nothing was attempted.
                if (device.backend != "ncnn") {
                    InfoRow(
                        stringResource(R.string.set_fp16), stringResource(when (device.fp16) {
                            "yes" -> R.string.set_fp16_yes
                            "no" -> R.string.set_fp16_no
                            // The control canary failed. Reporting this as "no" would push
                            // a working device onto the slower compatibility build.
                            else -> R.string.set_fp16_unknown
                        })
                    )
                    // The tier is an identifier, not a word: "v79" reads the same in
                    // every language and is what a bug report has to quote.
                    InfoRow(stringResource(R.string.set_context_binaries),
                            if (device.tier.isEmpty()) stringResource(R.string.set_unknown_dash)
                            else device.tier)
                }
            }
        }

        // -------------------------------------------------------------- runtime override
        //
        // Shown when ncnn is LINKED and either there is a choice to make or one has already
        // been made. In a QNN-only build the control could do nothing, and on a genuine
        // non-Qualcomm phone running automatically there is nothing to choose between.
        //
        // It exists because the non-Qualcomm path is otherwise untestable: Auto tries QNN
        // first and QNN wins on every device this project owns, so without a switch the
        // ncnn path could only ever be exercised on hardware that is not on the bench.
        //
        // ⚠ `|| forcedBackend.isNotEmpty()` is the whole fix, and the bug it closes was a
        // ONE-WAY DOOR. The test used to be `device.backend == "qnn"` alone -- which is the
        // ACTIVE runtime, not the available one -- so the moment you pinned GPU + CPU the
        // card that did the pinning disappeared, and nothing in the app could undo it. The
        // second case is worse: pinning NPU on a phone that has none leaves `backend` at
        // "none", the pipeline unable to start, AND no control to recover with. Clearing
        // app storage was the only way out, and that takes the ~600 MB of models with it.
        //
        // So: if a runtime is pinned, the control that unpins it is always reachable.
        if (onForceBackend != null && (device.backend == "qnn" || forcedBackend.isNotEmpty())) {
            Spacer(Modifier.height(6.dp))
            Caption(stringResource(R.string.set_runtime))
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp),
                       verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        // The override note assumes the phone is running on its NPU, which
                        // is untrue the moment anything is pinned -- and that is exactly
                        // when the reader needs to be told how to get back.
                        stringResource(if (forcedBackend.isEmpty())
                                           R.string.set_runtime_override_note
                                       else R.string.set_runtime_pinned_note),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        // "" is automatic, and it is what a fresh install has.
                        listOf("" to stringResource(R.string.set_runtime_auto),
                               "qnn" to stringResource(R.string.set_runtime_force_npu),
                               "ncnn" to stringResource(R.string.set_runtime_force_gpu))
                            .forEach { (value, label) ->
                                FilterChip(
                                    selected = forcedBackend == value,
                                    onClick = { onForceBackend(value) },
                                    label = { Text(label) },
                                )
                            }
                    }
                    if (forcedBackend.isNotEmpty()) Text(
                        stringResource(R.string.set_runtime_forced, forcedBackend),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                }
            }
        }

        // ---------------------------------------------------------------- remote API
        //
        // Reads ApiService's state directly, the way the download overlay reads
        // ModelDownload's: the service owns it, and threading it through the Activity would
        // only add a copy that can be stale.
        Caption(stringResource(R.string.set_remote_api))
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(stringResource(R.string.set_api_title),
                             style = MaterialTheme.typography.bodyMedium)
                        Text(
                            stringResource(R.string.set_api_body),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Switch(
                        checked = ApiService.running,
                        onCheckedChange = { onApiToggle(it, ApiService.allowLan) },
                    )
                }

                // Changing this restarts the server: the address is fixed when the socket
                // opens, so a live switch would name an address it is not listening on.
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(stringResource(R.string.set_api_lan_title),
                             style = MaterialTheme.typography.bodyMedium)
                        // The red warning belongs to an OPEN PORT, not to a preference.
                        // It used to appear whenever `allowLan` was true -- including on a
                        // fresh launch, where `restore` sets it from prefs and nothing is
                        // listening at all. A network-exposure warning about a server that
                        // is not running is alarming and untrue.
                        val exposed = ApiService.allowLan && ApiService.running
                        Text(
                            stringResource(when {
                                exposed -> R.string.set_api_lan_on
                                ApiService.allowLan -> R.string.set_api_lan_on_pending
                                else -> R.string.set_api_lan_off
                            }),
                            style = MaterialTheme.typography.bodySmall,
                            color = if (exposed) MaterialTheme.colorScheme.error
                                    else MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Switch(
                        checked = ApiService.allowLan,
                        onCheckedChange = onApiLan,
                    )
                }

                if (ApiService.running)
                    InfoRow(stringResource(R.string.set_api_open_this), ApiService.address)
                if (!ApiService.allowLan) InfoRow(
                    stringResource(R.string.set_api_over_usb), "adb forward tcp:8760 tcp:8760")
                ApiService.error?.let {
                    Text(stringResource(R.string.set_api_failed, it),
                         style = MaterialTheme.typography.bodySmall,
                         color = MaterialTheme.colorScheme.error)
                }

                if (ApiService.log.isNotEmpty()) Text(
                    ApiService.log.trimEnd().lines().takeLast(4).joinToString(System.lineSeparator()),
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        Caption(stringResource(R.string.set_supported_devices))
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(vertical = 4.dp)) {
                TIERS.forEachIndexed { i, (tier, chipsRes) ->
                    val chips = stringResource(chipsRes)
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
                                stringResource(R.string.set_in_use),
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.primary,
                            )
                    }
                }
            }
        }
        Spacer(Modifier.height(6.dp))

        // ---------------------------------------------------------------- about
        // -------------------------------------------------------------- bug report
        //
        // ⚠ This card is why the whole feature is not a feature. The README, the release
        // notes and the "Tested on" box have all told people for four releases to use
        // "Settings > Share bug report" -- and there has never been a control here. The
        // only one in the app was on the Swap screen, and it appeared solely when the
        // status line began with the word "Failed". So the reports this project most needs,
        // from devices it does not own and cannot reproduce on, were the ones with no
        // button: a swap that finishes and looks wrong is not a failure, and neither is
        // "Cannot read video".
        //
        // Reported by a user who went looking for it exactly where the documentation said.
        Caption(stringResource(R.string.set_bug_report))
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    stringResource(R.string.set_bug_report_body),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                OutlinedButton(onShareBugReport, Modifier.fillMaxWidth()) {
                    Text(stringResource(R.string.set_bug_report_button))
                }
            }
        }
        Spacer(Modifier.height(6.dp))

        Caption(stringResource(R.string.set_about))
        val uris = LocalUriHandler.current
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    stringResource(R.string.set_about_body),
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    stringResource(R.string.set_about_author),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                OutlinedButton(
                    { uris.openUri("https://github.com/AbrahamPaulJ/facefusion-mobile") },
                    Modifier.fillMaxWidth(),
                ) { Text(stringResource(R.string.set_link_project)) }
                OutlinedButton(
                    { uris.openUri("https://github.com/facefusion/facefusion") },
                    Modifier.fillMaxWidth(),
                ) { Text(stringResource(R.string.set_link_upstream)) }
                OutlinedButton(
                    { uris.openUri("https://github.com/facefusion/facefusion/blob/master/LICENSE.md") },
                    Modifier.fillMaxWidth(),
                ) { Text(stringResource(R.string.set_link_licence)) }
                Text(
                    stringResource(R.string.set_licence_note),
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
            title = { Text(stringResource(R.string.set_delete_title, m.label)) },
            text = {
                Text(
                    stringResource(R.string.set_delete_body, m.fileName, mb(m.bytes)) +
                        if (m.required) stringResource(R.string.set_delete_required_warning)
                        else ""
                )
            },
            confirmButton = {
                TextButton({ onDeleteModel(m); confirming = null }) {
                    Text(stringResource(R.string.set_delete),
                         color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton({ confirming = null }) { Text(stringResource(R.string.set_cancel)) }
            },
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
/**
 * Arch tier -> which chips it covers.
 *
 * The tier and the CHIP NAMES are identifiers and stay as they are in every language; the
 * connecting prose ("and older", "or any part with under 8 MB VTCM") is a sentence and does
 * not. Hence a resource per row rather than one big translated blob: a translator who
 * rewrote "SM8750" would break the one row a user matches their phone against.
 */
private val TIERS = listOf(
    "v68" to R.string.tier_v68,
    "v73" to R.string.tier_v73,
    "v79" to R.string.tier_v79,
    "v81" to R.string.tier_v81,
)
