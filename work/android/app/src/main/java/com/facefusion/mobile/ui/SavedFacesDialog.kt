package com.facefusion.mobile.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.facefusion.mobile.R
import java.text.DateFormat
import java.util.Date

/**
 * The faces this app is keeping, one per row, with the two things you can do to one.
 *
 * ⚠ **Use** goes through the caller's normal source entry point, not around it. A face
 * arriving from here is a source like any other and is examined wherever a pipeline is
 * built; a shortcut that set the source directly would be a source that reached the
 * pipeline by a route nothing else takes.
 *
 * Deliberately a list of ROWS rather than the grid of circles the pick rows use. This is
 * where a face is deleted, and a grid of near-identical thumbnails is the layout in which
 * you delete the wrong one -- a row gives each face a date and puts the destructive
 * control at arm's length from the harmless one.
 */
@Composable
fun SavedFacesDialog(
    faces: List<SavedFace>,
    onUse: (SavedFace) -> Unit,
    onDelete: (SavedFace) -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.faces_title)) },
        text = {
            if (faces.isEmpty()) {
                // The empty state says how one gets here, because the button that opens
                // this dialog is visible before anything has ever been kept.
                Text(
                    stringResource(R.string.faces_empty),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                Column(
                    Modifier.heightIn(max = 380.dp).verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    faces.forEach { face ->
                        Row(
                            Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Image(
                                face.thumb.asImageBitmap(),
                                contentDescription = null,
                                contentScale = ContentScale.Crop,
                                modifier = Modifier.size(44.dp).clip(CircleShape),
                            )
                            Spacer(Modifier.width(10.dp))
                            Text(
                                DateFormat.getDateInstance(DateFormat.MEDIUM)
                                    .format(Date(face.addedMs)),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.weight(1f),
                            )
                            TextButton({ onUse(face) }) {
                                Text(stringResource(R.string.faces_use))
                            }
                            TextButton({ onDelete(face) }) {
                                Text(stringResource(R.string.set_delete),
                                     color = MaterialTheme.colorScheme.error)
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.swap_close))
            }
        },
    )
}
