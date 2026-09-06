package com.facefusion.mobile

import android.graphics.Bitmap
import android.net.Uri
import java.io.File

/**
 * One queued target — roadmap 14.
 *
 * The batch is ONE SOURCE, MANY TARGETS. That is the shape a user asks for ("put my face on
 * these twelve clips") and it is the shape the warm pipeline already has: `setSource` is
 * called once and every target reuses it, so the models and the identity are paid for once
 * instead of twelve times. The two alternatives were considered and rejected — many sources
 * against one target is a comparison sheet nobody asked for, and the cartesian product of
 * both is a way to fill a phone by accident.
 *
 * Immutable, and replaced rather than mutated, because Compose observes the LIST: mutating
 * an item in place leaves the list reference equal and the row never redraws.
 */
data class BatchItem(
    val uri: Uri,
    val name: String,
    val state: BatchState = BatchState.Waiting,
    /** Where it landed. Null until it succeeds. */
    val output: File? = null,
    /**
     * Why it did not land, already a finished sentence for the user.
     *
     * A refusal carries the gate's own wording here — it is not an error, and the row that
     * shows it must not offer a bug report for a safety check doing its job.
     */
    val detail: String? = null,
    /**
     * A small frame from [output], for the queue row.
     *
     * Made once, when the clip finishes, on the worker thread that produced it -- a
     * retriever call on the main thread would stutter the list at exactly the moment the
     * next clip starts encoding. Null until then, and null for anything that did not
     * produce a file.
     */
    val thumb: Bitmap? = null,
    /**
     * Where this clip landed in the gallery, once it has been saved.
     *
     * ⚠ PER ITEM, because "saved" is a fact about a clip and the Activity only had room for
     * one of them. Saving clip two by hand and swiping to clip three moved the single
     * `savedUri` with the pane, so clip two's button went back to reading "Save" and
     * offered to write a second copy of a file already in the gallery.
     */
    val savedUri: Uri? = null,
)

/**
 * Where one queued target got to.
 *
 * ⚠ [Refused] is deliberately NOT [Failed]. The content gate blocking a clip is the app
 * working, and a batch of twelve in which one is refused has eleven successes and one
 * correct refusal — not a failure to investigate. They are shown differently and counted
 * separately for that reason.
 */
enum class BatchState { Waiting, Running, Done, Refused, Failed, Skipped }
