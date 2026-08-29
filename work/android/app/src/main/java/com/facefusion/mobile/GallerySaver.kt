package com.facefusion.mobile

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File

/**
 * Copy a finished MP4 into the shared Movies collection so it appears in the gallery.
 *
 * The app writes its output to getExternalFilesDir first, which is private to the app --
 * nothing else on the phone can see it, which is why "saved" was not the same as "findable".
 * MediaStore is the only route to the shared collection that needs no storage permission on
 * API 29+, which is why this is an explicit button rather than something done automatically:
 * it publishes the file to every gallery app on the device.
 */
object GallerySaver {

    fun save(context: Context, file: File, displayName: String = file.name): Result<Uri> =
        runCatching {
            val resolver = context.contentResolver
            val values = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, displayName)
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.Video.Media.RELATIVE_PATH,
                        Environment.DIRECTORY_MOVIES + "/FaceFusion")
                    // IS_PENDING hides the entry until the bytes are written; without it a
                    // gallery can index a zero-length file and cache it as broken.
                    put(MediaStore.Video.Media.IS_PENDING, 1)
                }
            }
            val uri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
                ?: error("MediaStore refused the insert")

            resolver.openOutputStream(uri).use { out ->
                file.inputStream().use { it.copyTo(out!!) }
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.clear()
                values.put(MediaStore.Video.Media.IS_PENDING, 0)
                resolver.update(uri, values, null, null)
            }
            uri
        }

    /**
     * Copy a still into the shared Pictures collection.
     *
     * Same IS_PENDING dance as [save] and for the same reason: without it a gallery can
     * index a zero-length file and cache it as broken, which no later write repairs.
     *
     * PNG, not JPEG. This saves either an image swap or a frame lifted out of a finished
     * video, and both have already been through one lossy encode -- re-encoding a swapped
     * face a second time is where the artefacts everyone notices come from.
     */
    fun saveImage(context: Context, bitmap: Bitmap, displayName: String): Result<Uri> =
        runCatching {
            val resolver = context.contentResolver
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.Images.Media.RELATIVE_PATH,
                        Environment.DIRECTORY_PICTURES + "/FaceFusion")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
            }
            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                ?: error("MediaStore refused the insert")

            resolver.openOutputStream(uri).use { out ->
                if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, out!!))
                    error("PNG encode failed")
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                resolver.update(uri, values, null, null)
            }
            uri
        }
}
