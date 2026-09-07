package com.facefusion.mobile

import android.graphics.Bitmap

/**
 * The bitmap as ARGB_8888, without copying one that already is.
 *
 * ⚠ Every path that hands pixels to the native side does the same three steps -- copy to
 * ARGB_8888, `getPixels` into an IntArray, `argbToBgr` -- and the copy was unconditional.
 * On a 50 MP camera photo that is a needless **200 MB**, on top of the 200 MB bitmap, the
 * 200 MB IntArray and the 150 MB BGR buffer. It is half of why big photos crashed the app
 * on an 8 Elite Gen 5.
 *
 * `decodeOriented` asks ImageDecoder for a software, mutable bitmap, which is ARGB_8888
 * already, so in practice this now copies nothing at all. The copy is kept for the cases
 * that are not: a hardware bitmap has no pixel array to read, and RGB_565 would be read
 * back as the wrong colours rather than failing.
 *
 * ⚠ NON-NULL, unlike `Bitmap.copy`, and it falls back to the bitmap itself if the copy
 * fails. That is not a shortcut: `getPixels` works on any config EXCEPT hardware, so a
 * failed copy leaves the caller no worse off than it was, while returning null would turn
 * a memory hiccup into a refusal. Callers that still write `?:` are harmless -- Kotlin
 * warns the branch is unreachable and keeps their old behaviour.
 */
internal fun Bitmap.asArgb8888(): Bitmap =
    if (config == Bitmap.Config.ARGB_8888) this
    else copy(Bitmap.Config.ARGB_8888, false) ?: this
