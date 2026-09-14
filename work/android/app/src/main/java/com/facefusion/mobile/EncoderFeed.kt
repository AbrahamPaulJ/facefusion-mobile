package com.facefusion.mobile

import android.media.MediaCodec

/**
 * Hand one BGR frame to a video encoder through its own input buffer [ix].
 *
 * Shared by [VideoSwapper] (the file swap) and [LiveRecorder] (the camera), because the two
 * had the same body twice and the bug below was in both copies.
 *
 * ⚠ **Through `getInputImage`, not `getInputBuffer` + a packed I420 blob.**
 * `COLOR_FormatYUV420Flexible` does not imply I420, and plenty of encoders are semi-planar.
 * Writing I420 into one of those puts luma in the right place and chroma in the wrong one --
 * a greyscale picture with green and pink blobs. The Image view exposes the real strides, so
 * both layouts work. The packed blob stays as the fallback for a codec that offers no Image.
 *
 * ⚠ **The queued SIZE is clamped to what the buffer actually holds.** It used to be
 * `rowStride * h * 3 / 2` and nothing else, which assumes the input buffer is exactly the
 * luma stride times the frame height times 1.5. That is true of this bench's encoder and is
 * not a rule: a component whose Image is strided more widely than its buffer is allocated
 * makes `queueInputBuffer` throw out of `native_queueInputBuffer` -- reported from the field
 * on a non-Qualcomm phone, mid-run, with the trace pointing at this line and nothing in it
 * saying which number was wrong. The planes are bounds-checked on the native side
 * (`bgrToImagePlanes` clips every write against `GetDirectBufferCapacity`), so the clamp
 * only ever loses padding nobody wrote.
 *
 * [h] is used as-is for the pixel loop and rounded DOWN to even for the size: a 4:2:0 chroma
 * plane is half-height, and a caller whose frame is odd has configured its encoder for the
 * even size (see `LiveRecorder.ensure`).
 *
 * Throws whatever MediaCodec throws. Callers wrap it with [codecWhy]: a CodecException's
 * `message` is routinely empty, and the vendor's `diagnosticInfo` is the only part worth
 * reading.
 */
internal fun queueBgrFrame(
    enc: MediaCodec,
    ix: Int,
    bgr: ByteArray,
    w: Int,
    h: Int,
    ptsUs: Long,
    onLog: (String) -> Unit = {},
) {
    // Asked for FIRST and defensively: this is the only honest statement of how much room
    // there is, and a codec that offers an Image is not obliged to offer a buffer too.
    val cap = runCatching { enc.getInputBuffer(ix)?.capacity() ?: 0 }.getOrDefault(0)
    val img = runCatching { enc.getInputImage(ix) }.getOrNull()
    if (img != null) {
        val p = img.planes
        val ok = NativePipe.bgrToImagePlanes(
            bgr, w, h,
            p[0].buffer, p[0].rowStride, p[0].pixelStride,
            p[1].buffer, p[1].rowStride, p[1].pixelStride,
            p[2].buffer, p[2].rowStride, p[2].pixelStride,
        )
        if (ok) {
            var size = p[0].rowStride.toLong() * (h and 1.inv()) * 3L / 2L
            if (cap > 0 && size > cap) size = cap.toLong()
            enc.queueInputBuffer(ix, 0, size.toInt(), ptsUs.coerceAtLeast(0), 0)
            return
        }
        // Not direct buffers, so nothing was written. This used to log and queue the frame
        // anyway, which encodes whatever the buffer happened to contain.
        onLog("encoder planes are not direct buffers -- packing I420 instead")
    }
    val i420 = NativePipe.bgrToI420(bgr, w, h)
    val buf = enc.getInputBuffer(ix)
        ?: error("encoder input $ix is neither an Image nor a buffer")
    if (i420.size > buf.capacity())
        error("encoder input buffer holds ${buf.capacity()} bytes, " +
              "and an I420 ${w}x$h needs ${i420.size}")
    buf.clear()
    buf.put(i420)
    enc.queueInputBuffer(ix, 0, i420.size, ptsUs.coerceAtLeast(0), 0)
}
