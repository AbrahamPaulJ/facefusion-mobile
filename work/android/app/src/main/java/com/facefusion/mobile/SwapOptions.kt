package com.facefusion.mobile

import android.content.Context

/**
 * The runtime knobs, with FaceFusion's own names, defaults and ranges.
 *
 * Only options that cost **nothing at the graph level** are here. A QNN context binary has
 * its input shapes baked in at conversion, so anything that changes a tensor shape --
 * detector size, the swapper's 256x256 input, batching -- is not a setting, it is a
 * rebuild. Everything below is either a scalar fed to a graph, a threshold applied to its
 * output, or CPU geometry around it.
 *
 * [pixelBoost] is the exception worth understanding: it raises the *output* resolution
 * without touching the graph, by warping the face larger and running the same 256x256
 * context over polyphase sub-images. It costs `pixelBoost^2` invocations per face.
 */
data class SwapOptions(
    /** `hyperswap` (256, ships) or `inswapper` (128, converted, needs its binary pushed). */
    val swapper: String = "hyperswap",

    /**
     * `--face-swapper-weight`, 0.0-1.0 step 0.05.
     *
     * Blends the source and target IDENTITY embeddings, not the images. 0.5 is the source
     * unmodified; above it the source is amplified and the target identity subtracted,
     * which is the direction to push for a stronger resemblance.
     */
    val weight: Float = 0.5f,

    /** `--face-mask-blur`, 0.0-1.0 step 0.05. How soft the swapped region's edge is. */
    val maskBlur: Float = 0.3f,

    /** `--face-mask-padding`, top/right/bottom/left, 0-100 percent. Shrinks the region. */
    val maskPadding: List<Int> = listOf(0, 0, 0, 0),

    /** `--face-detector-score`, 0.0-1.0 step 0.05. Raise it to ignore uncertain faces. */
    val detectorScore: Float = 0.5f,

    /** `--face-landmarker-score`, 0.0-1.0 step 0.05. Below it, the 5-point set is kept. */
    val landmarkerScore: Float = 0.5f,

    /** `--face-swapper-pixel-boost` as a per-axis factor: 1=256, 2=512, 3=768, 4=1024. */
    val pixelBoost: Int = 1,

    /** `--face-selector-mode`: false = `many`, true = `one` (largest face only). */
    val largestOnly: Boolean = false,
) {
    val pixelBoostLabel get() = "${256 * pixelBoost}x${256 * pixelBoost}"

    /** How many swapper invocations one face costs at this setting. */
    val invocationsPerFace get() = pixelBoost * pixelBoost

    fun save(context: Context) {
        prefs(context).edit()
            .putString(K_SWAPPER, swapper)
            .putFloat(K_WEIGHT, weight)
            .putFloat(K_BLUR, maskBlur)
            .putString(K_PADDING, maskPadding.joinToString(","))
            .putFloat(K_DET, detectorScore)
            .putFloat(K_LMK, landmarkerScore)
            .putInt(K_BOOST, pixelBoost)
            .putBoolean(K_LARGEST, largestOnly)
            .apply()
    }

    companion object {
        private const val FILE = "swap_options"
        private const val K_SWAPPER = "swapper"
        private const val K_WEIGHT = "weight"
        private const val K_BLUR = "mask_blur"
        private const val K_PADDING = "mask_padding"
        private const val K_DET = "detector_score"
        private const val K_LMK = "landmarker_score"
        private const val K_BOOST = "pixel_boost"
        private const val K_LARGEST = "largest_only"

        private fun prefs(context: Context) =
            context.getSharedPreferences(FILE, Context.MODE_PRIVATE)

        fun load(context: Context): SwapOptions {
            val p = prefs(context)
            val d = SwapOptions()
            // Each field falls back to its own default, so a partially written store or a
            // field added in a later version still loads rather than resetting everything.
            val padding = runCatching {
                p.getString(K_PADDING, null)?.split(",")?.map { it.trim().toInt() }
                    ?.takeIf { it.size == 4 }
            }.getOrNull() ?: d.maskPadding
            return SwapOptions(
                swapper = p.getString(K_SWAPPER, d.swapper) ?: d.swapper,
                weight = p.getFloat(K_WEIGHT, d.weight),
                maskBlur = p.getFloat(K_BLUR, d.maskBlur),
                maskPadding = padding,
                detectorScore = p.getFloat(K_DET, d.detectorScore),
                landmarkerScore = p.getFloat(K_LMK, d.landmarkerScore),
                pixelBoost = p.getInt(K_BOOST, d.pixelBoost),
                largestOnly = p.getBoolean(K_LARGEST, d.largestOnly),
            )
        }
    }
}
