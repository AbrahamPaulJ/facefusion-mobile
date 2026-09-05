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

    /**
     * Frames between real face detections during a VIDEO run. 0 = detect every frame,
     * which is upstream's behaviour and the default.
     *
     * The only option here that trades OUTPUT for speed. Everything else in this class
     * either changes the result on purpose (weight, blur) or is free; this one reconstructs
     * the detector's box from the previous frame's landmarks and skips yoloface plus its
     * whole-frame letterbox -- measured at 7.07 ms/frame saved at period 4, against a
     * deviation from every-frame detection of 45.5 dB over the swapped region at natural
     * motion and 42.2 dB at 6x motion. For scale, this port's native-vs-host-reference
     * error is 42.0 dB over the same region, so at ordinary motion it costs less than the
     * quantisation already does.
     *
     * ⚠ Applies to video runs ONLY. The preview shares `Pipeline::analyse`, where
     * consecutive calls are unrelated frames the user seeked to.
     */
    val trackPeriod: Int = 0,

    /**
     * `--face-enhancer`, gpen_bfr_256. Off by default.
     *
     * Costs 8.57 GMAC per face on top of the swapper's 31.93, and needs its own context
     * binary -- so asking for it does not mean getting it. The pipeline skips the stage
     * when the model is absent, and the UI only offers the switch when
     * [NativePipe.hasEnhancer] is true.
     */
    val faceEnhance: Boolean = false,

    /**
     * `--face-enhancer-blend`, upstream's 0-100 as a 0.0-1.0 fraction, step 0.05.
     *
     * 1.0 is the enhancer's output alone; 0.0 leaves the swap untouched. Upstream's
     * default is 80.
     */
    val enhanceBlend: Float = 0.8f,

    /**
     * `--processors lip_syncer`, edtalk_256. Off by default.
     *
     * Costs 1.95 ms per face on the NPU, which is small; the audio front end costs 28.4 ms
     * per SECOND of audio once per clip, which is where the time actually goes. Needs its
     * own context binary, so asking for it does not mean getting it -- VideoSwapper checks
     * [NativePipe.hasLipSyncer] and the clip's audio track, and falls back to a plain swap
     * with a log line rather than failing.
     *
     * Video only. A photo has no audio to sync to.
     */
    val lipSync: Boolean = false,

    /**
     * `--lip-syncer-weight`, 0.0-1.0 step 0.05, upstream default 0.5. ONE knob shared by
     * edtalk, per `lip_syncer/core.py`: the third model input, `weight` -- a lip-direction
     * scale the generator reads directly. This app drove it at a hardcoded 1.0 until 0.4.34.
     *
     * (wav2lip, removed in 0.6.0, applied the same knob completely differently: it scaled
     * the ALREADY-COMPUTED mel window by `weight * 2.0` in `prepare_audio_frame`, not the
     * raw audio and not the target crop. One name, two meanings, which is part of why
     * carrying both graphs cost more than it saved.)
     */
    val lipSyncWeight: Float = 0.5f,

    /**
     * Output frame rate. **0 means "same as the input"**, which is the default and the only
     * value that cannot be wrong -- every other choice is a resample.
     *
     * The UI never offers a rate above the input's: raising it would duplicate frames and
     * cost NPU time producing nothing new, so the cap is structural rather than validated
     * after the fact.
     */
    val outputFps: Int = 0,
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
            .putInt(K_TRACK, trackPeriod)
            .putBoolean(K_ENHANCE, faceEnhance)
            .putFloat(K_ENHANCE_BLEND, enhanceBlend)
            .putBoolean(K_LIP_SYNC, lipSync)
            .putFloat(K_LIP_SYNC_WEIGHT, lipSyncWeight)
            .putInt(K_FPS, outputFps)
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
        private const val K_TRACK = "track_period"
        private const val K_ENHANCE = "face_enhance"
        private const val K_ENHANCE_BLEND = "face_enhance_blend"
        private const val K_LIP_SYNC = "lip_sync"
        private const val K_LIP_SYNC_WEIGHT = "lip_sync_weight"
        private const val K_FPS = "output_fps"

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
                trackPeriod = p.getInt(K_TRACK, d.trackPeriod),
                faceEnhance = p.getBoolean(K_ENHANCE, d.faceEnhance),
                enhanceBlend = p.getFloat(K_ENHANCE_BLEND, d.enhanceBlend),
                lipSync = p.getBoolean(K_LIP_SYNC, d.lipSync),
                lipSyncWeight = p.getFloat(K_LIP_SYNC_WEIGHT, d.lipSyncWeight),
                outputFps = p.getInt(K_FPS, d.outputFps),
            )
        }
    }
}
