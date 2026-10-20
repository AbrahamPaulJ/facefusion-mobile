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
    /** `hyperswap` 1a (256, ships), `hyperswap_1b` / `hyperswap_1c` (256, same I/O, separate weights). */
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
     * `--reference-face-distance`, 0.0-1.0, upstream default 0.3.
     *
     * Only meaningful once a reference face has been picked by tapping one
     * (`NativePipe.setReferenceFaceAt`), which is why there is no mode field beside it:
     * the pipeline holds the reference, and holding one IS the mode. Upstream's own
     * comparison, from face_selector.py:
     *
     *     d = (1 - dot(embedding_norm, reference.embedding_norm)) / 2  ->  match if d < this
     *
     * so 0.3 accepts anything above 0.4 cosine similarity. Raise it and other people start
     * being swapped too; lower it and a face that turns away stops matching itself.
     */
    val referenceDistance: Float = 0.3f,

    /**
     * Copy every finished BATCH clip straight into the gallery.
     *
     * ⚠ Not a pipeline field -- it never reaches native, like `outputFps`. It lives here
     * only because it is a preference worth remembering: a batch is unattended by nature,
     * and being asked to re-tick it on every run defeats the point of leaving one running.
     */
    val batchAutoSave: Boolean = false,

    /**
     * Cap the output's SHORT EDGE, in pixels. 0 keeps the source's own size.
     *
     * Upstream 3.8.2 has no resolution argument at all: it takes `--output-video-scale`, a
     * 0.25-8.0 multiplier defaulting to 1.0. The upward half of that range is a trap here --
     * the swapper runs at 256 whatever the frame is, so 2x output is the same swap enlarged
     * at four times the bitrate. This is the useful half, named the way people say it.
     *
     * ⚠ THE SHORT EDGE, so the aspect ratio never changes and "480p" means what it means
     * everywhere else: a 16:9 clip becomes 854x480 and a portrait one 480x854. And it only
     * ever shrinks -- picking 1080p on a 720p clip leaves it alone, which is what upstream's
     * own `restrict_video_resolution` does.
     *
     * Applied at DECODE, not at encode. Detector prep and paste-back both scale with frame
     * AREA, so capping a 4K clip to 1080p makes the run itself faster rather than merely
     * making the file smaller.
     *
     * ⚠ It is a real trade, not a free win: swap quality follows how many pixels THE FACE
     * has, not the frame. A face filling a quarter of the frame has plenty either way; a
     * face far away in a 4K shot has half as many pixels at 1080p and will look worse.
     * Hence 0 by default.
     */
    val outputMaxShortEdge: Int = 0,

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
     * `--processors face_editor`, LivePortrait. Off by default.
     *
     * Three more graphs per face (~160 ms on an 8 Elite) and its own download, so asking for
     * it does not mean getting it: the pipeline skips the stage when the set is absent, and
     * the UI offers the switch only when it is installed. Snapdragon only -- there is no
     * ncnn build of it.
     */
    val faceEdit: Boolean = false,

    /**
     * face_swapper in `--processors`. Upstream runs any subset of processors, so the editor,
     * the enhancer or the lip syncer can run on a target with no swap at all -- and then no
     * source face is needed. Read [swapsFace], never this: OFF only counts while another
     * processor is ON, so the app can never be asked to run nothing.
     */
    val swapEnabled: Boolean = true,

    /**
     * face_animator: a PHOTO target's face moved like a driver video's -- LivePortrait's own
     * use, which upstream does not offer. Never with the swapper (swapping the frames of an
     * animated photo is just animating a different photo), so while this is on [swapsFace]
     * is false. Photo targets only; loading a video target clears it.
     */
    val faceAnimate: Boolean = false,

    /**
     * face_editor's sliders, each -1..1, in [EDITOR_SLIDERS] order. All zero is the
     * identity, and the stage is then skipped rather than run to no effect.
     *
     * A List, not a FloatArray: this class is compared by VALUE to decide whether the
     * preview needs a refresh, and an array compares by identity.
     */
    val editor: List<Float> = List(EDITOR_SLIDERS.size) { 0f },

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

    /** Whether this run swaps faces, and therefore needs a source. See [swapEnabled]. */
    val swapsFace: Boolean get() =
        !faceAnimate && (swapEnabled || !(faceEdit || faceEnhance || lipSync))

    /** How many swapper invocations one face costs at this setting. */
    val invocationsPerFace get() = pixelBoost * pixelBoost

    /**
     * ⚠ THE TARGET-DEPENDENT SETTINGS ARE NOT SAVED, deliberately.
     *
     * `outputFps` and `outputMaxShortEdge` are answers about ONE clip: "24 of this clip's
     * 30" and "720p of this clip's 2160p". Restoring them onto the next clip applies a
     * decision that was never made about it -- a 720p cap silently inherited by a 720p clip
     * does nothing visible, and by a 4K one quietly halves what the user gets. Everything
     * else here is a preference about how the app should WORK and does survive.
     *
     * They still persist for the length of a session; `loadTarget` resets them when the
     * clip they were chosen for goes away.
     */
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
            .putFloat(K_REF_DISTANCE, referenceDistance)
            .putBoolean(K_BATCH_AUTOSAVE, batchAutoSave)
            .putInt(K_TRACK, trackPeriod)
            .putBoolean(K_ENHANCE, faceEnhance)
            .putFloat(K_ENHANCE_BLEND, enhanceBlend)
            .putBoolean(K_LIP_SYNC, lipSync)
            .putFloat(K_LIP_SYNC_WEIGHT, lipSyncWeight)
            .putBoolean(K_FACE_EDIT, faceEdit)
            .putBoolean(K_SWAP, swapEnabled)
            .putBoolean(K_ANIMATE, faceAnimate)
            .putString(K_EDITOR, editor.joinToString(","))
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
        private const val K_REF_DISTANCE = "reference_distance"
        private const val K_BATCH_AUTOSAVE = "batch_autosave"
        private const val K_TRACK = "track_period"
        private const val K_ENHANCE = "face_enhance"
        private const val K_ENHANCE_BLEND = "face_enhance_blend"
        private const val K_LIP_SYNC = "lip_sync"
        private const val K_LIP_SYNC_WEIGHT = "lip_sync_weight"
        private const val K_FACE_EDIT = "face_edit"
        private const val K_SWAP = "face_swap"
        private const val K_ANIMATE = "face_animate"
        private const val K_EDITOR = "face_editor"

        /**
         * face_editor's sliders, in the order the native side reads them
         * (ffportrait::Slider) -- upstream's own option names, minus the prefix.
         */
        val EDITOR_SLIDERS = listOf(
            "eyebrow_direction", "eye_gaze_horizontal", "eye_gaze_vertical", "eye_open_ratio",
            "lip_open_ratio", "mouth_grim", "mouth_pout", "mouth_purse", "mouth_smile",
            "mouth_position_horizontal", "mouth_position_vertical", "head_pitch", "head_yaw",
            "head_roll",
        )

        /**
         * "mouth_smile=1,head_yaw=-0.5" -> the 14 sliders, clamped; unnamed ones 0, unknown
         * names and unparsable values ignored. The one parser for every scripted way in --
         * the adb self-test and the API's `edit` parameter -- so they cannot disagree on
         * what a spec means.
         */
        fun parseEditor(spec: String): List<Float> {
            val v = MutableList(EDITOR_SLIDERS.size) { 0f }
            for (kv in spec.split(",")) {
                val i = EDITOR_SLIDERS.indexOf(kv.substringBefore("=").trim())
                val x = kv.substringAfter("=", "").trim().toFloatOrNull()
                if (i >= 0 && x != null && x.isFinite()) v[i] = x.coerceIn(-1f, 1f)
            }
            return v
        }

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
            val editor = runCatching {
                p.getString(K_EDITOR, null)?.split(",")
                    ?.map { it.trim().toFloat().coerceIn(-1f, 1f) }
                    ?.takeIf { it.size == EDITOR_SLIDERS.size }
            }.getOrNull() ?: d.editor
            return SwapOptions(
                swapper = (p.getString(K_SWAPPER, d.swapper) ?: d.swapper)
                    .let { if (it == "inswapper") "hyperswap" else it },
                weight = p.getFloat(K_WEIGHT, d.weight),
                maskBlur = p.getFloat(K_BLUR, d.maskBlur),
                maskPadding = padding,
                detectorScore = p.getFloat(K_DET, d.detectorScore),
                landmarkerScore = p.getFloat(K_LMK, d.landmarkerScore),
                pixelBoost = p.getInt(K_BOOST, d.pixelBoost),
                largestOnly = p.getBoolean(K_LARGEST, d.largestOnly),
                referenceDistance = p.getFloat(K_REF_DISTANCE, d.referenceDistance),
                batchAutoSave = p.getBoolean(K_BATCH_AUTOSAVE, d.batchAutoSave),
                trackPeriod = p.getInt(K_TRACK, d.trackPeriod),
                faceEnhance = p.getBoolean(K_ENHANCE, d.faceEnhance),
                enhanceBlend = p.getFloat(K_ENHANCE_BLEND, d.enhanceBlend),
                lipSync = p.getBoolean(K_LIP_SYNC, d.lipSync),
                lipSyncWeight = p.getFloat(K_LIP_SYNC_WEIGHT, d.lipSyncWeight),
                faceEdit = p.getBoolean(K_FACE_EDIT, d.faceEdit),
                swapEnabled = p.getBoolean(K_SWAP, d.swapEnabled),
                faceAnimate = p.getBoolean(K_ANIMATE, d.faceAnimate),
                editor = editor,
            )
        }
    }
}
