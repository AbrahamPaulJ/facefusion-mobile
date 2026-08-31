package com.facefusion.mobile

/**
 * JNI surface onto libffnative.so.
 *
 * The native side is the same ffqnn + ffcv + ffpipe code the headless CLI links
 * (work/native/ffswap_main.cpp), so a swap verified over adb is the same computation the
 * app performs.  Colour conversion is native because it is per-pixel work on every frame.
 */
object NativePipe {

    @Volatile private var loaded = false

    fun ensureLoaded() {
        if (!loaded) { System.loadLibrary("ffnative"); loaded = true }
    }

    /**
     * libDir holds libQnnHtp.so/libQnnSystem.so; skelDir the hexagon skel.
     *
     * Prefer [init], which takes a [SwapOptions] instead of eleven positional arguments.
     * Every numeric argument is re-clamped natively -- they come from sliders, and a bad
     * `pixelBoost` would cost its square in graph invocations per face.
     */
    @JvmStatic external fun initEx(
        libDir: String, skelDir: String, modelDir: String, swapper: String,
        weight: Float, maskBlur: Float, maskPadding: IntArray,
        detectorScore: Float, landmarkerScore: Float,
        pixelBoost: Int, largestOnly: Boolean,
        faceEnhance: Boolean, enhanceBlend: Float,
    ): Boolean

    /** Load the pipeline with [opts] applied. */
    fun init(libDir: String, skelDir: String, modelDir: String,
             opts: SwapOptions = SwapOptions()): Boolean =
        initEx(libDir, skelDir, modelDir, opts.swapper,
               opts.weight, opts.maskBlur, opts.maskPadding.toIntArray(),
               opts.detectorScore, opts.landmarkerScore,
               opts.pixelBoost, opts.largestOnly,
               opts.faceEnhance, opts.enhanceBlend)

    /**
     * Tiers to skip at the next [init], comma-separated; "" clears.
     *
     * ⚠ Pushed rather than passed for the same reason the content gate enumerates its
     * paths: there are four callers of [init] and a per-call argument is a list something
     * can be left out of. Set it once, from [ModelPaths.apply], and no future path can
     * forget it.
     */
    @JvmStatic external fun setSkipTiers(tiers: String)

    /**
     * The tier that LOADED and then would not execute, or "".
     *
     * Meaningful after any [init], failed OR successful: a device can reject its best tier
     * and run on the next one, and that rejection is worth recording either way -- it is a
     * property of the silicon, not of this run, and re-proving it costs a full context load
     * on every launch.
     *
     * ⚠ Not the same as "init failed". A tier whose files are missing is a DOWNLOAD
     * problem and never appears here; only a tier the chip refused to run does.
     */
    @JvmStatic external fun rejectedTier(): String
    @JvmStatic external fun release()

    /**
     * Whether `gpen_<tier>.bin` was found when the pipeline last initialised.
     *
     * Only meaningful after a successful [init]; false before that, which is the safe
     * direction -- the UI hides the enhancer switch rather than offering one that cannot
     * work. Same rule `inswapper` follows.
     */
    @JvmStatic external fun hasEnhancer(): Boolean

    /**
     * Which context-binary tier this chip needs -- "v68" / "v73" / "v79".
     *
     * Measured off the HTP (arch, VTCM, soc_model) with no model loaded, so it can be
     * asked before the binaries are even present.  Returns the most permissive tier when
     * the probe fails, so an unrecognised device behaves like an old one rather than not
     * at all.
     */
    @JvmStatic external fun probeTier(libDir: String, skelDir: String): String

    /**
     * Every tier this chip can load, best first, comma-joined -- "v81,v73,v68".
     *
     * [probeTier] names the tier the hardware DESERVES; this names the ones it can
     * actually use. They differ exactly when the app has learned about an arch whose
     * context binaries are not hosted yet, which is the normal state of affairs for a
     * day or two after a new tier lands. The downloader walks this and takes the first
     * tier the manifest carries; ffpipe walks it and takes the first present on disk.
     *
     * ⚠ Not simply "every older tier": the v79 build is pinned to soc_model 69, so it is
     * absent from a v81 chain. Do not reconstruct this list in Kotlin.
     */
    @JvmStatic external fun probeTierChain(libDir: String, skelDir: String): String

    /**
     * Whether this chip accepts the fp16 stamp every QAIRT 2.49 context carries.
     *
     * "yes" | "no" | "unknown".  ⚠ "unknown" means the CONTROL canary failed -- the probe
     * is broken and the chip has said nothing.  It must never be treated as "no": that
     * verdict pushes a working device onto the slower compatibility build.
     *
     * @param canaryDir holds canary_249.bin and canary_228.bin, unpacked from assets.
     */
    @JvmStatic external fun probeFp16(libDir: String, skelDir: String,
                                      canaryDir: String): String

    /**
     * What the HTP reports about itself, as `key=value;` pairs:
     * `ok`, `arch`, `vtcm`, `soc`, `signedPd`, `dlbc`, `tier`.
     *
     * ⚠ `ok=0` means the PROBE failed and every other field is absent. It does not mean the
     * chip is unsupported -- the same distinction [probeTier] makes when it falls back.
     */
    /**
     * Which runtime this device will use: "qnn", "ncnn", or "none" when neither starts.
     *
     * Decides which MODEL SET to download, so it is asked before any file exists. It is
     * answered by TRYING rather than probing -- the HTP cannot be interrogated until QNN is
     * running, so an "ask first" version reports no-NPU on every device.
     */
    @JvmStatic external fun probeBackend(libDir: String, skelDir: String): String

    /**
     * Pin the runtime for the rest of this process: "qnn", "ncnn", or "" for automatic.
     *
     * Auto tries QNN first and QNN wins on any Qualcomm part, so without this the
     * non-Qualcomm path could only be exercised on a phone with no Hexagon -- which is not
     * the bench, and an untestable path is an unverified one. It is the same `FFBACKEND`
     * the headless CLI reads; the app sets it on itself because an Android process has no
     * environment anyone outside can set.
     *
     * ⚠ Call [release] FIRST, and clear [ModelPaths]'s caches after: the cached backend and
     * tier chain are answers from the runtime this replaces.
     */
    @JvmStatic external fun setForcedBackend(name: String)

    /**
     * Whether the ncnn backend is compiled into THIS build.
     *
     * Not "is there a GPU", and not "which backend is running": whether the code is in the
     * binary at all. `FF_NCNN` is off unless `work/android/ncnn/` was staged, so a QNN-only
     * APK is a normal build -- and offering to switch to a runtime that is not linked is a
     * control that silently does nothing.
     */
    @JvmStatic external fun hasNcnnBackend(): Boolean

    @JvmStatic external fun probeDeviceInfo(libDir: String, skelDir: String): String

    /**
     * Upstream's content-gate statistic for one BGR frame: `logit[0] - logit[1]`, flagged
     * above [ContentGate.THRESHOLD].  Returns **NaN** when the graph did not run.
     *
     * ⚠ NaN, not `false`: an error that read as "allow" would open the gate exactly when
     * it broke.  Every comparison against a threshold is false for NaN, so callers must
     * test `isNaN()` explicitly -- see [ContentGate].
     */
    @JvmStatic external fun contentScore(bgr: ByteArray, w: Int, h: Int): Float

    /** True when this tier had no fp32 gate context; see [ContentGate.QUANTISED_BIAS]. */
    @JvmStatic external fun contentGateIsQuantised(): Boolean

    @JvmStatic external fun setSource(bgr: ByteArray, w: Int, h: Int): Boolean
    /** Swaps every face in place; returns the face count, or -1 on error. */
    @JvmStatic external fun processFrame(bgr: ByteArray, w: Int, h: Int): Int

    @JvmStatic external fun argbToBgr(argb: IntArray, w: Int, h: Int): ByteArray
    @JvmStatic external fun yuvToBgr(
        y: ByteArray, yRow: Int,
        u: ByteArray, uRow: Int, uPix: Int,
        v: ByteArray, vRow: Int, vPix: Int,
        w: Int, h: Int,
    ): ByteArray
    /**
     * Rotate a packed BGR frame clockwise by 0/90/180/270.
     *
     * For the container's rotation flag, which MediaCodec does NOT apply. 90 and 270 swap
     * the dimensions -- the caller sizes the encoder and everything downstream to match.
     */
    @JvmStatic external fun rotateBgr(bgr: ByteArray, w: Int, h: Int, degrees: Int): ByteArray

    @JvmStatic external fun bgrToI420(bgr: ByteArray, w: Int, h: Int): ByteArray
    /**
     * Write a BGR frame into the encoder's own input planes, honouring its strides.
     * COLOR_FormatYUV420Flexible is NOT necessarily I420 -- this device's AVC encoder is
     * semi-planar -- so the layout is taken from the Image rather than assumed.
     */
    @JvmStatic external fun bgrToImagePlanes(
        bgr: ByteArray, w: Int, h: Int,
        y: java.nio.ByteBuffer, yRow: Int, yPix: Int,
        u: java.nio.ByteBuffer, uRow: Int, uPix: Int,
        v: java.nio.ByteBuffer, vRow: Int, vPix: Int,
    ): Boolean
    /** Box-downsampled ARGB_8888 for the live preview; scaling natively avoids a
     *  full-resolution Bitmap allocation per frame. */
    @JvmStatic external fun bgrToArgb(bgr: ByteArray, w: Int, h: Int,
                                      dstW: Int, dstH: Int): IntArray

    @JvmStatic external fun lastError(): String
}
