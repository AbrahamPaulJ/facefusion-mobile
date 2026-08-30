package com.facefusion.mobile

import android.content.Context
import java.io.File

/**
 * Where the models are and which tier this chip loads.
 *
 * Extracted from MainActivity because the API server needs the same answers and there must
 * be exactly one rule: if the server resolves a different tier from the screen, "models
 * missing" and "model loaded" can disagree on the same device and the report reads like a
 * download bug.
 *
 * ⚠ The rule has to match `Pipeline::init` on the native side, which resolves against disk
 * the same way. Naming a tier the pipeline will not open is the failure mode this shape
 * exists to prevent.
 */
object ModelPaths {

    /**
     * The models directory, created by the APP.
     *
     * A directory created by `adb push` is owned by `shell`, and this app cannot traverse
     * it -- open() fails with a bare ENOENT. mkdirs() here makes it app-owned first.
     */
    fun dir(ctx: Context): File = File(ctx.getExternalFilesDir(null), "models").apply { mkdirs() }

    /**
     * Every tier this chip can load, best first.
     *
     * NOT "this tier and every older one" -- see NativePipe.probeTierChain -- so it must
     * never be reconstructed from a single tier.
     */
    /**
     * Cached because the chain is a property of the SILICON and cannot change while the
     * process lives -- and because probing it brings the QNN backend up, which is far too
     * expensive to do from composition.
     *
     * ⚠ The chain is cached; the RESOLVED TIER below is deliberately not. That distinction
     * is the whole bug fixed on 2026-08-30: what a device *can* load is fixed, what is *on
     * disk* changes the moment a download finishes.
     */
    @Volatile private var chainCache: List<String>? = null
    @Volatile private var backendCache: String? = null

    // ---------------------------------------------------------------- forced runtime

    private const val PREFS = "ff_backend"
    private const val KEY_FORCED = "forced"

    /**
     * Which runtime the user has pinned: "qnn", "ncnn", or "" for automatic.
     *
     * Persisted, because the point of it is to survive the app restart that a runtime
     * change wants -- a switch that reset itself on launch could never be used to run a
     * whole video through the other backend.
     */
    fun forcedBackend(ctx: Context): String =
        ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY_FORCED, "") ?: ""

    /**
     * Pin the runtime, or "" to go back to automatic.
     *
     * ⚠ ORDER MATTERS and the caller must have released the pipeline first. This drops both
     * caches because they hold answers from the runtime being replaced: the cached backend
     * IS the question being changed, and the tier chain is "v79,v73,v68" on one runtime and
     * "ncnn" on the other. Leaving either in place makes the app download one runtime's
     * models and load the other's.
     */
    fun setForcedBackend(ctx: Context, value: String) {
        ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putString(KEY_FORCED, value).apply()
        apply(ctx)
    }

    /**
     * Push the persisted choice into the native seam and invalidate what it invalidates.
     *
     * Called at startup BEFORE any probe -- `backend()` and `tierChain()` cache their first
     * answer, so a setting applied after them would be ignored for the life of the process.
     */
    fun apply(ctx: Context) {
        backendCache = null
        chainCache = null
        NativePipe.ensureLoaded()
        NativePipe.setForcedBackend(forcedBackend(ctx))
    }

    /**
     * "qnn" or "ncnn" -- which runtime, and therefore which MODEL SET this device needs.
     *
     * A silicon property like the tier CHAIN, so it is cached; unlike the tier it does not
     * depend on what is on disk. "none" is not cached: a failed probe must not be
     * remembered as an answer.
     */
    fun backend(ctx: Context): String {
        backendCache?.let { return it }
        val lib = ctx.applicationInfo.nativeLibraryDir
        val b = NativePipe.probeBackend(lib, lib)
        if (b == "qnn" || b == "ncnn") backendCache = b
        return b
    }

    /**
     * The files one variant needs, by logical name -> filename.
     *
     * ⚠ This MIRRORS what ffnn_qnn.cpp and ffnn_ncnn.cpp resolve at runtime, and the two
     * must not drift. It exists because the app has to know what to DOWNLOAD before any
     * backend has opened anything -- native answers "where is yoloface", this answers
     * "which files must exist first", and only the second question can be asked offline.
     *
     * QNN is one context binary per model, tier in the name. ncnn is a param/bin PAIR per
     * model, no tier, named after the ONNX graph rather than the role.
     */
    fun filesFor(tier: String, name: String): List<String> =
        if (tier == NCNN_TIER) {
            val stem = NCNN_STEMS[name] ?: return emptyList()
            listOf("$stem.ncnn.param", "$stem.ncnn.bin")
        } else {
            listOf(name + "_" + tier + ".bin")
        }

    const val NCNN_TIER = "ncnn"

    private val NCNN_STEMS = mapOf(
        "yoloface" to "yoloface_8n_b1",
        "fan2d" to "2dfan4_heatmaps",
        "arcface" to "arcface_w600k_r50_b1",
        "hyperswap" to "hyperswap_1a_256_fp32",
        "gpen" to "gpen_ncnn",
        // One graph serves both gate names: "nsfwq2" exists because a QNN tier below v79
        // cannot finalize the fp32 gate, which is a QNN fact and means nothing to ncnn.
        "nsfw" to "nsfw_2_sim",
        "nsfwq2" to "nsfw_2_sim",
    )

    fun tierChain(ctx: Context): List<String> {
        chainCache?.let { return it }
        val lib = ctx.applicationInfo.nativeLibraryDir
        val c = NativePipe.probeTierChain(lib, lib)
            .split(",").map { it.trim() }.filter { it.isNotEmpty() }
        // Only a non-empty answer is cached: a failed probe must not be remembered as
        // "this chip can load nothing" for the rest of the process.
        if (c.isNotEmpty()) chainCache = c
        return c
    }

    /**
     * The tier whose files are actually on disk, else the best this chip could load.
     *
     * With nothing downloaded yet the answer is the best tier, because that is the set to
     * fetch.
     *
     * ⚠ **Never cache this, and never substitute `pickTier`/`probeDeviceInfo`'s tier for
     * it.** That tier is `tierChain().front()` -- what the SILICON can load, with no
     * reference to disk. They differ on exactly the devices whose best arch is not hosted
     * yet: a v81 part resolves the chain `v81,v73,v68`, the manifest publishes no v81, the
     * downloader correctly fetches **v73**, and anything naming files `_v81` then reports
     * a complete download as missing. That shipped in 0.2.0 and is what this warning is
     * for. Recomputing is a handful of `canRead()` calls; the chain above is what was
     * expensive, and it is cached.
     */
    fun tier(ctx: Context): String {
        // ncnn has no tiers: one set of files runs on every part, so the answer is fixed
        // and there is nothing on disk to resolve against.
        if (backend(ctx) == NCNN_TIER) return NCNN_TIER
        val chain = tierChain(ctx)
        val d = dir(ctx)
        return chain.firstOrNull { present(d, it, "yoloface") }
            ?: chain.firstOrNull()
            ?: ctx.applicationInfo.nativeLibraryDir.let { NativePipe.probeTier(it, it) }
    }

    /** Every file of [name] for [tier] is on disk. A pair is present only when BOTH are. */
    fun present(d: File, tier: String, name: String): Boolean {
        val f = filesFor(tier, name)
        return f.isNotEmpty() && f.all { File(d, it).canRead() }
    }

    /**
     * Which required files are absent for [tier], by display name.
     *
     * The gate counts as required because it BLOCKS: without it there is nothing to refuse
     * with, and a run that cannot check is a run that must not happen. Either build
     * satisfies it, fp32 (`nsfw_`) or the quantised `nsfwq2_` the lower tiers carry.
     * (`nsfwq2_`, not `nsfwq_`: the old name is calibrated for an input range this
     * app no longer produces -- see work/qnn/convert.sh.)
     */
    fun missing(ctx: Context, tier: String, swapper: String): List<String> {
        val d = dir(ctx)
        val absent = listOf("yoloface", "fan2d", "arcface", swapper)
            .filter { !present(d, tier, it) }
            .toMutableList()
        if (!present(d, tier, "nsfw") && !present(d, tier, "nsfwq2")) absent += "nsfw"
        return absent
    }
}
