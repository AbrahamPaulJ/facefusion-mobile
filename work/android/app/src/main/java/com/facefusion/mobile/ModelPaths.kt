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
    fun tierChain(ctx: Context): List<String> {
        val lib = ctx.applicationInfo.nativeLibraryDir
        return NativePipe.probeTierChain(lib, lib)
            .split(",").map { it.trim() }.filter { it.isNotEmpty() }
    }

    /**
     * The tier whose files are actually on disk, else the best this chip could load.
     *
     * With nothing downloaded yet the answer is the best tier, because that is the set to
     * fetch.
     */
    fun tier(ctx: Context): String {
        val chain = tierChain(ctx)
        val d = dir(ctx)
        return chain.firstOrNull { File(d, "yoloface_" + it + ".bin").canRead() }
            ?: chain.firstOrNull()
            ?: ctx.applicationInfo.nativeLibraryDir.let { NativePipe.probeTier(it, it) }
    }

    /**
     * Which required files are absent for [tier], by display name.
     *
     * The gate counts as required because it BLOCKS: without it there is nothing to refuse
     * with, and a run that cannot check is a run that must not happen. Either build
     * satisfies it, fp32 (`nsfw_`) or the quantised `nsfwq_` the lower tiers carry.
     */
    fun missing(ctx: Context, tier: String, swapper: String): List<String> {
        val d = dir(ctx)
        val absent = listOf("yoloface", "fan2d", "arcface", swapper)
            .filter { !File(d, it + "_" + tier + ".bin").canRead() }
            .toMutableList()
        if (!File(d, "nsfw_" + tier + ".bin").canRead() &&
            !File(d, "nsfwq_" + tier + ".bin").canRead())
            absent += "nsfw"
        return absent
    }
}
