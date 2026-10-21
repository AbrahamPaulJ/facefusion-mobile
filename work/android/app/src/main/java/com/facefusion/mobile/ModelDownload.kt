package com.facefusion.mobile

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Fetching the NPU context binaries for THIS device's tier.
 *
 * The models are not in the APK: they are ~275 MB per tier and several carry licences that
 * are not ours to sublicense, so they are hosted and pulled on demand -- the same thing
 * FaceFusion itself does.
 *
 * Two properties this has to have, both learned the hard way on this project:
 *
 *  * **Resumable.** A 196 MB file over a flaky link is not an atomic operation. Bytes land
 *    in `<name>.part` and a retry continues from its length with an HTTP Range request.
 *  * **Verified.** `adb push` twice reported success while leaving a truncated file of
 *    entirely plausible size. A context binary that is short does not fail loudly; it fails
 *    at load, four layers away from the cause. So every file is SHA256'd against the
 *    manifest before it is allowed to take its real name.
 */
object ModelDownload {

    /**
     * Unversioned since 0.4.0.  The name used to carry "-0.1.0" on the theory that a repo
     * per model revision would let old builds keep resolving the files they were tested
     * against -- but that is not how it was ever used: one repo carried v68/v73/v79/v81
     * and a replaced `gpen` across four releases, so the version in the name only ever
     * said something untrue.  `manifest.json` + SHA256 is what actually pins a build to
     * its files.
     *
     * The old name still resolves and MUST keep doing so -- 0.1.1 through 0.3.0 have it
     * compiled in.  HF redirects a renamed repo, and this downloader was already proven
     * against exactly that hop: `resolve/main/<file>` has always been a 307 to a different
     * host (`us.aws.cdn.hf.co`) and still returns 206 with the Range header intact.  After
     * the rename it is two hops instead of one, measured 200/206 on both names.
     * ⚠ Never create a new repo at the old name; the redirect is the compatibility story.
     */
    const val REPO = "AbrahamPJ/facefusion-mobile-models"

    /**
     * WHERE the files come from: Hugging Face itself, or hf-mirror.com (issue #18), an
     * unofficial mirror with the same `resolve/main/<file>` layout, for networks where
     * Hugging Face is blocked or slow. Same repo, same manifest, same checks. Measured
     * 2026-10-22 from outside China: the mirror answers with a 308 to huggingface.co, and
     * the Range header survives the hop (206).
     */
    const val SOURCE_HF = "hf"
    const val SOURCE_MIRROR = "mirror"
    private const val PREFS = "model_download"
    private const val KEY_SOURCE = "source"
    @Volatile private var source = SOURCE_HF
    private val base: String
        get() = (if (source == SOURCE_MIRROR) "https://hf-mirror.com/" else "https://huggingface.co/") +
                "$REPO/resolve/main/"

    /** Read the saved choice. Called by every entry point that downloads: the UI and the service. */
    fun loadSource(context: Context): String {
        source = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString(KEY_SOURCE, SOURCE_HF)?.takeIf { it == SOURCE_MIRROR } ?: SOURCE_HF
        return source
    }

    fun setSource(context: Context, value: String) {
        source = if (value == SOURCE_MIRROR) SOURCE_MIRROR else SOURCE_HF
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putString(KEY_SOURCE, source).apply()
    }

    /**
     * The content-check models, pinned to these exact files. The manifest's own hashes
     * only prove a file matches the manifest it came with, and with a mirror both come
     * from a third party; these were compiled in from what this project hosts, so no
     * host can change what the check runs. A new build of one of them needs an app
     * update, which a change to the check needs anyway.
     */
    private val PINNED = mapOf(
        "nsfwq_v68.bin" to "27884c9d1835ac45a596650dcdcabf4027475c931f4f9f22037ea5c761f0c629",
        "nsfwq2_v68.bin" to "37b58374e66ceb2b8c84621f9ae52d4d9d03a42848e0bcbe91fd10dc3729b3b1",
        "nsfwq_v73.bin" to "f85590597217088875a39fffd43ddb2b61594528656c1f976e380d56d7b1e930",
        "nsfwq2_v73.bin" to "197a25ad82ce6db89fa583cbe130686acbe7a45478e31d363412c9846102ac93",
        "nsfw_v79.bin" to "74755eb82955fd03abe7275e93a86e46352e36f9b8aa5a30f67e64da32b9f2d0",
        "nsfwq2_v81.bin" to "16ee9369d8fd62b8c7dce0b7d4f5cf2e57e6885effd6c25bc100aeac606e58e1",
        "nsfwq2_v81s97.bin" to "9d44b6132df404eedce2f9d0a825429e2e0abe4e09543194979e7a9677e4ae47",
        "nsfw_2_sim.ncnn.param" to "6200fb9e859cfa59dba49a104b5c2f30b4778e99e1fd8dc3989f8ea804fee7a1",
        "nsfw_2_sim.ncnn.bin" to "6fbff5ebe0919c3ed5518c2abebd08027bb4d3ffb2f37e805e5e4a3849ec67d1",
    )

    /**
     * One hosted file.
     *
     * [name] is the LOCAL filename and [remote] the repo-relative path it is fetched from.
     * They are the same for every QNN tier, whose files sit at the repo root -- but the
     * ncnn set lives in an `ncnn/` folder, and both the native loader
     * (`ffnn_ncnn.cpp` resolves `<modelDir>/<stem>.ncnn.param`) and [ModelPaths] expect a
     * FLAT models directory. Keeping the two apart is what lets the hosting layout change
     * without the device layout following it.
     */
    data class Entry(val name: String, val bytes: Long, val sha256: String,
                     val remote: String = name)

    /** What the UI draws. Compose observes this object directly. */
    var running by mutableStateOf(false); private set
    var fileIndex by mutableStateOf(0); private set
    var fileCount by mutableStateOf(0); private set
    var currentName by mutableStateOf(""); private set
    var doneBytes by mutableStateOf(0L); private set
    var totalBytes by mutableStateOf(0L); private set
    var error by mutableStateOf<String?>(null); private set
    var finished by mutableStateOf(false); private set

    /** 0..1 across the whole set, not the current file. */
    val progress: Float
        get() = if (totalBytes > 0) (doneBytes.toDouble() / totalBytes).toFloat() else 0f

    @Volatile private var cancelled = false

    fun cancel() { cancelled = true }

    fun reset() {
        running = false; fileIndex = 0; fileCount = 0; currentName = ""
        doneBytes = 0; totalBytes = 0; error = null; finished = false; cancelled = false
        queued = emptySet()
    }

    /** True when the connection is metered, so the caller can warn before spending ~275 MB. */
    fun isMetered(context: Context): Boolean {
        val cm = context.getSystemService(ConnectivityManager::class.java) ?: return false
        val caps = cm.getNetworkCapabilities(cm.activeNetwork) ?: return false
        return !caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED)
    }

    /**
     * Which files to fetch, from the hosted manifest. Network call; not on main.
     *
     * Takes the whole tier CHAIN ("v81,v73,v68", from [NativePipe.probeTierChain]) rather
     * than one tier, and returns the best one the manifest actually publishes along with
     * its files. The two differ whenever the app knows about an arch whose binaries are
     * not hosted yet -- which is the normal state for a day or two after a tier lands, and
     * used to be a hard "no models published for tier v81" for every user of that chip.
     *
     * A single tier is still accepted: a string with no comma is a one-entry chain.
     */
    fun manifestFor(chain: String): Pair<String, List<Entry>> {
        val want = chain.split(",").map { it.trim() }.filter { it.isNotEmpty() }
        if (want.isEmpty()) error("no tier requested")
        val root = JSONObject(URL(base + "manifest.json").readText())
        val tiers = root.optJSONObject("tiers") ?: JSONObject()
        val tier = want.firstOrNull { tiers.has(it) || root.has(legacyKey(it)) }
            ?: error("no models published for any of " + want.joinToString(", "))
        val arr = (tiers.optJSONObject(tier) ?: root.getJSONObject(legacyKey(tier)))
            .getJSONArray("files")
        return tier to (0 until arr.length()).map {
            val o = arr.getJSONObject(it)
            val remote = o.getString("name")
            // The LOCAL name is the basename. A hosted `ncnn/yoloface_8n_b1.ncnn.param`
            // has to land as `yoloface_8n_b1.ncnn.param` in a flat models dir, because that
            // is where ffnn_ncnn.cpp looks for it.
            Entry(remote.substringAfterLast('/'), o.getLong("bytes"), o.getString("sha256"),
                  remote)
        }
    }

    /**
     * Where a variant lives in the manifest when it is not under `tiers`.
     *
     * `ncnn` is published as a top-level `ncnn_preview` block, deliberately kept OUT of
     * `tiers` so that no app shipped before 0.4.0 could select it -- those builds have no
     * ncnn backend linked, and a tier they can download but not load is worse than one
     * they cannot see. 0.4.0 is the build that can, so it looks in both places, `tiers`
     * first: the day the block moves under `tiers` this keeps working and this function
     * stops being reached.
     */
    private fun legacyKey(variant: String): String =
        if (variant == ModelPaths.NCNN_TIER) "ncnn_preview" else variant

    /**
     * The LOCAL filenames this download is actually fetching, while it runs.
     *
     * ⚠ Exists because "is a download running" is not the same question as "is THIS model
     * downloading", and the Settings inventory was asking the first and displaying the
     * answer to the second. Every row said "Downloading" for the whole run -- harmless
     * while the bulk fetch really did take everything, and a plain lie the moment the
     * optional models stopped being in it.
     */
    var queued by mutableStateOf<Set<String>>(emptySet())
        private set

    /**
     * Stages that are OFF until the user turns them on, so a first run must not pay for
     * them. Each has its own button in the Settings inventory.
     *
     * ⚠ wav2lip is here because ffpipe stopped opening it in 0.6.0 -- it is not merely
     * optional, it is unreachable, and it is still published for tiers built before then.
     */
    // ⚠ hyperswap_1b and 1c are OPTIONAL and must stay that way. They are ~205 MB
    // EACH per tier, so a required pair would put 1.6 GB behind the one button a new
    // install taps to get started -- on a phone, on whatever connection it has. The list
    // is an EXCLUSION so that a model added to the manifest later is required by default;
    // these two are named here because they are alternatives to a swapper the app already
    // has, not additions to the set it needs.
    private val OPTIONAL_MODELS =
        listOf("gpen", "edtalk", "wav2lip", "hyperswap_1b", "hyperswap_1c",
               ModelPaths.LIVEPORTRAIT)

    /** Which of [entries] are not already present and correct in [dir]. */
    fun missing(dir: File, entries: List<Entry>): List<Entry> =
        entries.filter { e ->
            val f = File(dir, e.name)
            !f.canRead() || f.length() != e.bytes
        }

    /**
     * Download everything missing. Blocking; call from a service worker thread.
     *
     * @param only the LOCAL filenames to fetch, or null for "whatever this device
     *   requires". ⚠ The optional models can ONLY be reached through this parameter: the
     *   null path excludes them by name, so a caller that wants gpen or edtalk and passes
     *   null gets a run that fetches nothing and reports success. That was the 0.7.0
     *   regression -- the Settings row's Download button called the bulk fetch, which
     *   skipped the very model the button was attached to, and the app then said "Models
     *   ready" because as far as the required set was concerned it was.
     * @return null on success, an error string otherwise.
     */
    fun run(dir: File, chain: String, only: Set<String>? = null,
            onTick: () -> Unit): String? {
        cancelled = false
        error = null; finished = false; running = true
        try {
            val (tier, published) = try {
                manifestFor(chain)
            } catch (e: Exception) {
                return fail("Could not read the model list: ${e.message}")
            }

            // ⚠ REQUIRED ONLY. This used to hand the manifest's whole file list straight to
            // missing(), so "Download models" fetched the face enhancer and the lip syncer
            // as well -- about 85 MB per tier of stages that are OFF by default, that the
            // Settings inventory already offers one button each for, and that the README
            // describes as separate downloads. A first run paid for two features it had not
            // been asked about.
            //
            // Written as an EXCLUSION rather than a whitelist on purpose: a model added to
            // the manifest later is required until somebody says otherwise, so the failure
            // direction is a download that is too big, not an app missing a file it needs.
            //
            // filesFor() returns an empty list for a name a variant does not carry (edtalk
            // and wav2lip have no ncnn stems), so this is a no-op there rather than wrong.
            val optional = OPTIONAL_MODELS
                .flatMap { ModelPaths.filesFor(tier, it) }
                .toSet()
            val entries =
                if (only != null) published.filter { it.name in only }
                else published.filterNot { it.name in optional }

            // An explicit request that matches nothing is an ERROR, not an empty success.
            // The row offering the button was built from `hostedFiles`, so this can only
            // happen if the manifest moved under it -- and reporting "ready" for a file
            // that is not there is precisely the failure this whole change is about.
            if (only != null && entries.isEmpty())
                return fail("Not published for tier " + tier + ": " +
                            only.joinToString(", "))

            // Asked for by name, so fetch it even if a file of the right length is already
            // there: the row's other states (present, outdated) are what decide whether
            // the button says Download or Update, and Update means re-fetch.
            val todo = if (only != null) entries else missing(dir, entries)
            // Say WHAT is about to be fetched and what was kept. A download that quietly
            // re-fetches a 196 MB file it already has, and one that fetches only the file
            // you asked for, look identical from outside -- a progress bar and a wait.
            android.util.Log.i("ffmodels", "manifest " + published.size + " files, " +
                (published.size - entries.size) + " optional skipped, " + entries.size +
                " required, fetching " +
                todo.map { it.name } + ", keeping " +
                entries.filterNot { e -> todo.any { it.name == e.name } }.map { it.name })
            queued = todo.map { it.name }.toSet()
            fileCount = todo.size
            totalBytes = todo.sumOf { it.bytes }
            doneBytes = 0
            if (todo.isEmpty()) { finished = true; return null }

            for ((i, e) in todo.withIndex()) {
                if (cancelled) return fail("Cancelled")
                fileIndex = i + 1
                currentName = e.name
                val err = fetch(dir, e, onTick)
                if (err != null) return fail(err)
            }
            finished = true
            return null
        } finally {
            running = false
            // Cleared with running, not left behind: a stale queue would keep a row
            // labelled "Downloading" after the download it belonged to had ended.
            queued = emptySet()
            onTick()
        }
    }

    private fun fail(msg: String): String {
        error = msg
        return msg
    }

    /**
     * One file, resuming a `.part` if there is one.
     *
     * The temp file is only renamed to the real name after its hash matches, so a partial
     * or corrupt download can never present itself to the loader as a model.
     */
    private fun fetch(dir: File, e: Entry, onTick: () -> Unit): String? {
        val part = File(dir, e.name + ".part")
        val dest = File(dir, e.name)
        val already = doneBytes

        // A .part longer than the target is from a different revision of the file.
        if (part.exists() && part.length() > e.bytes) part.delete()

        var conn: HttpURLConnection? = null
        try {
            var from = if (part.exists()) part.length() else 0L
            if (from == e.bytes) {
                // Fully transferred last time but never verified.
                return verifyAndCommit(part, dest, e)
            }

            conn = (URL(base + e.remote).openConnection() as HttpURLConnection).apply {
                connectTimeout = 30_000
                readTimeout = 60_000
                if (from > 0) setRequestProperty("Range", "bytes=$from-")
            }
            val code = conn.responseCode
            if (from > 0 && code != HttpURLConnection.HTTP_PARTIAL) {
                // The server ignored the Range, so the bytes coming back start at zero.
                // Appending them to the .part would silently corrupt it.
                part.delete()
                from = 0
            }
            if (code != HttpURLConnection.HTTP_OK && code != HttpURLConnection.HTTP_PARTIAL)
                return "HTTP $code for ${e.name}"

            doneBytes = already + from
            conn.inputStream.use { input ->
                java.io.FileOutputStream(part, from > 0).use { out ->
                    val buf = ByteArray(1 shl 16)
                    var lastTick = 0L
                    while (true) {
                        if (cancelled) return "Cancelled"
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        doneBytes += n
                        // The UI cannot use more than a few updates a second, and a state
                        // write per 64 KB is pure recomposition churn.
                        val now = System.currentTimeMillis()
                        if (now - lastTick > 250) { lastTick = now; onTick() }
                    }
                }
            }
            if (cancelled) return "Cancelled"
            return verifyAndCommit(part, dest, e)
        } catch (t: Throwable) {
            // The .part is KEPT: it is exactly what a resume needs.
            return "${e.name}: ${t.message ?: t.toString()}"
        } finally {
            conn?.disconnect()
        }
    }

    private fun verifyAndCommit(part: File, dest: File, e: Entry): String? {
        if (part.length() != e.bytes) {
            part.delete()
            return "${e.name}: expected ${e.bytes} bytes, got ${part.length()}"
        }
        val md = MessageDigest.getInstance("SHA-256")
        part.inputStream().use { s ->
            val buf = ByteArray(1 shl 20)
            while (true) {
                val n = s.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        val got = md.digest().joinToString("") { "%02x".format(it) }
        if (!got.equals(e.sha256, ignoreCase = true)) {
            // Not resumable: a hash mismatch means the bytes we have are wrong, not short.
            part.delete()
            return "${e.name}: checksum mismatch"
        }
        if (e.name.startsWith("nsfw") && !got.equals(PINNED[e.name], ignoreCase = true)) {
            part.delete()
            return "${e.name}: not a file this app accepts"
        }
        dest.delete()
        return if (part.renameTo(dest)) null else "${e.name}: could not be saved"
    }
}
