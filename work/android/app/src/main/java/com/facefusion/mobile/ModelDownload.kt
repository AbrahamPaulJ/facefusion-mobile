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
    private const val BASE = "https://huggingface.co/$REPO/resolve/main/"

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
        val root = JSONObject(URL(BASE + "manifest.json").readText())
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
     * Stages that are OFF until the user turns them on, so a first run must not pay for
     * them. Each has its own button in the Settings inventory.
     *
     * ⚠ wav2lip is here because ffpipe stopped opening it in 0.6.0 -- it is not merely
     * optional, it is unreachable, and it is still published for tiers built before then.
     */
    private val OPTIONAL_MODELS = listOf("gpen", "edtalk", "wav2lip")

    /** Which of [entries] are not already present and correct in [dir]. */
    fun missing(dir: File, entries: List<Entry>): List<Entry> =
        entries.filter { e ->
            val f = File(dir, e.name)
            !f.canRead() || f.length() != e.bytes
        }

    /**
     * Download everything missing. Blocking; call from a service worker thread.
     *
     * @return null on success, an error string otherwise.
     */
    fun run(dir: File, chain: String, onTick: () -> Unit): String? {
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
            val entries = published.filterNot { it.name in optional }

            val todo = missing(dir, entries)
            // Say WHAT is about to be fetched and what was kept. A download that quietly
            // re-fetches a 196 MB file it already has, and one that fetches only the file
            // you asked for, look identical from outside -- a progress bar and a wait.
            android.util.Log.i("ffmodels", "manifest " + published.size + " files, " +
                (published.size - entries.size) + " optional skipped, " + entries.size +
                " required, fetching " +
                todo.map { it.name } + ", keeping " +
                entries.filterNot { e -> todo.any { it.name == e.name } }.map { it.name })
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

            conn = (URL(BASE + e.remote).openConnection() as HttpURLConnection).apply {
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
        dest.delete()
        return if (part.renameTo(dest)) null else "${e.name}: could not be saved"
    }
}
