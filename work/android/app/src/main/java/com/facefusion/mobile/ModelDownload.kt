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
     * Versioned, following upstream's own scheme: a new repo per model revision means old
     * builds keep resolving the files they were tested against.
     */
    const val REPO = "AbrahamPJ/facefusion-mobile-models-0.1.0"
    private const val BASE = "https://huggingface.co/$REPO/resolve/main/"

    data class Entry(val name: String, val bytes: Long, val sha256: String)

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
        val text = URL(BASE + "manifest.json").readText()
        val tiers = JSONObject(text).getJSONObject("tiers")
        val tier = want.firstOrNull { tiers.has(it) }
            ?: error("no models published for any of " + want.joinToString(", "))
        val arr = tiers.getJSONObject(tier).getJSONArray("files")
        return tier to (0 until arr.length()).map {
            val o = arr.getJSONObject(it)
            Entry(o.getString("name"), o.getLong("bytes"), o.getString("sha256"))
        }
    }

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
            val entries = try {
                manifestFor(chain).second
            } catch (e: Exception) {
                return fail("Could not read the model list: ${e.message}")
            }

            val todo = missing(dir, entries)
            // Say WHAT is about to be fetched and what was kept. A download that quietly
            // re-fetches a 196 MB file it already has, and one that fetches only the file
            // you asked for, look identical from outside -- a progress bar and a wait.
            android.util.Log.i("ffmodels", "manifest " + entries.size + " files, fetching " +
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

            conn = (URL(BASE + e.name).openConnection() as HttpURLConnection).apply {
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
