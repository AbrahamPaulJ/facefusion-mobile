package com.facefusion.mobile

import android.content.Context
import android.content.Intent
import android.os.Build
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * A report the user can send when something goes wrong.
 *
 * An app cannot read logcat on modern Android without a system permission, so a crash
 * leaves nothing behind unless it is caught on the way out. [install] hooks the uncaught
 * handler and persists the stack trace; the next launch finds it and offers to send it.
 *
 * Everything here is assembled on demand and shared through ACTION_SEND, so nothing is
 * collected, stored or transmitted unless the user taps the button. There is no telemetry
 * in this app and this is not it.
 */
object BugReport {

    private const val CRASH = "last_crash.txt"

    /** Persist uncaught exceptions, then hand back to the platform so it still dies loudly. */
    fun install(context: Context) {
        val app = context.applicationContext
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, error ->
            runCatching {
                File(app.filesDir, CRASH).writeText(
                    "when   : ${stamp()}\nthread : ${thread.name}\n\n" +
                        error.stackTraceToString()
                )
            }
            previous?.uncaughtException(thread, error)
        }
    }

    fun crashFile(context: Context) = File(context.filesDir, CRASH)

    fun hasCrash(context: Context) = crashFile(context).canRead()

    fun clearCrash(context: Context) { crashFile(context).delete() }

    private fun stamp() =
        SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date())

    /**
     * The report body.
     *
     * Deliberately narrow: device and build facts, the model inventory, the app's own run
     * log, and a stored crash if there is one. No file paths outside the app, no media, no
     * identifiers.
     */
    fun compose(
        context: Context,
        log: String,
        device: String,
        models: List<String>,
        status: String,
    ): String {
        // Name AND code.  0.2.0 shipped twice -- versionCode 3, and the 4 that fixes the
        // v81 "no models" bug -- so the name alone cannot answer "which build is this?",
        // which is the first question any report about that bug needs answered.
        val v = runCatching {
            val pi = context.packageManager.getPackageInfo(context.packageName, 0)
            val code = if (Build.VERSION.SDK_INT >= 28) pi.longVersionCode
                       else @Suppress("DEPRECATION") pi.versionCode.toLong()
            pi.versionName + " (" + code + ")"
        }.getOrNull() ?: "?"
        val sb = StringBuilder()
        sb.append("FaceFusion Mobile bug report\n")
        sb.append("generated : ").append(stamp()).append('\n')
        sb.append("app       : ").append(v).append('\n')
        sb.append("device    : ").append(Build.MANUFACTURER).append(' ').append(Build.MODEL)
          .append("  (Android ").append(Build.VERSION.RELEASE)
          .append(", API ").append(Build.VERSION.SDK_INT).append(")\n")
        sb.append("npu       : ").append(device).append('\n')
        if (status.isNotEmpty()) sb.append("status    : ").append(status).append('\n')

        sb.append("\n-- models --\n")
        if (models.isEmpty()) sb.append("(none installed)\n")
        else models.forEach { sb.append("  ").append(it).append('\n') }

        sb.append("\n-- run log --\n")
        sb.append(if (log.isBlank()) "(empty)\n" else log)

        val crash = crashFile(context)
        if (crash.canRead()) {
            sb.append("\n-- last crash --\n")
            sb.append(runCatching { crash.readText() }.getOrDefault("(unreadable)"))
        }
        return sb.toString()
    }

    /**
     * The most an EXTRA_TEXT may carry.
     *
     * An Intent crosses a Binder transaction, and the whole transaction buffer is about
     * 1 MB and SHARED with everything else in flight. Going over it throws
     * TransactionTooLargeException from `startActivity` -- so the report that would fail is
     * the enormous one from the device that crashed hardest, which is the report most worth
     * having. The run log is capped at 4 KB, but the persisted crash is not bounded at all.
     *
     * 128 KB is far below the limit and far above any real report.
     */
    private const val MAX_SHARE_CHARS = 128 * 1024

    /**
     * Hand the text to whatever the user wants to send it with.
     *
     * @return null on success, or a reason the report could not be handed over.
     *
     * Returns rather than throws, and the caller SAYS SO. A share that silently does
     * nothing is indistinguishable from a dead button -- which is exactly how this feature
     * was reported.
     */
    fun share(context: Context, text: String): String? {
        // Trim the MIDDLE, not the tail: the header (version, device, models) is at the top
        // and the crash is at the bottom, and those are the two ends worth keeping.
        val body = if (text.length <= MAX_SHARE_CHARS) text else {
            val half = MAX_SHARE_CHARS / 2
            text.take(half) + "\n\n... [" + (text.length - MAX_SHARE_CHARS) +
                " characters omitted] ...\n\n" + text.takeLast(half)
        }
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_SUBJECT, "FaceFusion Mobile bug report")
            // EXTRA_TEXT rather than a file attachment: no FileProvider to configure, and
            // every target -- mail, chat, notes -- accepts plain text.
            putExtra(Intent.EXTRA_TEXT, body)
        }
        return try {
            context.startActivity(Intent.createChooser(send, "Send bug report"))
            null
        } catch (t: Throwable) {
            // ActivityNotFoundException on a device with nothing that accepts text/plain,
            // and TransactionTooLargeException if the cap above is ever raised past what
            // Binder will carry.
            t.message ?: t.javaClass.simpleName
        }
    }
}
