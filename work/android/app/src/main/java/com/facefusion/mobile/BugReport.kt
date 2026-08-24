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
        val v = runCatching {
            context.packageManager.getPackageInfo(context.packageName, 0).versionName
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

    /** Hand the text to whatever the user wants to send it with. */
    fun share(context: Context, text: String) {
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_SUBJECT, "FaceFusion Mobile bug report")
            // EXTRA_TEXT rather than a file attachment: no FileProvider to configure, and
            // every target -- mail, chat, notes -- accepts plain text.
            putExtra(Intent.EXTRA_TEXT, text)
        }
        context.startActivity(Intent.createChooser(send, "Send bug report"))
    }
}
