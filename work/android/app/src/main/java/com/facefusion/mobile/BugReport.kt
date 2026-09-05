package com.facefusion.mobile

import android.app.ActivityManager
import android.app.ApplicationExitInfo
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
 * That handler sees JVM exceptions ONLY, which for this app is the less interesting half:
 * the pipeline is C++, and a native SIGSEGV never unwinds through Java. [exitHistory] is
 * the other half -- see its doc.
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
     * What the SYSTEM recorded about this app's last few deaths.
     *
     * ## Why this is here
     *
     * [install]'s handler catches uncaught JVM exceptions. A native crash kills the process
     * outright -- no unwind, no handler, no file written -- so the report for the crash most
     * worth reporting was the report with an empty "last crash" section. 0.6.0 had a SIGSEGV
     * in `ffcv::warpAffine` that fired whenever a face reached the corner of the frame, and
     * the only way to see it was `adb logcat -b crash` over a cable.
     *
     * The system writes a full tombstone to `/data/tombstones`, which an unprivileged app
     * cannot open. `ActivityManager.getHistoricalProcessExitReasons` is the part it CAN
     * read: reason, signal, description and timestamp for the last several exits, no
     * permission required, this package only.
     *
     * ⚠ Every exit is listed, not only the crashes. "The user swiped it away" and "the
     * kernel killed it for memory" are answers too, and a section that only ever showed
     * crashes could not tell "it crashed" apart from "it was killed" -- which is the most
     * common wrong diagnosis in any report about an app that "just closes".
     */
    private fun exitHistory(context: Context): String {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            ?: return "(no ActivityManager)"
        val infos = runCatching {
            am.getHistoricalProcessExitReasons(context.packageName, 0, 6)
        }.getOrNull() ?: return "(unavailable)"
        if (infos.isEmpty()) return "(nothing recorded)"

        val sb = StringBuilder()
        for (i in infos) {
            sb.append("  ").append(stamp(i.timestamp)).append("  ").append(reasonName(i.reason))
            // status carries the SIGNAL for a native crash and the exit code otherwise: it
            // is the field that says SIGSEGV rather than merely "it crashed".
            if (i.status != 0) sb.append(" status=").append(i.status)
            i.description?.let { if (it.isNotBlank()) sb.append("  ").append(it) }
            sb.appendLine()
            if (i.reason == ApplicationExitInfo.REASON_CRASH_NATIVE ||
                i.reason == ApplicationExitInfo.REASON_ANR) {
                trace(i)?.let { sb.append(it) }
            }
        }
        return sb.toString()
    }

    private fun stamp(millis: Long) =
        SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date(millis))

    private fun reasonName(reason: Int) = when (reason) {
        ApplicationExitInfo.REASON_ANR -> "ANR"
        ApplicationExitInfo.REASON_CRASH -> "crash (JVM)"
        ApplicationExitInfo.REASON_CRASH_NATIVE -> "CRASH (native)"
        ApplicationExitInfo.REASON_DEPENDENCY_DIED -> "dependency died"
        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "excessive resource use"
        ApplicationExitInfo.REASON_EXIT_SELF -> "exited itself"
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "init failure"
        ApplicationExitInfo.REASON_LOW_MEMORY -> "low memory"
        ApplicationExitInfo.REASON_OTHER -> "other"
        ApplicationExitInfo.REASON_PERMISSION_CHANGE -> "permission change"
        ApplicationExitInfo.REASON_SIGNALED -> "signalled"
        ApplicationExitInfo.REASON_USER_REQUESTED -> "user closed it"
        ApplicationExitInfo.REASON_USER_STOPPED -> "user stopped it"
        else -> "reason " + reason
    }

    /** How much of one tombstone to keep. Enough for a backtrace, far below the share cap. */
    private const val MAX_TRACE_CHARS = 12 * 1024

    /**
     * The tombstone itself, as far as an app is allowed to read one.
     *
     * ⚠ For an ANR this stream is the plain text `adb` would print. For a NATIVE CRASH it is
     * a PROTOBUF (`tombstone.proto`), not text -- so rather than link a protobuf runtime to
     * decode a diagnostic, the printable runs are pulled out of it. That keeps the part that
     * answers "where": the signal, the library paths and the demangled frame symbols. It
     * loses the frame ORDER and the addresses, so read the result as a set of names and not
     * as a backtrace. `ffcv::warpAffine` appearing at all was the entire diagnosis in the
     * case this was written for.
     */
    private fun trace(info: ApplicationExitInfo): String? {
        val bytes = runCatching {
            info.traceInputStream?.use { it.readBytes() }
        }.getOrNull() ?: return null
        if (bytes.isEmpty()) return null

        val nl = 10.toByte()
        val printable = bytes.count { (it >= 0x20 && it < 0x7F) || it == nl }
        val text = if (printable > bytes.size / 10 * 9) {
            String(bytes, Charsets.UTF_8)          // an ANR trace: already text
        } else {
            val out = StringBuilder()
            val run = StringBuilder()
            fun flush() {
                // 8 is long enough to skip protobuf framing that happens to be ASCII, and
                // short enough to keep "SIGSEGV" in the company of its neighbours.
                if (run.length >= 8) out.append("    ").append(run).appendLine()
                run.setLength(0)
            }
            for (b in bytes) {
                val c = b.toInt() and 0xFF
                if (c in 0x20..0x7E) run.append(c.toChar()) else flush()
            }
            flush()
            out.toString()
        }
        if (text.isBlank()) return null
        val body = if (text.length <= MAX_TRACE_CHARS) text
                   else text.take(MAX_TRACE_CHARS) + "    ... [trace truncated] ..."
        return "  -- trace --" + System.lineSeparator() + body.trimEnd('\n') +
            System.lineSeparator()
    }

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

        // LAST, because it is the longest section. But never omitted: a native crash leaves
        // the section above empty, and then this is the only evidence in the report.
        sb.append("\n-- how this app last died (system record) --\n")
        sb.append(runCatching { exitHistory(context) }.getOrDefault("(unavailable)"))
        sb.append('\n')
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
