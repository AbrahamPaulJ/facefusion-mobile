package com.facefusion.mobile

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import java.io.File
import kotlin.concurrent.thread

/**
 * The model download, as a foreground service.
 *
 * A foreground service rather than a coroutine in the Activity because this moves ~275 MB:
 * the user will leave the app, and a background thread owned by a destroyed Activity is
 * killed at the system's convenience, halfway through a 196 MB file. The notification is
 * not decoration -- it is the price of being allowed to keep running, and it is also the
 * only progress the user can see once they have switched away.
 */
class DownloadService : Service() {

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_CANCEL) {
            ModelDownload.cancel()
            return START_NOT_STICKY
        }

        val chain = intent?.getStringExtra(EXTRA_TIERS).orEmpty()
        val dir = File(getExternalFilesDir(null), "models").apply { mkdirs() }

        createChannel()
        startForeground(NOTIF_ID, build(getString(R.string.notif_preparing), 0, 0))

        thread(name = "model-download") {
            // try/finally around the WHOLE run. Without it, anything `run` fails to catch
            // leaves the ongoing notification on screen for ever: ongoing notifications
            // cannot be swiped away, so the only way out is force-stopping the app.
            var err: String? = "interrupted"
            try {
                var lastPost = 0L
                err = ModelDownload.run(dir, chain) {
                    // Throttled: the notification manager rate-limits updates anyway, and
                    // posting per chunk is wasted work on a 275 MB transfer.
                    val now = System.currentTimeMillis()
                    if (now - lastPost > 500) {
                        lastPost = now
                        if (ModelDownload.running) notify(progressNotification())
                    }
                }
            } finally {
                finish(err)
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    /**
     * Replace the ongoing notification with a terminal one, or clear it.
     *
     * ⚠ Three separate faults lived in the two lines this replaces, and together they
     * produced one symptom: a finished download that sat in the shade looking stuck.
     *
     *  1. The success notification was built with `progress = 100, max = 100`, so
     *     `setProgress` was still called and the FULL PROGRESS BAR stayed on screen. A
     *     completed download is not a download at 100%; it has no bar at all.
     *  2. The title stayed "Downloading models" for ever, because it was hardcoded in
     *     `build` and never varied. So the shade read "Downloading models / Models ready"
     *     over a full bar -- which is exactly how a stalled transfer looks.
     *  3. `stat_sys_download` is the ANIMATED downloading arrow. It kept animating after
     *     the download had finished.
     *
     * Cancelling now removes the notification outright rather than reporting itself as
     * "Download failed: Cancelled". The user pressed cancel; they know.
     */
    private fun finish(err: String?) {
        val nm = getSystemService(NotificationManager::class.java)
        if (err == null) {
            notify(build(getString(R.string.notif_ready_text), 0, 0,
                         title = getString(R.string.notif_title_ready),
                         icon = android.R.drawable.stat_sys_download_done,
                         ongoing = false))
            // DETACH, so the terminal notification outlives the service -- the whole point
            // of a foreground service here is that the user has left the app, and the
            // result is the only thing that will tell them it finished.
            stopForeground(STOP_FOREGROUND_DETACH)
        } else if (CANCELLED.equals(err, ignoreCase = true)) {
            stopForeground(STOP_FOREGROUND_REMOVE)
            nm.cancel(NOTIF_ID)
        } else {
            notify(build(err, 0, 0,
                         title = getString(R.string.notif_title_failed),
                         icon = android.R.drawable.stat_notify_error,
                         ongoing = false))
            stopForeground(STOP_FOREGROUND_DETACH)
        }
    }

    private fun progressNotification(): Notification {
        val pct = (ModelDownload.progress * 100).toInt()
        val mb = ModelDownload.doneBytes / 1048576
        val totalMb = ModelDownload.totalBytes / 1048576
        return build(
            getString(R.string.notif_progress, ModelDownload.currentName, mb, totalMb),
            pct, 100,
            sub = getString(R.string.notif_file_of,
                            ModelDownload.fileIndex, ModelDownload.fileCount),
        )
    }

    private fun build(
        text: String,
        progress: Int,
        max: Int,
        sub: String? = null,
        ongoing: Boolean = true,
        title: String = getString(R.string.notif_title_downloading),
        icon: Int = android.R.drawable.stat_sys_download,
    ): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val b = Notification.Builder(this, CHANNEL)
            .setContentTitle(title)
            .setContentText(text)
            .setSmallIcon(icon)
            .setContentIntent(open)
            .setOnlyAlertOnce(true)
            .setOngoing(ongoing)
            // Tapping a FINISHED notification dismisses it. An ongoing one must not
            // auto-cancel: it would vanish on the first tap while the download continued
            // invisibly behind it.
            .setAutoCancel(!ongoing)
        if (sub != null) b.setSubText(sub)
        if (max > 0) b.setProgress(max, progress, false)
        if (ongoing) {
            val cancel = PendingIntent.getService(
                this, 1,
                Intent(this, DownloadService::class.java).setAction(ACTION_CANCEL),
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
            )
            b.addAction(Notification.Action.Builder(null as android.graphics.drawable.Icon?,
                                                    getString(R.string.notif_cancel),
                                                    cancel).build())
        }
        return b.build()
    }

    private fun notify(n: Notification) =
        getSystemService(NotificationManager::class.java).notify(NOTIF_ID, n)

    private fun createChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        // Recreated on every start so the NAME follows the app's language. Android keeps
        // a channel's user-set importance across this; only the text is refreshed.
        val ch = NotificationChannel(CHANNEL, getString(R.string.notif_channel_name),
                                     NotificationManager.IMPORTANCE_LOW)
        ch.description = getString(R.string.notif_channel_desc)
        getSystemService(NotificationManager::class.java).createNotificationChannel(ch)
    }

    companion object {
        private const val CHANNEL = "model_download"
        private const val NOTIF_ID = 1001
        private const val EXTRA_TIERS = "tiers"
        private const val ACTION_CANCEL = "com.facefusion.mobile.CANCEL_DOWNLOAD"

        /**
         * What [ModelDownload] returns when the user pressed Cancel.
         *
         * ⚠ Matched against the string because that is the whole vocabulary the downloader
         * has -- it reports every outcome as a message. It is NOT a localized string and
         * must not become one: this is an internal token, and the day it is translated the
         * cancel path silently starts reporting "Download failed" instead.
         */
        private const val CANCELLED = "Cancelled"

        /** @param tiers the comma-joined tier chain, best first -- "v81,v73,v68". */
        fun start(context: Context, tiers: String) {
            val i = Intent(context, DownloadService::class.java).putExtra(EXTRA_TIERS, tiers)
            context.startForegroundService(i)
        }
    }
}
