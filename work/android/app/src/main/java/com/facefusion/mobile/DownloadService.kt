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

        val tier = intent?.getStringExtra(EXTRA_TIER).orEmpty()
        val dir = File(getExternalFilesDir(null), "models").apply { mkdirs() }

        createChannel()
        startForeground(NOTIF_ID, build("Preparing...", 0, 0))

        thread(name = "model-download") {
            var lastPost = 0L
            val err = ModelDownload.run(dir, tier) {
                // Throttled: the notification manager rate-limits updates anyway, and
                // posting per chunk is wasted work on a 275 MB transfer.
                val now = System.currentTimeMillis()
                if (now - lastPost > 500) {
                    lastPost = now
                    if (ModelDownload.running) notify(progressNotification())
                }
            }
            notify(
                if (err == null) build("Models ready", 100, 100, ongoing = false)
                else build("Download failed: $err", 0, 0, ongoing = false)
            )
            stopForeground(STOP_FOREGROUND_DETACH)
            stopSelf()
        }
        return START_NOT_STICKY
    }

    private fun progressNotification(): Notification {
        val pct = (ModelDownload.progress * 100).toInt()
        val mb = ModelDownload.doneBytes / 1048576
        val totalMb = ModelDownload.totalBytes / 1048576
        return build(
            "${ModelDownload.currentName}  ($mb / $totalMb MB)",
            pct, 100,
            sub = "File ${ModelDownload.fileIndex} of ${ModelDownload.fileCount}",
        )
    }

    private fun build(
        text: String,
        progress: Int,
        max: Int,
        sub: String? = null,
        ongoing: Boolean = true,
    ): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val b = Notification.Builder(this, CHANNEL)
            .setContentTitle("Downloading models")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentIntent(open)
            .setOnlyAlertOnce(true)
            .setOngoing(ongoing)
        if (sub != null) b.setSubText(sub)
        if (max > 0) b.setProgress(max, progress, false)
        if (ongoing) {
            val cancel = PendingIntent.getService(
                this, 1,
                Intent(this, DownloadService::class.java).setAction(ACTION_CANCEL),
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
            )
            b.addAction(Notification.Action.Builder(null as android.graphics.drawable.Icon?,
                                                    "Cancel", cancel).build())
        }
        return b.build()
    }

    private fun notify(n: Notification) =
        getSystemService(NotificationManager::class.java).notify(NOTIF_ID, n)

    private fun createChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val ch = NotificationChannel(CHANNEL, "Model download",
                                     NotificationManager.IMPORTANCE_LOW)
        ch.description = "Progress while the NPU models are downloading"
        getSystemService(NotificationManager::class.java).createNotificationChannel(ch)
    }

    companion object {
        private const val CHANNEL = "model_download"
        private const val NOTIF_ID = 1001
        private const val EXTRA_TIER = "tier"
        private const val ACTION_CANCEL = "com.facefusion.mobile.CANCEL_DOWNLOAD"

        fun start(context: Context, tier: String) {
            val i = Intent(context, DownloadService::class.java).putExtra(EXTRA_TIER, tier)
            context.startForegroundService(i)
        }
    }
}
