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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/**
 * What a running batch is doing, shared between the runner and the notification.
 *
 * A singleton for the same reason [ModelDownload] is one: the Activity owns the work but
 * the SERVICE has to describe it, and a service cannot reach into an Activity's fields.
 * Compose observes `running` directly so the UI needs no second copy.
 */
object BatchStatus {
    var running by mutableStateOf(false); private set
    @Volatile var index = 0; private set
    @Volatile var total = 0; private set
    @Volatile var name = ""; private set

    /**
     * Set by the notification's Cancel action.
     *
     * ⚠ Read by the runner IN ADDITION to its own cancel flag, never instead of it. The
     * button in the app and the button in the shade are two ways to ask for the same thing,
     * and a runner that only watched one of them would ignore whichever the user reached
     * for first.
     */
    @Volatile var cancelled = false

    fun begin(count: Int) {
        running = true; index = 0; total = count; name = ""; cancelled = false
    }

    fun item(i: Int, itemName: String) { index = i; name = itemName }

    fun end() { running = false; index = 0; total = 0; name = "" }
}

/**
 * Keeps the process alive while a batch runs — roadmap 14, part 2.
 *
 * ⚠ **What this does and does not buy, precisely.** A foreground service stops Android
 * killing the PROCESS for memory while a twelve-clip batch grinds away in the background,
 * and it gives the user a progress line and a Cancel button in the shade. It does NOT make
 * the batch survive the Activity being destroyed: the loop still runs in
 * `MainActivity.lifecycleScope`, so swiping the app out of recents still ends it.
 *
 * Fixing that properly means moving the runner into an object with its own scope, the way
 * `ModelDownload` is written — and moving a GATED processing path is not something to do
 * without a device to prove it on afterwards. This is the honest half that can be written
 * blind; the other half is still roadmap 14.
 *
 * The service therefore owns no work at all. It is a notification and a promise to the
 * scheduler, which is why it can be this short.
 */
class BatchService : Service() {

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_CANCEL) {
            BatchStatus.cancelled = true
            return START_NOT_STICKY
        }
        createChannel()
        startForeground(NOTIF_ID, build())
        // Poll rather than observe: the state lives in an object the runner writes from a
        // worker thread, and a notification refreshed twice a second is cheaper than any
        // callback plumbing across a service boundary.
        Thread({
            while (BatchStatus.running) {
                runCatching { Thread.sleep(500) }
                if (!BatchStatus.running) break
                runCatching {
                    getSystemService(NotificationManager::class.java).notify(NOTIF_ID, build())
                }
            }
            // The batch ended -- by finishing, by cancelling, or because the Activity that
            // owned it went away. All three mean the same thing here.
            runCatching { stopForeground(STOP_FOREGROUND_REMOVE) }
            stopSelf()
        }, "batch-notify").start()
        return START_NOT_STICKY
    }

    private fun build(): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val cancel = PendingIntent.getService(
            this, 1, Intent(this, BatchService::class.java).setAction(ACTION_CANCEL),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val b = Notification.Builder(this, CHANNEL)
            .setContentTitle(getString(R.string.notif_batch_title))
            .setContentText(
                if (BatchStatus.total > 0)
                    getString(R.string.notif_batch_text, BatchStatus.index,
                              BatchStatus.total, BatchStatus.name)
                else getString(R.string.notif_batch_title))
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentIntent(open)
            .setOnlyAlertOnce(true)
            .setOngoing(true)
            .setAutoCancel(false)
        if (BatchStatus.total > 0)
            b.setProgress(BatchStatus.total, BatchStatus.index, false)
        b.addAction(Notification.Action.Builder(
            null as android.graphics.drawable.Icon?,
            getString(R.string.notif_cancel), cancel).build())
        return b.build()
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val ch = NotificationChannel(CHANNEL, getString(R.string.notif_batch_channel),
                                     NotificationManager.IMPORTANCE_LOW)
        getSystemService(NotificationManager::class.java).createNotificationChannel(ch)
    }

    companion object {
        private const val CHANNEL = "batch_run"
        private const val NOTIF_ID = 1002
        private const val ACTION_CANCEL = "com.facefusion.mobile.CANCEL_BATCH"

        fun start(context: Context) {
            runCatching {
                context.startForegroundService(Intent(context, BatchService::class.java))
            }
        }

        /** Nothing to stop explicitly: the notify thread exits when [BatchStatus] ends. */
        fun stop(context: Context) {
            runCatching { context.stopService(Intent(context, BatchService::class.java)) }
        }
    }
}
