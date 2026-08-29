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
 * The API server's lifetime, as a foreground service.
 *
 * The same reason the download is one: the point of a server is that the PC can use it
 * while the phone is in a pocket with the screen off, and a socket owned by a stopped
 * Activity is closed at the system's convenience -- mid-request, from the client's side.
 * The notification is the price of being allowed to keep the port open, and it doubles as
 * the only honest indicator that this phone is currently reachable, which for a face
 * swapper is worth showing whether or not anyone reads it.
 */
class ApiService : Service() {

    private var server: ApiServer? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            shutdown()
            return START_NOT_STICKY
        }
        if (server != null) return START_STICKY

        val lan = intent?.getBooleanExtra(EXTRA_LAN, false) ?: false
        createChannel()
        // Before the socket: startForeground has a deadline, and a bind failure that
        // happens first would kill the process for never calling it.
        startForeground(NOTIF_ID, notification("Starting..."))

        val s = ApiServer(applicationContext, lan) { line ->
            log = (log + line + "\n").takeLast(4000)
            android.util.Log.i("ffapi", line)
        }
        try {
            s.start()
        } catch (e: Exception) {
            // Almost always "Address already in use": a previous process still holds it.
            log = (log + "could not start: " + e.message + "\n").takeLast(4000)
            error = e.message
            running = false
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }
        server = s
        error = null
        allowLan = lan
        address = if (lan) ApiServer.lanUrl(applicationContext)
                  else "http://127.0.0.1:" + ApiServer.PORT
        running = true
        notify(notification(address))
        return START_STICKY
    }

    override fun onDestroy() {
        shutdown()
        super.onDestroy()
    }

    private fun shutdown() {
        server?.stop()
        server = null
        running = false
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun notification(text: String): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val stop = PendingIntent.getService(
            this, 1, Intent(this, ApiService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setContentTitle("FaceFusion API is running")
            .setContentText(text)
            .setOngoing(true)
            .setContentIntent(open)
            .addAction(Notification.Action.Builder(null, "Stop", stop).build())
            .build()
    }

    private fun notify(n: Notification) =
        getSystemService(NotificationManager::class.java).notify(NOTIF_ID, n)

    private fun createChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val ch = NotificationChannel(CHANNEL, "Remote API",
                                     NotificationManager.IMPORTANCE_LOW)
        ch.description = "Shown while this phone is serving swaps over HTTP"
        getSystemService(NotificationManager::class.java).createNotificationChannel(ch)
    }

    companion object {
        private const val CHANNEL = "api_server"
        private const val NOTIF_ID = 1002
        private const val ACTION_STOP = "com.facefusion.mobile.STOP_API"
        private const val EXTRA_LAN = "lan"
        private const val PREFS = "ffapi"

        /** Compose state, so Settings redraws when the service starts or dies. */
        var running by mutableStateOf(false); private set
        var address by mutableStateOf(""); private set
        var allowLan by mutableStateOf(false); private set
        var error by mutableStateOf<String?>(null); private set
        var log by mutableStateOf(""); private set

        /**
         * Remember which way the switch was thrown, not whether it was on.
         *
         * A reinstall or a low-memory kill takes the process, and the switch came back
         * saying loopback while the person had deliberately opened it to the network --
         * so the address they were using stopped answering with nothing on screen to say
         * why. The SERVER still does not start by itself: opening a port is a decision,
         * and it stays one.
         */
        fun restore(ctx: Context) {
            allowLan = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .getBoolean("lan", false)
        }

        /**
         * @param remember whether this choice is the USER's. The `--es api start` intent
         *   always asks for loopback, and it must not write that down: doing so turned a
         *   convenience command into something that silently switched off a setting the
         *   person had deliberately turned on.
         */
        fun start(ctx: Context, lan: Boolean, remember: Boolean = true) {
            if (remember) ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .edit().putBoolean("lan", lan).apply()
            ctx.startForegroundService(
                Intent(ctx, ApiService::class.java).putExtra(EXTRA_LAN, lan))
        }

        fun stop(ctx: Context) {
            ctx.startService(Intent(ctx, ApiService::class.java).setAction(ACTION_STOP))
        }
    }
}
