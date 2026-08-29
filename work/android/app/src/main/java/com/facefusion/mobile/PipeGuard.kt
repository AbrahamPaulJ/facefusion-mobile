package com.facefusion.mobile

import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit

/**
 * One owner at a time for the native pipeline.
 *
 * `g_pipe` on the C++ side is a single global: `init` replaces it and `release` frees it,
 * so two callers using it at once is a use-after-free, not a race for throughput. Until the
 * API server existed there was only one caller (the Activity) and its own `busy` /
 * `previewBusy` flags were enough. A server request arrives on a socket thread with no idea
 * what the screen is doing, so the invariant needs somewhere to live.
 *
 * A [Semaphore] rather than a lock, deliberately: the callers are coroutines that hop
 * threads at every `withContext`, and a ReentrantLock is thread-affine -- acquiring on the
 * IO dispatcher and releasing on the main one throws IllegalMonitorStateException. A
 * semaphore permit belongs to nobody in particular, which is exactly what is wanted here.
 *
 * ⚠ Every [acquire] that returns true MUST be matched by exactly one [release], from a
 * `finally`. A leaked permit is not a crash: it is a phone whose preview never updates
 * again until the app is restarted, which is far harder to recognise.
 */
object PipeGuard {

    private val permit = Semaphore(1, true)

    /** Who holds it, for logs and for telling the user why something is refusing. */
    @Volatile
    var holder: String? = null
        private set

    /**
     * How many times the pipeline has been taken, ever.
     *
     * This is how a caller that keeps the pipeline WARM across its own requests -- the API
     * server, and PreviewEngine -- finds out that someone else has been in. If my last
     * acquisition was number n and this one is not n+1, another holder ran in between, and
     * whatever they left loaded is theirs: their options, their source face. A warm flag
     * that survives that does not crash, it silently swaps the wrong face, which is worse.
     *
     * Counting acquisitions rather than asking each caller to announce an init is
     * deliberate: an announcement can be forgotten at one of six call sites, and the one
     * that is forgotten is the one that produces wrong output rather than an error.
     */
    @Volatile
    var sequence: Int = 0
        private set

    /** True when [previous] was the acquisition immediately before this one. */
    fun uninterrupted(previous: Int): Boolean = sequence == previous + 1

    /**
     * Take the pipeline, or fail immediately.
     *
     * [timeoutMs] of 0 is a bare try, which is what a UI action wants: the answer "the API
     * is mid-job" now beats the same answer after a wait the user did not ask for.
     */
    fun acquire(who: String, timeoutMs: Long = 0): Boolean {
        val got = if (timeoutMs <= 0) permit.tryAcquire()
                  else permit.tryAcquire(timeoutMs, TimeUnit.MILLISECONDS)
        if (got) {
            holder = who
            sequence++
        }
        return got
    }

    fun release() {
        holder = null
        permit.release()
    }

    /** For a message: "the app is busy" reads better than a failure with no subject. */
    fun busyMessage(): String = holder?.let { "busy: $it" } ?: "busy"
}
