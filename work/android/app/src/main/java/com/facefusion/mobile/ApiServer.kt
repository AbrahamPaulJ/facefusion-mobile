package com.facefusion.mobile

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.OutputStream
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.NetworkInterface
import java.net.ServerSocket
import java.net.Socket
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * The swap, over HTTP, so a PC can drive the phone's NPU.
 *
 * Local Dream solves this by making the inference itself a standalone native server and
 * reducing its app to a launcher. That shape does not fit here, and copying it would be a
 * mistake: on this project the pipeline is native but three things around it are not.
 * The content gate's POLICY is Kotlin (thresholds, video sampling, the refusal); the video
 * path is MediaCodec and MediaExtractor, which are Android APIs with no native equivalent
 * in this tree; and the models live in the app's own external files dir. A native server
 * would have to reimplement all three, and the first one it got wrong would be the gate --
 * a fourth processing path with nothing guarding it, reachable from any machine on the
 * network. So the server lives INSIDE the app and calls exactly what the screen calls.
 *
 * ## The API
 *
 * Bodies are raw bytes, not multipart: the client is `curl --data-binary @file`, and a
 * multipart parser is a hundred lines that can only add ways to be wrong.
 *
 * ```
 * GET  /                           -> the web UI, a single self-contained page
 * GET  /health                      -> JSON: tier, arch, models, whether a source is set
 * POST /source   <image bytes>      -> JSON: sets the face to swap FROM, gated, kept warm
 * POST /swap     <image bytes>      -> image/png, the swapped still
 * POST /swap_video <mp4 bytes>      -> video/mp4, the whole clip (blocks; can take minutes)
 * ```
 *
 * Options come from whatever the app's Advanced panel is set to, overridable per request
 * with a query string: `/swap?weight=0.8&boost=2&enhancer=1&largest=1&blur=0.3`.
 *
 * Every response is `Connection: close` with a Content-Length. Keep-alive buys nothing when
 * a single request occupies the NPU for the whole time it is open.
 *
 * ## What it refuses
 *
 * - The gate runs on the source in [source] and on every frame in [swap] and inside
 *   VideoSwapper's own path, which is the same code `refreshSwapped` and `runSwap` use.
 *   There is no way in here that skips it.
 * - One job at a time, through [PipeGuard]. A request that arrives while the screen is
 *   swapping gets 503 rather than sharing a global the C++ side assumes is exclusive.
 */
class ApiServer(
    private val ctx: Context,
    /**
     * 0.0.0.0 rather than loopback. Off by default, and the only thing standing between
     * this port and the network: there is no token, by request. Loopback means the only
     * way in is from the phone itself or through `adb forward`, which needs USB or an
     * authorised adb connection -- a machine that could install an APK anyway.
     */
    private val allowLan: Boolean,
    private val log: (String) -> Unit,
) {

    private var socket: ServerSocket? = null
    private val running = AtomicBoolean(false)
    // Two threads, not one: a client that opens a connection and stalls would otherwise
    // hold the door shut for everyone. The NPU work behind them is serialised by PipeGuard
    // regardless, so this is about liveness, not parallelism.
    private val workers = Executors.newFixedThreadPool(2)

    // ---- the warm pipeline, across requests
    //
    // Reloading the models per request costs seconds; a PC feeding frames would spend all
    // of its time there. So the source and the loaded options are kept, and reloaded only
    // when they are actually stale -- see PipeGuard.sequence for how "someone else used the
    // pipeline in between" is detected without asking every other caller to announce itself.
    private var sourceBgr: ByteArray? = null
    private var sourceW = 0
    private var sourceH = 0
    private var loadedOpts: SwapOptions? = null
    private var sourceApplied = false
    private var lastSeq = -2

    fun start(): Int {
        val addr = if (allowLan) InetAddress.getByName("0.0.0.0")
                   else InetAddress.getByName("127.0.0.1")
        val s = ServerSocket()
        s.reuseAddress = true
        s.bind(InetSocketAddress(addr, PORT), 8)
        socket = s
        running.set(true)
        Thread({ acceptLoop(s) }, "ffapi-accept").apply { isDaemon = true }.start()
        log("API listening on " + (if (allowLan) lanUrl(ctx) else "http://127.0.0.1:$PORT"))
        return s.localPort
    }

    fun stop() {
        running.set(false)
        runCatching { socket?.close() }
        socket = null
        workers.shutdownNow()
        // Whatever was loaded belonged to this server. Leaving it resident would hold the
        // context binaries in memory for a service the user has switched off.
        if (PipeGuard.acquire("api-stop")) {
            try {
                if (loadedOpts != null) NativePipe.release()
            } finally {
                loadedOpts = null; sourceApplied = false
                PipeGuard.release()
            }
        }
        log("API stopped")
    }

    private fun acceptLoop(s: ServerSocket) {
        while (running.get()) {
            val conn = try { s.accept() } catch (e: Exception) {
                if (running.get()) log("accept failed: " + e.message)
                return
            }
            runCatching { workers.execute { handle(conn) } }
                .onFailure { runCatching { conn.close() } }
        }
    }

    // ------------------------------------------------------------------ HTTP

    private class Request(
        val method: String,
        val path: String,
        val query: Map<String, String>,
        val headers: Map<String, String>,
        val body: ByteArray,
        /**
         * Set instead of [body] for uploads that are too big to hold.
         *
         * A clip arrived as: the request body in a ByteArray, a copy written to a temp
         * file, and then the finished mp4 read back into a third array to be sent. Three
         * full copies of the same video in a heap that also has ~275 MB of context
         * binaries in it -- the process was killed mid-job, which from the client looks
         * exactly like the server hanging up.
         */
        val bodyFile: File? = null,
    )

    private fun handle(conn: Socket) {
        conn.soTimeout = READ_TIMEOUT_MS
        conn.use { c ->
            val out = c.getOutputStream()
            val req = try { readRequest(BufferedInputStream(c.getInputStream())) } catch (e: Exception) {
                respond(out, 400, "text/plain", ("bad request: " + e.message).toByteArray())
                return
            }
            if (req == null) { respond(out, 400, "text/plain", "bad request".toByteArray()); return }

            try {
                route(req, out)
            } catch (t: Throwable) {
                log("500 " + req.path + ": " + describe(t))
                respond(out, 500, "application/json", json("error" to describe(t)))
            }
        }
    }

    /** Request line, headers, then exactly Content-Length bytes. No chunked encoding. */
    private fun readRequest(input: BufferedInputStream): Request? {
        val head = ByteArrayOutputStream()
        var state = 0
        while (state < 4) {
            val b = input.read()
            if (b < 0) return null
            head.write(b)
            state = when {
                b == '\r'.code && (state == 0 || state == 2) -> state + 1
                b == '\n'.code && (state == 1 || state == 3) -> state + 1
                else -> 0
            }
            if (head.size() > 64 * 1024) return null
        }
        val lines = head.toString("UTF-8").split("\r\n").filter { it.isNotEmpty() }
        if (lines.isEmpty()) return null
        val parts = lines[0].split(" ")
        if (parts.size < 2) return null
        val headers = lines.drop(1).mapNotNull {
            val i = it.indexOf(':')
            if (i <= 0) null else it.substring(0, i).trim().lowercase() to it.substring(i + 1).trim()
        }.toMap()

        val rawPath = parts[1]
        val q = rawPath.indexOf('?')
        val path = if (q < 0) rawPath else rawPath.substring(0, q)
        val query = if (q < 0) emptyMap() else rawPath.substring(q + 1).split("&")
            .mapNotNull {
                val i = it.indexOf('=')
                if (i <= 0) null else it.substring(0, i) to it.substring(i + 1)
            }.toMap()

        val method = parts[0].uppercase()
        val len = headers["content-length"]?.toLongOrNull() ?: 0L

        // A video goes to disk as it arrives and is never held whole.
        if (path == "/swap_video") {
            if (len > MAX_UPLOAD)
                throw IllegalArgumentException("upload over ${MAX_UPLOAD / 1048576} MB")
            val f = File(ctx.cacheDir, "api_in.mp4")
            var left = len
            val buf = ByteArray(64 * 1024)
            f.outputStream().use { o ->
                while (left > 0) {
                    val n = input.read(buf, 0, minOf(buf.size.toLong(), left).toInt())
                    if (n < 0) throw IllegalStateException(
                        "client closed with $left of $len bytes to go")
                    o.write(buf, 0, n)
                    left -= n
                }
            }
            return Request(method, path, query, headers, ByteArray(0), f)
        }

        if (len > MAX_BODY) throw IllegalArgumentException("body over ${MAX_BODY / 1048576} MB")
        val body = ByteArray(len.toInt())
        var read = 0
        while (read < body.size) {
            val n = input.read(body, read, body.size - read)
            if (n < 0) throw IllegalStateException("client closed after $read of $len bytes")
            read += n
        }
        return Request(method, path, query, headers, body)
    }

    private fun respond(out: OutputStream, code: Int, type: String, body: ByteArray) {
        val reason = when (code) {
            200 -> "OK"; 400 -> "Bad Request"; 401 -> "Unauthorized"; 403 -> "Forbidden"
            404 -> "Not Found"; 409 -> "Conflict"; 503 -> "Service Unavailable"
            else -> "Internal Server Error"
        }
        val head = StringBuilder("HTTP/1.1 $code $reason\r\n")
            .append("Content-Type: ").append(type).append("\r\n")
            .append("Content-Length: ").append(body.size).append("\r\n")
            .append("Connection: close\r\n\r\n")
        runCatching {
            out.write(head.toString().toByteArray())
            out.write(body)
            out.flush()
        }
    }

    /** The same headers, but the body comes off disk rather than out of the heap. */
    private fun respondFile(out: OutputStream, type: String, file: File) {
        val head = StringBuilder("HTTP/1.1 200 OK\r\n")
            .append("Content-Type: ").append(type).append("\r\n")
            .append("Content-Length: ").append(file.length()).append("\r\n")
            .append("Connection: close\r\n\r\n")
        runCatching {
            out.write(head.toString().toByteArray())
            file.inputStream().use { it.copyTo(out, 64 * 1024) }
            out.flush()
        }
    }

    private fun json(vararg pairs: Pair<String, Any?>): ByteArray =
        pairs.joinToString(",", "{", "}") { (k, v) ->
            val value = when (v) {
                null -> "null"
                is Number, is Boolean -> v.toString()
                is List<*> -> v.joinToString(",", "[", "]") { "\"" + esc(it.toString()) + "\"" }
                else -> "\"" + esc(v.toString()) + "\""
            }
            "\"" + esc(k) + "\":" + value
        }.toByteArray()

    /**
     * An exception a person can act on.
     *
     * Several of the ones this path can throw have no message at all -- MediaCodec's
     * CodecException is the one that bit -- so the class name goes in front, and the cause
     * behind, because "configure failed" is only useful once you know what refused.
     */
    private fun describe(t: Throwable): String {
        val parts = generateSequence(t) { it.cause }.take(3).map { e ->
            val m = e.message?.trim().orEmpty()
            if (m.isEmpty()) e.javaClass.simpleName else e.javaClass.simpleName + ": " + m
        }
        log(t.stackTraceToString().lineSequence().take(12).joinToString(" | "))
        return parts.joinToString(" <- ")
    }

    private fun esc(s: String) = s.replace("\\", "\\\\").replace("\"", "\\\"")
        .replace("\n", " ").replace("\r", " ")

    // ------------------------------------------------------------------ routes

    private fun route(req: Request, out: OutputStream) = when {
        req.method == "GET" && req.path == "/" -> page(out)
        req.method == "GET" && req.path == "/health" -> health(out)
        req.method == "POST" && req.path == "/source" -> source(req, out)
        req.method == "POST" && req.path == "/swap" -> swap(req, out)
        req.method == "POST" && req.path == "/swap_video" -> swapVideo(req, out)
        else -> respond(out, 404, "application/json", json("error" to "no route ${req.method} ${req.path}"))
    }

    /**
     * The web UI.
     *
     * Served by the phone, so it assumes nothing is reachable: no CDN, no framework, one
     * file with its CSS and JS inline. A page that needed the internet would fail exactly
     * where this is most useful -- a phone and a laptop alone on a hotspot.
     */
    private fun page(out: OutputStream) {
        val html = runCatching { ctx.assets.open("webui.html").use { it.readBytes() } }
            .getOrElse {
                return respond(out, 500, "text/plain",
                               "webui.html missing from assets".toByteArray())
            }
        respond(out, 200, "text/html; charset=utf-8", html)
    }

    private fun health(out: OutputStream) {
        val tier = ModelPaths.tier(ctx)
        val opts = SwapOptions.load(ctx)
        val missing = ModelPaths.missing(ctx, tier, opts.swapper)
        respond(out, 200, "application/json", json(
            "ok" to true,
            "version" to appVersion(ctx),
            "tier" to tier,
            "models_missing" to missing,
            "ready" to missing.isEmpty(),
            "source_set" to (sourceBgr != null),
            "busy" to (PipeGuard.holder != null),
            "holder" to PipeGuard.holder,
            "enhancer" to ModelPaths.present(ModelPaths.dir(ctx), tier, "gpen"),
        ))
    }

    /**
     * The face to swap FROM.
     *
     * Gated before it is embedded, and in that order for the same reason the preview does
     * it that way: the check needs the models up, but `setSource` is already processing --
     * it detects, aligns and embeds. Anything that must not be processed has to be refused
     * in the gap between those two.
     */
    private fun source(req: Request, out: OutputStream) {
        val bmp = decode(req.body)
            ?: return respond(out, 400, "application/json", json("error" to "cannot decode image"))
        withPipeline(out, SwapOptions.load(ctx).overrides(req.query), applySource = false) { _ ->
            val gate = ContentGate.checkImage(bmp)
            log("api source score %+.3f".format(gate.score))
            if (!gate.ok) {
                respond(out, 403, "application/json",
                        json("error" to ContentGate.message("the source image", gate),
                             "verdict" to gate.verdict.name,
                             "score" to gate.score,
                             // The native reason when the check could not RUN. A refusal
                             // and a broken checker are different problems and must not
                             // arrive looking the same.
                             "detail" to gate.detail))
                return@withPipeline
            }
            val soft = bmp.copy(Bitmap.Config.ARGB_8888, false)
            val px = IntArray(soft.width * soft.height)
            soft.getPixels(px, 0, soft.width, 0, 0, soft.width, soft.height)
            val bgr = NativePipe.argbToBgr(px, soft.width, soft.height)
            if (!NativePipe.setSource(bgr, soft.width, soft.height)) {
                respond(out, 400, "application/json",
                        json("error" to "no face found in the source image"))
                return@withPipeline
            }
            sourceBgr = bgr; sourceW = soft.width; sourceH = soft.height
            sourceApplied = true
            respond(out, 200, "application/json",
                    json("ok" to true, "width" to soft.width, "height" to soft.height))
        }
    }

    /** One still, swapped. The frame is gated exactly as the preview pane gates it. */
    private fun swap(req: Request, out: OutputStream) {
        if (sourceBgr == null)
            return respond(out, 409, "application/json", json("error" to "POST /source first"))
        val bmp = decode(req.body)
            ?: return respond(out, 400, "application/json", json("error" to "cannot decode image"))
        withPipeline(out, SwapOptions.load(ctx).overrides(req.query)) { _ ->
            val gate = ContentGate.checkImage(bmp)
            if (!gate.ok) {
                respond(out, 403, "application/json",
                        json("error" to ContentGate.message("the target image", gate),
                             "verdict" to gate.verdict.name,
                             "score" to gate.score,
                             "detail" to gate.detail))
                return@withPipeline
            }
            val soft = bmp.copy(Bitmap.Config.ARGB_8888, false)
            val w = soft.width; val h = soft.height
            val px = IntArray(w * h)
            soft.getPixels(px, 0, w, 0, 0, w, h)
            val bgr = NativePipe.argbToBgr(px, w, h)
            val faces = NativePipe.processFrame(bgr, w, h)
            if (faces < 0) {
                respond(out, 500, "application/json", json("error" to NativePipe.lastError()))
                return@withPipeline
            }
            if (faces == 0) {
                // processFrame leaves the buffer untouched when it finds nothing, so
                // returning it would be an unswapped image with a 200 on it.
                respond(out, 422, "application/json", json("error" to "no face detected"))
                return@withPipeline
            }
            val argb = NativePipe.bgrToArgb(bgr, w, h, w, h)
            val outBmp = Bitmap.createBitmap(argb, w, h, Bitmap.Config.ARGB_8888)
            val png = ByteArrayOutputStream()
            outBmp.compress(Bitmap.CompressFormat.PNG, 100, png)
            respond(out, 200, "image/png", png.toByteArray())
        }
    }

    /**
     * A whole clip.
     *
     * Blocking, with no progress: an SSE channel would be nicer to watch but the client is
     * a script, and a script that has to parse events to learn it can stop waiting is worse
     * than one that waits. Minutes-long requests are the normal case, so clients need their
     * own timeouts raised -- documented in work/api/README.md.
     */
    private fun swapVideo(req: Request, out: OutputStream) {
        if (sourceBgr == null)
            return respond(out, 409, "application/json", json("error" to "POST /source first"))
        // Already on disk when it was read off the socket; only a tiny body would not be.
        val inFile = req.bodyFile ?: File(ctx.cacheDir, "api_in.mp4").apply {
            writeBytes(req.body)
        }
        val outFile = File(ctx.cacheDir, "api_out.mp4")
        withPipeline(out, SwapOptions.load(ctx).overrides(req.query)) { opts ->
            val gate = ContentGate.checkVideo(inFile)
            // `detail` is an ARGUMENT, never part of the format string: it contains a "%)"
            // that gets read as a conversion.
            log("api target content: %s, worst %+.3f".format(gate.detail, gate.score))
            if (!gate.ok) {
                respond(out, 403, "application/json",
                        json("error" to ContentGate.message("the target video", gate),
                             "verdict" to gate.verdict.name,
                             "score" to gate.score,
                             "detail" to gate.detail))
                return@withPipeline
            }
            val t0 = System.currentTimeMillis()
            var frames = 0
            val r = VideoSwapper(
                outputFps = opts.outputFps,
                onProgress = { done, _ -> frames = done },
                onLog = { log(it) },
            ).swap(inFile.absolutePath, outFile.absolutePath)
            r.fold({
                val secs = (System.currentTimeMillis() - t0) / 1000.0
                log("api video: %d frames in %.1f s".format(frames, secs))
                respondFile(out, "video/mp4", outFile)
            }, {
                // `message` alone is not enough and this cost a debugging round trip:
                // MediaCodec.CodecException carries an empty one, so the whole failure
                // arrived at the client as {"error":""}. The TYPE is the useful half.
                respond(out, 500, "application/json", json("error" to describe(it)))
            })
        }
    }

    // ------------------------------------------------------------------ pipeline

    /**
     * Run [block] owning the pipeline, loaded for [opts] with the current source applied.
     *
     * Refuses rather than waits when the screen has it: a phone in someone's hand is doing
     * something a person is watching, and a queued request that lands minutes later is
     * indistinguishable from a hang.
     */
    private inline fun withPipeline(
        out: OutputStream,
        opts: SwapOptions,
        applySource: Boolean = true,
        block: (SwapOptions) -> Unit,
    ) {
        if (!PipeGuard.acquire("api", ACQUIRE_WAIT_MS)) {
            respond(out, 503, "application/json", json("error" to PipeGuard.busyMessage()))
            return
        }
        try {
            // Someone else held the pipeline since the last request, so what is loaded is
            // theirs: their options, their source face.
            if (!PipeGuard.uninterrupted(lastSeq)) { loadedOpts = null; sourceApplied = false }
            lastSeq = PipeGuard.sequence

            val tier = ModelPaths.tier(ctx)
            val missing = ModelPaths.missing(ctx, tier, opts.swapper)
            if (missing.isNotEmpty()) {
                respond(out, 503, "application/json",
                        json("error" to "models not on the device", "missing" to missing))
                return
            }
            if (loadedOpts != opts) {
                val lib = ctx.applicationInfo.nativeLibraryDir
                if (!NativePipe.init(lib, lib, ModelPaths.dir(ctx).absolutePath, opts)) {
                    respond(out, 500, "application/json",
                            json("error" to ("init: " + NativePipe.lastError())))
                    return
                }
                loadedOpts = opts
                sourceApplied = false
            }
            if (applySource && !sourceApplied) {
                val bgr = sourceBgr
                if (bgr == null) {
                    respond(out, 409, "application/json", json("error" to "POST /source first"))
                    return
                }
                // Re-embedding what the client already sent, rather than making it upload
                // the face again because the screen borrowed the NPU.
                if (!NativePipe.setSource(bgr, sourceW, sourceH)) {
                    respond(out, 500, "application/json",
                            json("error" to "source no longer usable: " + NativePipe.lastError()))
                    return
                }
                sourceApplied = true
            }
            block(opts)
        } finally {
            PipeGuard.release()
        }
    }

    /**
     * Decode through [ImageDecoder], never BitmapFactory.
     *
     * BitmapFactory ignores EXIF orientation, and a sideways face is not a cosmetic
     * problem: yoloface is not rotation invariant, so it is simply not detected and the
     * request fails with "no face found". That bug was paid for once already on the source
     * picker; it is not being reintroduced on a second entry point.
     */
    private fun decode(bytes: ByteArray): Bitmap? = runCatching {
        ImageDecoder.decodeBitmap(ImageDecoder.createSource(ByteBuffer.wrap(bytes))) { d, _, _ ->
            d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            d.isMutableRequired = true
        }
    }.getOrNull()

    companion object {
        const val PORT = 8760

        /**
         * The ceiling for a body held in MEMORY -- an image. Generous for a frame.
         */
        const val MAX_BODY = 64 * 1024 * 1024

        /**
         * The ceiling for /swap_video, which streams to disk and is bounded by the cache
         * partition rather than the heap. Still a ceiling: past this it is a mistake, and
         * a clip that long is better fed frame by frame anyway.
         */
        const val MAX_UPLOAD = 512L * 1024 * 1024

        /** Long enough for a big upload over USB, short enough to reap a dead client. */
        const val READ_TIMEOUT_MS = 120_000

        /** A brief wait, so a request arriving during a preview refresh does not bounce. */
        const val ACQUIRE_WAIT_MS = 15_000L

        private fun appVersion(ctx: Context): String = runCatching {
            ctx.packageManager.getPackageInfo(ctx.packageName, 0).versionName ?: "?"
        }.getOrDefault("?")

        /** The address to hand a PC, when the phone is reachable on the LAN. */
        fun lanUrl(ctx: Context): String {
            val ip = NetworkInterface.getNetworkInterfaces().toList()
                .filter { it.isUp && !it.isLoopback }
                .flatMap { it.inetAddresses.toList() }
                .firstOrNull { !it.isLoopbackAddress && it.hostAddress?.contains('.') == true }
                ?.hostAddress
            return "http://" + (ip ?: "127.0.0.1") + ":" + PORT
        }
    }
}

/** Per-request overrides, so a client can change a knob without touching the app. */
private fun SwapOptions.overrides(q: Map<String, String>): SwapOptions {
    if (q.isEmpty()) return this
    fun f(k: String) = q[k]?.toFloatOrNull()
    fun i(k: String) = q[k]?.toIntOrNull()
    fun b(k: String) = q[k]?.let { it == "1" || it.equals("true", true) }
    return copy(
        weight = f("weight")?.coerceIn(0f, 1f) ?: weight,
        maskBlur = f("blur")?.coerceIn(0f, 1f) ?: maskBlur,
        detectorScore = f("detector")?.coerceIn(0f, 1f) ?: detectorScore,
        pixelBoost = i("boost")?.coerceIn(1, 4) ?: pixelBoost,
        largestOnly = b("largest") ?: largestOnly,
        faceEnhance = b("enhancer") ?: faceEnhance,
        enhanceBlend = f("enhance_blend")?.coerceIn(0f, 1f) ?: enhanceBlend,
        outputFps = i("fps")?.coerceIn(0, 240) ?: outputFps,
        swapper = q["swapper"]?.takeIf { it == "hyperswap" || it == "inswapper" } ?: swapper,
    )
}
