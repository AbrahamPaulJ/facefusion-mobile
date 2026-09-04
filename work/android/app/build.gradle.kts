import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Which line this build came from.  The content gate is a BRANCH difference -- `dev` has
// none -- and an APK that does not say so can be handed to a store, or trusted in a test,
// by accident.
//
// Derived from the gate's own source file rather than set per branch: this file is then
// byte-identical on both lines, so it never conflicts on merge, and the version can never
// drift out of sync with what is actually compiled in.  Delete ContentGate.kt and the APK
// renames itself.
/**
 * Release signing material, from keystore.properties beside this module's project root.
 *
 * That file and the .jks are gitignored. If they are absent the release build is simply
 * UNSIGNED rather than broken, so a clone can still build without the private key.
 *
 * âš  The keystore is not recoverable. Losing it means never being able to update an
 * installed app again -- Android identifies an app by its signature, so a differently
 * signed build is a different app and forces an uninstall.
 */
val keystoreProps = Properties().apply {
    val f = rootProject.file("keystore.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}

val hasContentGate = file("src/main/java/com/facefusion/mobile/ContentGate.kt").exists()
val variantTag = if (hasContentGate) "" else "-dev"

// `dev` is a SEPARATE APP, not a differently-signed one.  Android identifies an installed
// app by its applicationId; a build that keeps this one and changes only the key cannot be
// installed beside the gated build, it can only refuse to install over it
// (INSTALL_FAILED_UPDATE_INCOMPATIBLE).  A distinct id is what lets both sit on one phone,
// and it also gives dev its own private files dir -- so the two can never share, or
// corrupt, each other's downloaded context binaries.  The price is that dev downloads its
// own ~300 MB tier.
val idSuffix = if (hasContentGate) "" else ".dev"
val appLabel = if (hasContentGate) "FaceFusion" else "FaceFusion Dev"

// The ncnn backend (roadmap 6), on when its staged build is present.
//
// Derived from the tree rather than set by hand, for the reason `hasContentGate` is: a flag
// that has to be remembered is a flag that is wrong in one of the two builds.  ncnn is
// compiled in WSL by the Linux NDK and COPIED here by work/android/stage_ncnn.sh, because
// Gradle and this CMake run on Windows and cannot reliably read a WSL UNC path.
//
// Delete work/android/ncnn/ and this builds exactly the QNN-only app 0.3.0 shipped.
val ncnnDir = file("../ncnn")
val hasNcnn = File(ncnnDir, "lib/libncnn.a").exists()

android {
    namespace = "com.facefusion.mobile"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.facefusion.mobile$idSuffix"
        minSdk = 31                 // SM8750 / HTP v79 is far above this
        targetSdk = 35
        // âš  0.1.1 IS SIGNED WITH A DIFFERENT KEY THAN 0.1.0.  The 0.1.0 keystore was lost,
        // and Android identifies an app by its signature, so this is a DIFFERENT app to
        // every device that already has 0.1.0: it cannot be installed as an update, and
        // anyone upgrading has to uninstall first -- which deletes the downloaded context
        // binaries with the app's private files dir.  Say so in the release notes.
        // 4 = the 0.2.0 HOTFIX (2026-08-30, the v81 "no models" bug).  versionName stays
        // "0.2.0" ON PURPOSE: archivesBaseName below is derived from it, so the release
        // asset keeps the filename the published download link already points at.
        //
        // âš  The CODE still had to go up.  Android compares versionCode, not versionName:
        // reusing 3 would have made the hotfix un-installable over the build it fixes,
        // which is every affected user.  A tag can be reused; a versionCode cannot.
        //
        // The cost is two different APKs both calling themselves 0.2.0, so BugReport now
        // prints the code alongside the name -- that is what tells them apart in a report.
        //
        // 23 = 0.4.15 (2026-08-31): a toast when something is saved, and when the models
        // finish downloading.
        //
        // Both already reported themselves on the status line, which sits below the panes
        // and is one line among several -- and on a save the user is looking at the pane
        // they just saved, not at it. The download is worse: ~300 MB is long enough that
        // the phone has been put down, and the notification was the only thing speaking.
        //
        // The download toast fires only for a download that STARTED here. The watch loop
        // also exits on the first composition of an install that already has its models,
        // and announcing a download to someone who did not ask for one is worse than
        // saying nothing.
        //
        // 22 = 0.4.14 (2026-08-31): the preview decodes its frame instead of asking for it.
        //
        // MediaMetadataRetriever cannot seek exactly on every file, and the bench proved
        // both of its methods out with two clips off the same phone:
        //
        //   getFrameAtTime(OPTION_CLOSEST) is a REQUEST. On a 12.5 s clip with a ~5 s
        //   keyframe interval it returned TWO distinct images for the whole timeline --
        //   every seek from 1.1 s to 4.5 s gave byte-identical pixels in a fresh Bitmap.
        //   On the other clip the same call was exact.
        //
        //   getFrameAtIndex IS exact, and unusable on precisely the files that need it: it
        //   reads METADATA_KEY_VIDEO_FRAME_COUNT itself and throws NumberFormatException:
        //   s == null when the container has none. Computing the count in Kotlin does not
        //   help, because the platform never sees our number. The failing clip has no
        //   count; the working one does.
        //
        // So FrameSeeker decodes: seek to the sync frame at or before the target, decode
        // forward, keep the first frame that reaches it. That is what VideoSwapper already
        // does for a real run, which is the other reason to prefer it -- a preview that
        // disagrees with the output is not a preview. Bounded by a deadline so a
        // pathological file costs one slow seek rather than a pane that never fills.
        //
        // 21 = 0.4.13 (2026-08-31): loading a second target previews it.
        //
        // âš  `targetFile` CANNOT be a state key. Every target is copied to the same path,
        // File(cacheDir, "target.mp4"), and File.equals compares path strings -- so the
        // File for a new video is EQUAL to the File for the old one, the LaunchedEffect
        // keyed on it never re-fires, and the auto-warm that draws the swapped pane never
        // runs for any target after the first. The bench log shows it plainly: loading a
        // second clip produced no `autowarm fired` line at all.
        //
        // Long-standing, and invisible until 0.4.5 because the pane fell back to the
        // PREVIOUS target's swapped frame. Clearing that stale frame turned a wrong preview
        // into an empty one, and an empty one is what finally got reported -- as "preview
        // not loaded on first load", which is really every load after the first.
        //
        // A monotonic counter replaces it: bumped on load and on clear, and it cannot be
        // equal to its predecessor by construction.
        //
        // Also: the frame count falls back to duration x fps when the container does not
        // publish METADATA_KEY_VIDEO_FRAME_COUNT, since without a count getFrameAtIndex has
        // no index to ask for and the keyframe snapping of 0.4.12 stands.
        //
        // 20 = 0.4.12 (2026-08-31): the preview follows the trim handle.
        //
        // MEASURED, after two builds of guessing: seeks to 1646 ms and 3616 ms returned
        // BYTE-IDENTICAL frames in freshly allocated Bitmaps, while 8838 ms -- in the next
        // GOP -- differed. The retriever was snapping to sync frames, so most of a drag
        // previewed the same picture. Nothing upstream was at fault: every seek fired,
        // every refresh started warm, the swap ran and found its 5 faces each time. The
        // frame handed to it was simply the wrong one.
        //
        // getFrameAtTime's OPTION_CLOSEST is a REQUEST, not a contract -- a hardware-backed
        // retriever may ignore it, and this one does. getFrameAtIndex is exact, so the time
        // becomes an index through the file's own frame count, with OPTION_CLOSEST kept as
        // the fallback for containers that do not publish one.
        //
        // âš  The two fixes in 0.4.9 were aimed at this and MISSED: the log shows not one
        // dropped refresh and not one late seek. They stay because both describe real
        // hazards -- a dropped redraw and a MediaMetadataRetriever entered from three
        // coroutines while close() can release it -- but neither was this bug, and neither
        // should be cited as having fixed it.
        //
        // 19 = 0.4.11 (2026-08-31): what each seek PRODUCED, not just that it ran.
        //
        // 0.4.10's log settled the scheduling question and killed both of 0.4.9's theories:
        // every seek fires, every refresh starts, the pipeline is warm throughout. So the
        // pane not changing is downstream of all of it. This logs the frame each seek
        // returned -- identity and three sampled pixels -- and what the swap made of it,
        // because a retriever that snaps to the same keyframe hands back a FRESH Bitmap
        // holding IDENTICAL pixels, which is indistinguishable from a pane that will not
        // repaint until you look at the content.
        //
        // 18 = 0.4.10 (2026-08-31): the preview path says why it did nothing.
        //
        // 0.4.9 guessed at two mechanisms for "the frame does not update on seek" and fixed
        // neither -- it also left the pane stuck on "Preparing preview" on first load. The
        // logcat from that build shows one getFrameAtTime and then silence: no ffqnn init,
        // no second seek. So refreshSwapped is returning early, and NOTHING RECORDS WHICH
        // of its five silent `return`s took it.
        //
        // That is the same defect as the QNN error string this session opened with: a
        // decision made and not written down, and two builds spent guessing because of it.
        // One debug line per refusal, plus the auto-warm effect and the seek job, on a tag
        // nothing else uses: `adb logcat -s ffpreview`.
        //
        // 17 = 0.4.9 (2026-08-31): seeking updates the panes every time.
        //
        // Two independent ways a scrub was lost, both older than this session.
        //
        // refreshSwapped `return`ed when previewBusy and nothing ever asked again, so a
        // redraw requested while a preview swap was in flight was DROPPED -- which is
        // precisely what dragging produces, and the swapped pane then kept the previous
        // position until something else happened to trigger it. It coalesces now.
        //
        // And MediaMetadataRetriever is NOT thread safe. Scrubbing cancels the previous
        // seek and starts another, but cancellation is cooperative and getFrameAtTime is a
        // blocking native call that does not observe it, so the old seek is still inside
        // the retriever when the new one enters. Two concurrent reads return null or the
        // wrong frame; closeTarget could release it under a read outright. One monitor now
        // covers open, close and every read.
        //
        // 16 = 0.4.8 (2026-08-31): the pipeline stops reloading for values it reads per
        // frame, and the frame-rate labels sit on the stops they name.
        //
        // The last of the reload bugs, and the root of all of them. `PreviewEngine` keyed
        // the LOADED pipeline on the whole `SwapOptions`, so every field forced a teardown.
        // But ffpipe reads every Config field once per FRAME -- weight, mask blur, padding,
        // detector and landmarker scores, pixel boost, largest-only, the enhancer's on/off
        // and its blend -- and consumes none of them at init. `gpen` in particular is opened
        // whether or not faceEnhance is set, because the flag decides whether the STAGE
        // RUNS. Turning the enhancer off reloaded ~300 MB of contexts to flip a bool.
        //
        // outputFps was the clearest case: it never reaches the native pipeline at all,
        // VideoSwapper takes it directly, and changing it still reloaded every model.
        //
        // So `Pipeline::updateConfig` + a `setOptions` JNI push the tunables to a loaded
        // pipeline, and the load is keyed on the SWAPPER alone -- the one option that
        // really does select a different model file. The clamping is now one shared helper,
        // because a second copy is how the live path ends up trusting Kotlin.
        //
        // The debounce in previewOptionsChanged stops being load-bearing: its own comment
        // said "a drag would otherwise ask for a full model reload dozens of times a
        // second", and now there is no reload to ask for.
        //
        // 15 = 0.4.7 (2026-08-31): three things asked for while testing 0.4.6.
        //
        // SAVE FRAME on the swapped PREVIEW. The output pane has had one since the video
        // path existed, but it reads frames back out of a FINISHED video -- so pulling one
        // still out of a clip meant swapping the whole clip first, which on the CPU backend
        // is minutes for a picture already on screen.
        //
        // FRAME RATE reaches 5/10/15 now, and is a slider. 24/30/60 could only take a 30 fps
        // clip down to 24 -- a 20% saving against a backend an order of magnitude slower
        // than the NPU -- and on a 24 fps clip nothing qualified, so the control hid itself
        // and offered no reduction at all. VideoSwapper already decimated BEFORE the swap,
        // so the saving was always real; the stops were the part that was not useful.
        //
        // PANE HEADERS are inside the pane. The caption and its buttons sat flush against
        // the pane's outer edge while the image below was clipped to a rounded box, so the
        // text read as belonging to the page rather than to the pane under it.
        //
        // 14 = 0.4.6 (2026-08-31): changing the SOURCE stopped reloading the models too,
        // and the empty target pane says TARGET.
        //
        // 13 fixed the target paths and left the source one, which has the same shape: the
        // warm key was `WarmKey(opts, sourceTag)`, so a new source tore down every context
        // and rebuilt it. No model depends on the source -- only `setSource` does, which
        // analyses one image and keeps a 512-float embedding. The key is now two fields:
        // options decide whether to RELOAD, the source decides whether to RE-APPLY.
        //
        // A failed `setSource` still releases. ffpipe::setSource leaves the previous
        // embedding in place when it finds no face, so a pipeline that survived a failed
        // source change is one that would swap the PREVIOUS person's face.
        //
        // âš  The code goes up even though 0.4.5 was never released: it went to the phone's
        // Downloads, which is where builds leave this machine, and two different binaries
        // calling themselves 0.4.5 is the 0.2.0 ambiguity again.
        //
        // 13 = 0.4.5 (2026-08-31): picking a new target stopped reloading the models.
        //
        // The photo path called invalidatePreview(), which tears the native pipeline down.
        // Nothing about a target is loaded -- the warm key is the options and the SOURCE,
        // and frames go to processFrame one at a time -- so every context was reloaded to
        // show a frame the loaded models could already swap. Invisible on the NPU; seconds
        // of "Loading models" on the CPU backend, which is where it was reported.
        //
        // The video path had the mirror image of the same confusion: it cleared
        // `swappedFrame` but not `preview`, and the pane resolves `swappedFrame ?: preview`,
        // so a new video could show the PREVIOUS target's swapped face. One helper that
        // clears the frames without touching the pipeline fixes both.
        //
        // âš  The auto-warm effect also required `!previewWarm`, which made a warm pipeline
        // the one case it would not redraw for. Both bugs above were hiding that: each path
        // went cold by accident, so the guard was never the thing standing in the way.
        //
        // 12 = 0.4.4 (2026-08-31): TWO field bugs, both from users, both about a phone this
        // project does not own.
        //
        // The manifest declared libcdsprpc.so `required="true"`, which is an INSTALL GATE,
        // not a linker hint -- so 0.4.0 shipped a non-Qualcomm backend to an audience the
        // package manager refused to install it for. A Dimensity Poco read the manifest
        // comment on GitHub and worked out why before anyone here did. That is also the
        // whole reason "no non-Qualcomm phone has ever run the ncnn path": none could.
        //
        // And `ffnn_qnn.cpp`'s error string was sticky AND shadowed the runner's own, so
        // the expected `nsfw_` miss on every tier but v79 poisoned it before any real
        // failure could be reported. An 8 Elite Gen 5 owner was shown "open nsfw_v81.bin"
        // -- a file that is SUPPOSED to be absent -- while the actual `graphExecute`
        // failure was discarded. Two releases went to that device without the error code.
        //
        // 11 = 0.4.3 (2026-08-30): the swapped pane redrawing when you seek AFTER a run,
        // preview panes no longer eating the page scroll, and the Remote API's notification
        // stopping looking like an upload.
        //
        // 10 = 0.4.2 (2026-08-30): the Settings inventory listing EVERY model set rather
        // than only the one in use, rotation no longer destroying the screen, the enhancer
        // promoted to a Processors row, the source picker becoming a pane, and the two
        // switches that were one-way doors -- the runtime card that hid itself once used,
        // and the API's LAN switch that was inert whenever the server was off.
        //
        // 9 = 0.4.1 (2026-08-30): ç®€ä½“ä¸­æ–‡ and ç¹é«”ä¸­æ–‡, and the bug report button reaching
        // the place four releases of documentation said it already was. `Settings > Share
        // bug report` did not exist; the only control was on the Swap screen, and only when
        // the status began with "Failed" -- so a swap that finished and looked wrong, which
        // is the report this project most needs from hardware it does not own, had no
        // button at all.
        //
        // âš  The CODE goes up even though 0.4.0 was published hours ago. The published
        // 0.4.0 and this build are different binaries, and this repo has already paid once
        // for two APKs calling themselves the same version (0.2.0, versionCodes 3 and 4).
        //
        // 8 = 0.4.0 (2026-08-30): the NON-QUALCOMM path, linked and shipped. FF_NCNN is on
        // whenever work/android/ncnn/ is staged, the ncnn model set is selectable from the
        // downloader, and Settings can pin the runtime so the path is testable on a phone
        // that has a Hexagon -- which is the only kind of phone this project owns, and
        // therefore the difference between "written" and "verified".
        //
        // âš  The APK grows from 48.0 MB to 65.7 MB (+17.7). The static libraries are ~168 MB
        // and libffnative.so is 78.3 MB unpacked, so the INSTALLED footprint grows far more
        // than the download does -- jniLibs are stored compressed and extracted at install.
        // Measured, not estimated; the ~98 MB the handoff feared was the archive, not the
        // cost.
        //
        // 7 = 0.3.0 (2026-08-30): the tier FALLBACK -- a tier that loads but will not run
        // now falls back instead of leaving the app unusable -- the gate's failure reason
        // surfaced instead of discarded, and the v81 tier restored, which the fallback is
        // what makes safe. The ffnn runtime seam and the ncnn backend were in the tree but
        // NOT linked (FF_NCNN=OFF): unexercised through the APK, so not in that release.
        //
        // 6 = 0.2.2 (2026-08-30): the CONTENT GATE input range. facefusion feeds nsfw_2
        // [-1,1] and this port fed it [0,1] from the gate's first release, which moves the
        // decision statistic 4.6x its own threshold. The quantised gate is renamed
        // `nsfwq2_` so an app on the new range cannot silently load encodings calibrated
        // for the old one -- it reports the gate missing and offers the download instead.
        //
        // 5 = 0.2.1 (2026-08-30): the v81 tier, the 9.5x face enhancer, the output-file
        // leak, and "update available" in the model inventory.
        //
        // âš  The NAME moves this time, unlike the hotfix. Reusing "0.2.0" was right for 4:
        // it was the same release, refetched at the same link, by the same users. This is
        // not that -- it publishes a new model tier and replaces a hosted model in every
        // existing one. A third binary called 0.2.0 would have made the download link
        // ambiguous for the 47 people who already took the second one, so v0.2.1 is a NEW
        // tag and a NEW asset name, and v0.2.0 keeps pointing at what it always did.
        // archivesBaseName follows versionName, so the filename moves with it.
        versionCode = 43
        versionName = "0.5.0$variantTag"    // "-dev" == NO content gate
        setProperty("archivesBaseName", "facefusion-mobile-$versionName")
        manifestPlaceholders["appLabel"] = appLabel
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake {
                arguments += listOf("-DANDROID_STL=c++_shared")
                if (hasNcnn) {
                    // invariantSeparatorsPath, not absolutePath: CMake reads a Windows
                    // backslash as an escape, so the path arrives mangled and then simply
                    // does not exist -- which surfaces as a missing header, not a bad path.
                    arguments += listOf(
                        "-DFF_NCNN=ON",
                        "-DNCNN_DIR=" + ncnnDir.invariantSeparatorsPath,
                    )
                }
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    ndkVersion = "27.2.12479018"

    // The QNN runtime .so files ship in jniLibs and are dlopen'd by libffnative.so at
    // runtime.  They are NOT exec'd: a process exec'd out of the APK is denied the Hexagon
    // fastrpc device, which is the whole reason the backend is loaded in-process (see the
    // header of ffqnn.cpp).  useLegacyPackaging keeps them as real files on disk, which
    // dlopen by absolute path requires.
    packaging {
        jniLibs {
            useLegacyPackaging = true
            keepDebugSymbols += "**/*.so"
        }
    }

    signingConfigs {
        if (keystoreProps.getProperty("storeFile") != null) {
            create("release") {
                storeFile = rootProject.file(keystoreProps.getProperty("storeFile"))
                storePassword = keystoreProps.getProperty("storePassword")
                keyAlias = keystoreProps.getProperty("keyAlias")
                keyPassword = keystoreProps.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        debug { isMinifyEnabled = false }
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.findByName("release")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation(platform("androidx.compose:compose-bom:2024.10.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
}
