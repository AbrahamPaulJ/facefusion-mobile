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
 * ⚠ The keystore is not recoverable. Losing it means never being able to update an
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
        // ⚠ 0.1.1 IS SIGNED WITH A DIFFERENT KEY THAN 0.1.0.  The 0.1.0 keystore was lost,
        // and Android identifies an app by its signature, so this is a DIFFERENT app to
        // every device that already has 0.1.0: it cannot be installed as an update, and
        // anyone upgrading has to uninstall first -- which deletes the downloaded context
        // binaries with the app's private files dir.  Say so in the release notes.
        // 4 = the 0.2.0 HOTFIX (2026-08-30, the v81 "no models" bug).  versionName stays
        // "0.2.0" ON PURPOSE: archivesBaseName below is derived from it, so the release
        // asset keeps the filename the published download link already points at.
        //
        // ⚠ The CODE still had to go up.  Android compares versionCode, not versionName:
        // reusing 3 would have made the hotfix un-installable over the build it fixes,
        // which is every affected user.  A tag can be reused; a versionCode cannot.
        //
        // The cost is two different APKs both calling themselves 0.2.0, so BugReport now
        // prints the code alongside the name -- that is what tells them apart in a report.
        //
        // 8 = 0.4.0 (2026-08-30): the NON-QUALCOMM path, linked and shipped. FF_NCNN is on
        // whenever work/android/ncnn/ is staged, the ncnn model set is selectable from the
        // downloader, and Settings can pin the runtime so the path is testable on a phone
        // that has a Hexagon -- which is the only kind of phone this project owns, and
        // therefore the difference between "written" and "verified".
        //
        // ⚠ The APK grows from 48.0 MB to 65.7 MB (+17.7). The static libraries are ~168 MB
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
        // ⚠ The NAME moves this time, unlike the hotfix. Reusing "0.2.0" was right for 4:
        // it was the same release, refetched at the same link, by the same users. This is
        // not that -- it publishes a new model tier and replaces a hosted model in every
        // existing one. A third binary called 0.2.0 would have made the download link
        // ambiguous for the 47 people who already took the second one, so v0.2.1 is a NEW
        // tag and a NEW asset name, and v0.2.0 keeps pointing at what it always did.
        // archivesBaseName follows versionName, so the filename moves with it.
        versionCode = 8
        versionName = "0.4.0$variantTag"    // "-dev" == NO content gate
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
