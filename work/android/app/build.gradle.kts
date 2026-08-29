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

android {
    namespace = "com.facefusion.mobile"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.facefusion.mobile"
        minSdk = 31                 // SM8750 / HTP v79 is far above this
        targetSdk = 35
        // ⚠ 0.1.1 IS SIGNED WITH A DIFFERENT KEY THAN 0.1.0.  The 0.1.0 keystore was lost,
        // and Android identifies an app by its signature, so this is a DIFFERENT app to
        // every device that already has 0.1.0: it cannot be installed as an update, and
        // anyone upgrading has to uninstall first -- which deletes the downloaded context
        // binaries with the app's private files dir.  Say so in the release notes.
        versionCode = 2
        versionName = "0.1.1$variantTag"    // "-dev" == NO content gate
        setProperty("archivesBaseName", "facefusion-mobile-$versionName")
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake { arguments += listOf("-DANDROID_STL=c++_shared") } }
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
