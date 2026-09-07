# Debug build

Use JDK 17, Android SDK platform 35 and build tools 34.0.0. Set `sdk.dir` in the
ignored `local.properties` file, or set `ANDROID_HOME`.

The usual native source build also requires NDK 27.2.12479018, CMake 3.22.1,
Qualcomm QNN headers/runtime and optionally the staged ncnn libraries. These native
SDK inputs are intentionally absent from Git.

For Kotlin/UI-only work, matching prebuilt native libraries can be supplied:

```sh
./gradlew :app:assembleDebug -PprebuiltNativeDir=/absolute/path/to/native-libraries
```

That directory must contain `arm64-v8a/libffnative.so`, `libc++_shared.so` and the
matching `libQnn*.so` runtime libraries. Use trusted binaries matching the native
sources and `NativePipe.kt`; rebuild native code when either changes. Do not copy
AndroidX JNI libraries there: Gradle supplies them through the app dependencies.

For this microphone test build, native libraries come from the upstream signed
`facefusion-mobile-0.9.0-release.apk` (GitHub release `v0.9.0`). Its native sources
match this fork except for comments, and `NativePipe.kt` matches exactly.
The microphone changes are compiled from this checkout's Kotlin sources.

The debug app has application ID `com.facefusion.mobile.debug`, label
`FaceFusion Debug` and version suffix `-mic-debug`, so it installs alongside the
release app with separate app data/model downloads. It uses the local Android
debug signing key. APKs are generated in `app/build/outputs/apk/debug/`.

Device acceptance checks for the new recording option are in
[tests/live-microphone.md](tests/live-microphone.md).
