# Install the APK and push the models into its files dir.
#
# The models are NOT in the APK: ~266 MB, and three of the five are GPL-3.0 or
# non-commercial (docs/model-audit.md), so they are pushed separately.
#
# ORDER MATTERS.  The models directory must be created by the APP, not by adb: a directory
# created by `adb push` is owned by `shell`, and the app cannot traverse it -- open() then
# fails with nothing but ENOENT to explain it.  So this installs, launches the app once so
# MainActivity.modelDir() mkdirs it, and only then pushes.
#
# The models are named `<name>_<tier>.bin`, where the tier is the HTP arch the app
# measures for itself at startup (v68 / v73 / v79).  -Tier auto asks the device the same
# question the app will ask, through the on-device CLI, so the files pushed are the files
# the app will look for.  It falls back to v79 when ffswap is not staged.
#
# -Dev installs the UNGATED build instead.  It is a different applicationId, so it is a
# different app to the phone: it installs BESIDE the gated one rather than over it, and it
# gets its OWN files dir -- which is why -Dev seeds that dir from the gated app's copy
# before pushing anything.  A cp on the device costs nothing; re-pushing ~300 MB to arrive
# at byte-identical files costs a transfer that has silently truncated before.
#
#   .\work\device\install_app.ps1              # install + models for the measured tier
#   .\work\device\install_app.ps1 -NoModels    # APK only
#   .\work\device\install_app.ps1 -Tier v68    # force a tier, to test a lower one
#   .\work\device\install_app.ps1 -Dev         # the ungated build, alongside the gated one
param([switch]$NoModels, [switch]$Dev, [string]$Swapper = "hyperswap", [string]$Tier = "auto")

$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false
$adb    = if ($env:FF_ADB) { $env:FF_ADB } else { "adb" }
$ff     = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$serial = $env:FF_ADB_SERIAL
if (-not $serial) {
    # Whatever adb can currently see. Set FF_ADB_SERIAL to pin one when several
    # are attached -- a phone on both USB and TCP shows up twice.
    $serial = (& $adb devices | Select-String '\sdevice$' |
               ForEach-Object { ($_ -split '\s+')[0] } | Select-Object -First 1)
}
if (-not $serial) { Write-Output "no adb device; connect one or set FF_ADB_SERIAL"; exit 1 }
# Must match build.gradle.kts, which derives the id from whether ContentGate.kt exists.
$pkg    = if ($Dev) { "com.facefusion.mobile.dev" } else { "com.facefusion.mobile" }
$files  = "/sdcard/Android/data/$pkg/files"

& $adb connect $serial | Out-Null

# Both build types, newest first, filtered to the line asked for.  The variant is in the
# FILENAME (`-dev-`) because build.gradle.kts puts it there, so this cannot install a gated
# APK under the dev package or the reverse -- which would look like it worked and leave an
# ungated app wearing the gated app's name.
$apk = Get-ChildItem "$ff\work\android\app\build\outputs\apk\*\*.apk" -ErrorAction SilentlyContinue |
       Where-Object { ($_.Name -like '*-dev-*') -eq [bool]$Dev } |
       Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $apk) {
    $what = if ($Dev) { "a -dev APK (build it on the dev branch)" } else { "an APK" }
    Write-Output "no $what under app/build/outputs/apk -- run gradlew assembleRelease"; exit 1
}
Write-Output ("apk {0:N1} MB  {1}  -> {2}" -f ($apk.Length / 1MB), $apk.Name, $pkg)

& $adb -s $serial install -r -g $apk.FullName 2>&1 | Out-String -Stream | Select-Object -Last 2

if ($Tier -eq "auto") {
    $remote = "/data/local/tmp/ff"
    # -join: adb returns an ARRAY of lines, and `-match` on an array filters it instead of
    # populating $Matches -- which silently left $Tier as the literal string "auto".
    $probe = (& $adb -s $serial shell "[ -x $remote/ffswap ] && cd $remote && ./ffswap --lib $remote/lib --skel $remote/dsp --probe 2>/dev/null | grep '^tier'") -join "`n"
    if ($probe -match 'tier\s*:\s*(v\d+)') {
        $Tier = $Matches[1]
        Write-Output "measured tier: $Tier"
    } else {
        $Tier = "v79"
        Write-Output "tier probe unavailable (stage ffswap to enable it) -- assuming $Tier"
    }
}

if (-not $NoModels) {
    # The models dir must be created by the APP, not by adb (trap #13), so make sure it
    # exists -- but do NOT wipe it. This used to `rm -rf` first, which turns every install
    # into a 266 MB transfer; over Tailscale on mobile data that cost ~190 MB and then died
    # mid-push, leaving a truncated hyperswap the app could not load. Each file is now
    # compared by hash and pushed only if it differs (trap #19).
    & $adb -s $serial shell "am start -n $pkg/.MainActivity" | Out-Null
    Start-Sleep -Seconds 3
    & $adb -s $serial shell "am force-stop $pkg"

    $owner = & $adb -s $serial shell "ls -ld $files/models 2>/dev/null"
    Write-Output "models dir: $owner"

    if ($Dev) {
        # Seed from the gated app, which is almost always already on the phone with a tier
        # downloaded.  -n so an existing dev copy is never overwritten, and every file is
        # hash-checked by the loop below regardless: this is an optimisation, not trust.
        $src = "/sdcard/Android/data/com.facefusion.mobile/files/models"
        $n = ((& $adb -s $serial shell "ls $src/*.bin 2>/dev/null | wc -l") -join "").Trim()
        if ($n -match '^[0-9]+$' -and [int]$n -gt 0) {
            Write-Output "seeding from the gated app ($n files, on-device copy)"
            & $adb -s $serial shell "cp -n $src/*.bin $files/models/ 2>/dev/null"
        }
    }

    # The gate context ships on BOTH lines and is mandatory on both: it blocks on main, and
    # even with -Dev `ffpipe::init` opens it with the rest and fails hard when it is absent.
    # fp32 `nsfw` is the shipping build but only finalizes on v79; below that the quantised
    # `nsfwq` is the only one that exists, so push whichever this tier has.
    $models = @("yoloface", "fan2d", "arcface", $Swapper)
    if (Test-Path "$ff\work\device\nsfw_$Tier.bin") { $models += "nsfw" } else { $models += "nsfwq" }

    foreach ($m in $models) {
        $p = "$ff\work\device\${m}_$Tier.bin"
        if (-not (Test-Path $p)) { Write-Output "MISSING $p"; continue }
        $dst = "$files/models/${m}_$Tier.bin"

        $want = (Get-FileHash $p -Algorithm SHA256).Hash.ToLower()
        $have = (& $adb -s $serial shell "sha256sum $dst 2>/dev/null | cut -d' ' -f1") -join ""
        if ($have.Trim() -eq $want) {
            Write-Output ("{0,-10} up to date ({1:N1} MB, not pushed)" -f $m, ((Get-Item $p).Length / 1MB))
            continue
        }

        # A stale copy is deleted BEFORE the push, so an interrupted transfer leaves a
        # missing file rather than a half-written one that still passes a size check.
        & $adb -s $serial shell "rm -f $dst"
        Write-Output ("{0,-10} pushing {1:N1} MB ..." -f $m, ((Get-Item $p).Length / 1MB))
        & $adb -s $serial push $p $dst 2>&1 | Out-String -Stream | Select-Object -Last 1

        $after = (& $adb -s $serial shell "sha256sum $dst 2>/dev/null | cut -d' ' -f1") -join ""
        if ($after.Trim() -ne $want) {
            Write-Output "  HASH MISMATCH after push -- $dst is not $p"
        }
    }
}
Write-Output "== on device =="
& $adb -s $serial shell "ls -la $files/models/ 2>&1"
