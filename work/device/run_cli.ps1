# Push and run the native CLI on the phone, headless.
#
# Works with the screen locked and over Tailscale -- nothing in this path touches the
# display.  The binary lives in /data/local/tmp, where an executable IS allowed to reach
# the DSP; trap #33 only bites binaries launched out of an APK.
param([string]$Swapper = "hyperswap", [int]$Frames = 12, [switch]$SkipPush)

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
$remote = "/data/local/tmp/ff"

& $adb connect $serial | Out-Null

& $adb -s $serial push "$ff\work\device\ffswap" "$remote/ffswap" 2>&1 | Out-String -Stream | Select-Object -Last 1
& $adb -s $serial shell "chmod 755 $remote/ffswap"

if (-not $SkipPush) {
    & $adb -s $serial shell "mkdir -p $remote/raw"
    foreach ($f in @("source.bgr", "target12.bgr")) {
        & $adb -s $serial push "$ff\work\device\raw\$f" "$remote/raw/$f" 2>&1 | Out-String -Stream | Select-Object -Last 1
    }
}

Write-Output "== run =="
& $adb -s $serial shell "cd $remote && ./ffswap --lib $remote/lib --skel $remote/dsp --models $remote/ctx --swapper $Swapper --source $remote/raw/source.bgr --sw 1024 --sh 1024 --target $remote/raw/target12.bgr --tw 1366 --th 720 --frames $Frames --out $remote/raw/out.bgr 2>&1"

Write-Output "== pull =="
& $adb -s $serial pull "$remote/raw/out.bgr" "$ff\work\device\raw\out_$Swapper.bgr" 2>&1 | Out-String -Stream | Select-Object -Last 1
