# Stage the qnn-net-run harness on the phone.  Run from PowerShell -- Git Bash mangles
# /data/... paths (Neodragon trap #14 territory).
#
#   .\work\device\stage_device.ps1              # libs + binary only
#   .\work\device\stage_device.ps1 -Models      # also push every *_v79.bin
#
# The phone can attach twice (USB + wireless), so -s is always passed explicitly.
# adb writes push progress to STDERR on success, so $LASTEXITCODE is the only honest
# status check -- never $ErrorActionPreference (trap #14).
param([switch]$Models)

$ErrorActionPreference = 'Continue'
$adb    = if ($env:FF_ADB) { $env:FF_ADB } else { "adb" }
$qairt  = $env:FF_QAIRT
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
& $adb -s $serial shell "mkdir -p $remote/lib $remote/dsp $remote/ctx $remote/io $remote/out"

function Push-One($local, $dest) {
    & $adb -s $serial push $local $dest 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Output "  FAILED: $local"; return $false }
    Write-Output ("  ok  {0,-42} {1,8:N2} MB" -f (Split-Path $local -Leaf), ((Get-Item $local).Length/1MB))
    return $true
}

Write-Output "== binary =="
Push-One "$qairt\bin\aarch64-android\qnn-net-run" "$remote/qnn-net-run" | Out-Null

Write-Output "== host libs (aarch64-android) =="
foreach ($l in @("libQnnHtp.so","libQnnHtpV79Stub.so","libQnnHtpPrepare.so",
                 "libQnnHtpNetRunExtensions.so","libQnnSystem.so")) {
    Push-One "$qairt\lib\aarch64-android\$l" "$remote/lib/$l" | Out-Null
}

Write-Output "== dsp skel (hexagon-v79) =="
foreach ($l in @("libQnnHtpV79Skel.so","libQnnHtpV79.so")) {
    Push-One "$qairt\lib\hexagon-v79\unsigned\$l" "$remote/dsp/$l" | Out-Null
}

if ($Models) {
    Write-Output "== context binaries =="
    Get-ChildItem "$ff\work\device\*_v79.bin" | ForEach-Object {
        Push-One $_.FullName "$remote/ctx/$($_.Name)" | Out-Null
    }
}

& $adb -s $serial shell "chmod 755 $remote/qnn-net-run"
Write-Output "== on device =="
& $adb -s $serial shell "ls -la $remote/ctx/ 2>/dev/null; df -h /data | tail -1"
