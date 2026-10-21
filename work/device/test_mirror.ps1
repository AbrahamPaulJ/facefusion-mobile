# Issue #18: download one required model through a chosen source, on the device, and
# verify it by SHA256. Deletes that ONE file first so the app has something to fetch.
#
#   $env:FF_ADB = <adb.exe>; $env:FF_ADB_SERIAL = <serial>
#   .\work\device\test_mirror.ps1 [-Source mirror] [-File yoloface_v79.bin]
#
# Waits on the file itself (present AND hashing right), not on a timer. Leaves the source
# set to -Restore afterwards (hf by default).
param([string]$Source = 'mirror', [string]$File = 'yoloface_v79.bin',
      [string]$Restore = 'hf', [int]$TimeoutS = 180)

$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false
$adb    = if ($env:FF_ADB) { $env:FF_ADB } else { 'adb' }
$serial = $env:FF_ADB_SERIAL
if (-not $serial) { Write-Output 'set FF_ADB_SERIAL'; exit 1 }
$pkg = 'com.facefusion.mobile'
$dir = "/sdcard/Android/data/$pkg/files/models"
$f   = "$dir/$File"

$want = (& $adb -s $serial shell "sha256sum $f" | Out-String).Split(' ')[0].Trim()
if ($want.Length -ne 64) { Write-Output "no $File on the device to compare against"; exit 1 }
Write-Output "before: $File $want"

& $adb -s $serial shell "am force-stop $pkg; rm -f $f $f.part; logcat -c"
& $adb -s $serial shell "am start -n $pkg/.MainActivity --es source $Source --es download 1" | Out-Null
$t0 = Get-Date
$got = ''
while (((Get-Date) - $t0).TotalSeconds -lt $TimeoutS) {
    Start-Sleep -Seconds 3
    $got = (& $adb -s $serial shell "[ -f $f ] && sha256sum $f" | Out-String).Split(' ')[0].Trim()
    if ($got.Length -eq 64) { break }
}
$secs = [int]((Get-Date) - $t0).TotalSeconds
if ($got -eq $want) { Write-Output "after : $File $got  OK via $Source in $secs s" }
else { Write-Output "after : '$got'  FAILED via $Source after $secs s" }
& $adb -s $serial shell 'logcat -d' | Select-String -Pattern 'models download from|HTTP [0-9]|checksum|accepts|ffdl|Download' |
    Select-Object -Last 8 | ForEach-Object { "  $_" }
& $adb -s $serial shell "am start -n $pkg/.MainActivity --es source $Restore" | Out-Null
