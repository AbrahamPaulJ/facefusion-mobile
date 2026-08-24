# Push I/O for one model, run it on the HTP, pull the outputs back.
#   .\work\device\run_device.ps1 arcface           # accuracy pass
#   .\work\device\run_device.ps1 arcface perf      # latency pass
param([Parameter(Mandatory=$true)][string]$Name, [string]$Mode = "acc")

$ErrorActionPreference = 'Continue'
# adb reports progress on stderr even when it succeeds -- trap #14
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

Write-Output "== push io/$Name =="
& $adb -s $serial shell "rm -rf $remote/io/$Name; mkdir -p $remote/io/$Name"
& $adb -s $serial push "$ff\work\device\io\$Name\in" "$remote/io/$Name/in" 2>&1 | Out-String -Stream | Select-Object -Last 1
& $adb -s $serial push "$ff\work\device\io\$Name\input_list.txt" "$remote/io/$Name/input_list.txt" 2>&1 | Out-String -Stream | Select-Object -Last 1
& $adb -s $serial push "$ff\work\device\run_model.sh" "$remote/run_model.sh" 2>&1 | Out-String -Stream | Select-Object -Last 1

# context binary may have been built after the last stage
if (Test-Path "$ff\work\device\${Name}_v79.bin") {
    & $adb -s $serial push "$ff\work\device\${Name}_v79.bin" "$remote/ctx/${Name}_v79.bin" 2>&1 | Out-String -Stream | Select-Object -Last 1
}

Write-Output "== run ($Mode) =="
& $adb -s $serial shell "sh $remote/run_model.sh $Name $Mode"

if ($Mode -eq "acc") {
    Write-Output "== pull =="
    $local = "$ff\work\device\out"
    New-Item -ItemType Directory -Force -Path $local | Out-Null
    if (Test-Path "$local\$Name") { Remove-Item -Recurse -Force "$local\$Name" }
    & $adb -s $serial pull "$remote/out/$Name" "$local\$Name" 2>&1 | Out-String -Stream | Select-Object -Last 1
    Write-Output "pulled to work\device\out\$Name"
}
