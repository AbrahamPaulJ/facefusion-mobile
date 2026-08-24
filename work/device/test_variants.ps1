# Run several context-binary variants of one model against the SAME held-out io/ set.
#
# trap #10: the DSP throttles, so variants that will be COMPARED must run interleaved in
# one thermal session -- which is what this does.
#
#   .\work\device\test_variants.ps1 arcface arcface,arcfacemin,arcfacepc
param([Parameter(Mandatory=$true)][string]$IoName,
      [Parameter(Mandatory=$true)][string]$Variants)

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

# Always re-push the held-out set.  Trusting whatever is already in io/<name> produced a
# silently wrong A/B once, because run_e2e_device.py had replaced it with its own frames.
& $adb -s $serial shell "rm -rf $remote/io/$IoName; mkdir -p $remote/io/$IoName"
& $adb -s $serial push "$ff\work\device\io\$IoName\in" "$remote/io/$IoName/in" 2>&1 | Out-String -Stream | Select-Object -Last 1 | Out-Null
& $adb -s $serial push "$ff\work\device\io\$IoName\input_list.txt" "$remote/io/$IoName/input_list.txt" 2>&1 | Out-String -Stream | Select-Object -Last 1 | Out-Null

foreach ($v in $Variants.Split(',')) {
    $bin = "$ff\work\device\${v}_v79.bin"
    if (-not (Test-Path $bin)) { Write-Output "skip $v (no $bin)"; continue }
    & $adb -s $serial push $bin "$remote/ctx/${v}_v79.bin" 2>&1 | Out-String -Stream | Select-Object -Last 1 | Out-Null
    & $adb -s $serial shell "sh $remote/run_model.sh $IoName acc $v" 2>&1 | Select-String -Pattern "Finished|ERROR|error" | Select-Object -First 2
    if (Test-Path "$ff\work\device\out\$v") { Remove-Item -Recurse -Force "$ff\work\device\out\$v" }
    & $adb -s $serial pull "$remote/out/$v" "$ff\work\device\out\$v" 2>&1 | Out-String -Stream | Select-Object -Last 1 | Out-Null
    Write-Output "ran $v"
}
