# Per-graph device latency, decoded from qnn-net-run's own profiling log.
#
# trap #10: the DSP throttles -- every graph in one $Models list is run back to back in ONE
# thermal session, and the reported number is the MINIMUM over inferences, never a mean.
# trap #43: LOAD is reported separately from EXECUTE; comparing a load-inclusive number to
# a published latency is meaningless.
#
#   .\work\device\measure_latency.ps1 -Models "yoloface,fan2d,arcface,hyperswap"
param([string]$Models = "yoloface,fan2d,arcface,hyperswap", [int]$Runs = 20)

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

foreach ($m in $Models.Split(',')) {
    $exists = & $adb -s $serial shell "[ -f $remote/ctx/${m}_v79.bin ] && echo yes"
    if ($exists -notmatch 'yes') { Write-Output "$m : no context binary on device"; continue }

    & $adb -s $serial shell "rm -rf $remote/out/perf_$m; cd $remote && LD_LIBRARY_PATH=$remote/lib ADSP_LIBRARY_PATH=$remote/dsp ./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/${m}_v79.bin --input_list io/$m/input_list.txt --output_dir out/perf_$m --perf_profile burst --profiling_level basic --num_inferences $Runs" 2>&1 | Out-Null

    $local = "$ff\work\device\prof\$m"
    New-Item -ItemType Directory -Force -Path $local | Out-Null
    & $adb -s $serial pull "$remote/out/perf_$m/qnn-profiling-data_0.log" "$local\prof.log" 2>&1 | Out-Null
    Write-Output "pulled $m"
}
Write-Output "now decode with: wsl ... work/device/decode_prof.sh <model>"
