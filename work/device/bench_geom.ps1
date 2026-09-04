# Interleaved A/B of the CPU geometry buckets, headless.
#
# ONE binary, two configurations selected by an env var, alternating order every round --
# because running A first every round biases B by ~20% on this phone, which is larger than
# most of the effects this bench exists to measure. Reports min-of-N per bucket; the
# warm-up round is discarded.
#
#   .\bench_geom.ps1                       # cached vs uncached box mask
#   .\bench_geom.ps1 -Rounds 5 -Frames 24
param(
    [string]$Swapper = "hyperswap",
    [int]$Frames = 12,
    [int]$Rounds = 3,
    [string]$EnvVar = "FFNOMASKCACHE",
    [string]$LabelA = "cached",
    [string]$LabelB = "uncached",
    [switch]$Push
)

$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false
$adb    = if ($env:FF_ADB) { $env:FF_ADB } else { "adb" }
$ff     = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$serial = $env:FF_ADB_SERIAL
if (-not $serial) {
    $serial = (& $adb devices | Select-String '\sdevice$' |
               ForEach-Object { ($_ -split '\s+')[0] } | Select-Object -First 1)
}
if (-not $serial) { Write-Output "no adb device; connect one or set FF_ADB_SERIAL"; exit 1 }
$remote = "/data/local/tmp/ff"

if ($Push) {
    & $adb -s $serial push "$ff\work\device\ffswap" "$remote/ffswap" 2>&1 | Out-String -Stream | Select-Object -Last 1
    & $adb -s $serial shell "chmod 755 $remote/ffswap"
}

# The env prefix goes on FFSWAP, not on the `cd`: "VAR=1 cd x && ./y" sets the variable
# for the cd and runs ./y with a clean environment, which silently benchmarks one build
# against itself and reports the difference as noise. It did exactly that once.
$base = "cd $remote && {ENV}./ffswap --lib $remote/lib --skel $remote/dsp --models $remote/ctx " +
        "--swapper $Swapper --source $remote/raw/source.bgr --sw 1024 --sh 1024 " +
        "--target $remote/raw/target12.bgr --tw 1366 --th 720 --frames $Frames " +
        "--out $remote/raw/out.bgr"

# bucket -> label -> list of per-frame ms
$res = @{}
function Record($label, $text) {
    foreach ($line in ($text -split "`n")) {
        if ($line -match '^\s+(detprep|warp|tensor|mask|paste)\s+([0-9.]+) ms/frame') {
            $k = $matches[1]; $v = [double]$matches[2]
            if (-not $res.ContainsKey($k)) { $res[$k] = @{} }
            if (-not $res[$k].ContainsKey($label)) { $res[$k][$label] = @() }
            $res[$k][$label] += $v
        }
        if ($line -match '^\s+geometry\s+([0-9.]+) ms') {
            $k = 'TOTAL'; $v = [double]$matches[1] / $Frames
            if (-not $res.ContainsKey($k)) { $res[$k] = @{} }
            if (-not $res[$k].ContainsKey($label)) { $res[$k][$label] = @() }
            $res[$k][$label] += $v
        }
    }
}

Write-Output "warm-up..."
& $adb -s $serial shell ($base -replace '\{ENV\}', '') 2>&1 | Out-Null

for ($r = 1; $r -le $Rounds; $r++) {
    # Alternate which configuration goes first, every round.
    $order = if ($r % 2 -eq 1) { @($LabelA, $LabelB) } else { @($LabelB, $LabelA) }
    foreach ($lab in $order) {
        $prefix = if ($lab -eq $LabelB) { "$EnvVar=1 " } else { "" }
        $cmd = $base -replace '\{ENV\}', $prefix
        Write-Output "round $r : $lab"
        $out = (& $adb -s $serial shell "$cmd" 2>&1) -join "`n"
        Record $lab $out
    }
}

Write-Output ""
Write-Output "== min-of-$Rounds, ms/frame =="
Write-Output ("{0,-10} {1,10} {2,10} {3,10}" -f "bucket", $LabelA, $LabelB, "delta")
foreach ($k in @('detprep','warp','tensor','mask','paste','TOTAL')) {
    if (-not $res.ContainsKey($k)) { continue }
    $a = if ($res[$k].ContainsKey($LabelA)) { ($res[$k][$LabelA] | Measure-Object -Minimum).Minimum } else { [double]::NaN }
    $b = if ($res[$k].ContainsKey($LabelB)) { ($res[$k][$LabelB] | Measure-Object -Minimum).Minimum } else { [double]::NaN }
    $ra = if ($res[$k].ContainsKey($LabelA)) { ($res[$k][$LabelA] | ForEach-Object { "{0:F2}" -f $_ }) -join "/" } else { "" }
    $rb = if ($res[$k].ContainsKey($LabelB)) { ($res[$k][$LabelB] | ForEach-Object { "{0:F2}" -f $_ }) -join "/" } else { "" }
    Write-Output ("{0,-10} {1,10:F2} {2,10:F2} {3,10:F2}   {4}  vs  {5}" -f $k, $a, $b, ($b - $a), $ra, $rb)
}
