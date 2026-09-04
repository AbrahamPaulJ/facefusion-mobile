# Interleaved A/B of one env-var VALUE (as opposed to bench_geom.ps1's presence toggle).
#
# Verifies the frame count of every run before believing its timing: a run that dies early
# still prints a per-frame average, and averaging over the frames it managed looks like a
# speed-up rather than the failure it is. A tile sweep reported three sizes at exactly
# 5.75 ms/frame once, which is what prompted this check.
param(
    [string]$Var = "FFPASTETILE",
    [string]$A = "64",
    [string]$B = "0",
    [string]$Bucket = "paste",
    [int]$Rounds = 5,
    [int]$Frames = 12
)

$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false
$adb    = if ($env:FF_ADB) { $env:FF_ADB } else { "adb" }
$serial = $env:FF_ADB_SERIAL
if (-not $serial) {
    $serial = (& $adb devices | Select-String '\sdevice$' |
               ForEach-Object { ($_ -split '\s+')[0] } | Select-Object -First 1)
}
if (-not $serial) { Write-Output "no adb device"; exit 1 }
$r = "/data/local/tmp/ff"

$base = "cd $r && $Var={V} ./ffswap --lib $r/lib --skel $r/dsp --models $r/ctx " +
        "--swapper hyperswap --source $r/raw/source.bgr --sw 1024 --sh 1024 " +
        "--target $r/raw/target12.bgr --tw 1366 --th 720 --frames $Frames --out $r/raw/out.bgr"

$res = @{ $A = @(); $B = @() }
Write-Output "warm-up..."
& $adb -s $serial shell ($base -replace '\{V\}', $A) 2>&1 | Out-Null

for ($i = 1; $i -le $Rounds; $i++) {
    $order = if ($i % 2 -eq 1) { @($A, $B) } else { @($B, $A) }
    foreach ($v in $order) {
        $out = (& $adb -s $serial shell ($base -replace '\{V\}', $v) 2>&1) -join "`n"
        # The frame count first. Only then is the timing worth reading.
        if ($out -notmatch "(\d+) frames, \d+ faces") { Write-Output "  $v : NO REPORT"; continue }
        $got = [int]$matches[1]
        if ($got -ne $Frames) { Write-Output "  $v : only $got frames, DISCARDED"; continue }
        if ($out -match "$Bucket\s+([0-9.]+) ms/frame") {
            $res[$v] += [double]$matches[1]
        }
    }
    Write-Output "round $i done"
}

Write-Output ""
Write-Output "$Var  ($Bucket, ms/frame)"
foreach ($v in @($A, $B)) {
    if ($res[$v].Count -eq 0) { Write-Output ("  {0,-6} no valid runs" -f $v); continue }
    $st = $res[$v] | Measure-Object -Minimum -Maximum -Average
    $all = ($res[$v] | ForEach-Object { "{0:F2}" -f $_ }) -join "/"
    Write-Output ("  {0,-6} min {1,7:F2}  max {2,7:F2}  mean {3,7:F2}   {4}" -f
                  $v, $st.Minimum, $st.Maximum, $st.Average, $all)
}
$ma = ($res[$A] | Measure-Object -Minimum).Minimum
$mb = ($res[$B] | Measure-Object -Minimum).Minimum
$spreadA = (($res[$A] | Measure-Object -Maximum).Maximum - $ma)
$spreadB = (($res[$B] | Measure-Object -Maximum).Maximum - $mb)
$delta = $mb - $ma
Write-Output ""
Write-Output ("delta (min) {0:F2} ms/frame; within-config spread {1:F2} / {2:F2}" -f $delta, $spreadA, $spreadB)
if ([Math]::Abs($delta) -lt [Math]::Max($spreadA, $spreadB)) {
    Write-Output "INCONCLUSIVE: the delta is smaller than the spread inside one configuration."
}
