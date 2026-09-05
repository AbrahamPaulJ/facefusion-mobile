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
    [switch]$Push,
    # CPU affinity mask, hex, for `taskset`.  "c0" = cpu6,7, the two prime cores on SM8750
    # (4.47 GHz; cpu0-5 are 3.53 GHz).  "" leaves the scheduler alone.
    #
    # ⚠ USE IT.  Unpinned, this bench is BIMODAL, not noisy: paste returns ~8 ms or ~13-14
    # ms depending on which core type the process landed on, giving a 6.34 ms
    # within-configuration spread against effects worth well under 1 ms.  No number of
    # rounds fixes that -- averaging two clock domains just estimates the mix.  Pinned, the
    # spread collapses and sub-millisecond effects become resolvable.
    #
    # ⚠ Android's toybox `taskset` takes a HEX MASK, not `-c 6,7`; the Linux `-c` form
    # fails with "Unknown option 'c'".  Environment variables DO pass through it (verified,
    # not assumed -- taskset is a real binary that execs, unlike the `cd` builtin that ate
    # the env prefix twice on this project).
    [string]$Pin = "c0",
    # Rounds excluded from PAIRED at the START of the run.  The single warm-up run is not
    # enough: pinned and charging, this phone stays in a boosted state for about two full
    # rounds -- TOTAL read 6.99 and 8.38 there against ~14 and ~17 for every round after,
    # and EVERY bucket was low together, so it is the clock, not the code.  Two boosted
    # rounds left in the average moved the whole-file NEON estimate by more than a
    # millisecond.  They are still printed, just not averaged.
    [int]$Discard = 2
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
# against itself and reports the difference as noise. It did exactly that once -- and then
# again in 2026-09-05, by hand over `adb shell`, where this comment is not.
#
# `taskset` sits BETWEEN the env prefix and ffswap, which is fine: it is a real binary that
# execs, so it inherits the variables and passes them on. Only shell builtins swallow them.
$pin = if ($Pin) { "taskset $Pin " } else { "" }
if ($Pin) { Write-Output "pinned to cpu mask 0x$Pin" } else { Write-Output "UNPINNED -- expect a bimodal spread; see the -Pin comment" }
$base = "cd $remote && {ENV}$pin./ffswap --lib $remote/lib --skel $remote/dsp --models $remote/ctx " +
        "--swapper $Swapper --source $remote/raw/source.bgr --sw 1024 --sh 1024 " +
        "--target $remote/raw/target12.bgr --tw 1366 --th 720 --frames $Frames " +
        "--out $remote/raw/out.bgr"

# bucket -> label -> round -> per-frame ms.  KEYED BY ROUND, not appended to a flat list.
#
# ⚠ A flat list pairs by INDEX, and that silently produces a wrong answer the moment any
# single run fails to report: 20 rounds once returned 15 values for one label and 16 for
# the other, so every pair after the first drop compared a cool run against a hot one.
# The paste bucket drifts 4.99 -> 14.77 across a long run, so a one-slot misalignment is
# worth ~10 ms -- twenty times the effect being measured, and it came out as a confident
# negative number with no indication anything had gone wrong.  Rounds missing either label
# are now DROPPED from the pairing and counted out loud.
$res = @{}
$dropped = @()
function Record($label, $round, $text) {
    # A run that dies early still prints a per-frame average, so believe a report only
    # after the frame count confirms it ran to the end.  This is the check bench_tile.ps1
    # already had and this script did not.
    $lines = $text -split "`n"
    $ran = $false
    foreach ($line in $lines) {
        if ($line -match '^(\d+) frames, \d+ faces') { if ([int]$matches[1] -eq $Frames) { $ran = $true } }
    }
    if (-not $ran) {
        Write-Output "    !! no complete report ($Frames frames) -- round $round/$label DROPPED"
        return $false
    }
    foreach ($line in $lines) {
        if ($line -match '^\s+(detprep|warp|tensor|mask|paste)\s+([0-9.]+) ms/frame') {
            $k = $matches[1]; $v = [double]$matches[2]
            if (-not $res.ContainsKey($k)) { $res[$k] = @{} }
            if (-not $res[$k].ContainsKey($label)) { $res[$k][$label] = @{} }
            $res[$k][$label][$round] = $v
        }
        if ($line -match '^\s+geometry\s+([0-9.]+) ms') {
            $k = 'TOTAL'; $v = [double]$matches[1] / $Frames
            if (-not $res.ContainsKey($k)) { $res[$k] = @{} }
            if (-not $res[$k].ContainsKey($label)) { $res[$k][$label] = @{} }
            $res[$k][$label][$round] = $v
        }
    }
    return $true
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
        if (-not (Record $lab $r $out)) { $dropped += "$r/$lab" }
    }
}

Write-Output ""
Write-Output "== $Rounds rounds, ms/frame =="
Write-Output "PAIRED is the headline: the mean of (B - A) taken WITHIN each round, which"
Write-Output "cancels the monotone thermal drift across the run. min-of-N does not --"
Write-Output "the coolest slot is always round 1's first run, so it flatters whichever"
Write-Output "label goes first, and it once reported 3.24 where the paired estimate was 1.65."
if ($Discard -gt 0) { Write-Output "PAIRED excludes the first $Discard round(s): this phone stays boosted that long." }
Write-Output "spreadA is the WITHIN-configuration range of label A -- the honest noise floor."
Write-Output "A PAIRED smaller than spreadA is not a result, however consistent its sign looks."
if ($dropped.Count) {
    Write-Output ""
    Write-Output "!! $($dropped.Count) run(s) produced no complete report and were dropped: $($dropped -join ', ')"
    Write-Output "!! Their ROUNDS are excluded from PAIRED entirely -- a half-round cannot be paired."
}
Write-Output ("{0,-10} {1,10} {2,10} {3,10} {4,10} {5,7} {6,8}" -f "bucket", $LabelA, $LabelB, "min-delta", "PAIRED", "pairs", "spreadA")
foreach ($k in @('detprep','warp','tensor','mask','paste','TOTAL')) {
    if (-not $res.ContainsKey($k)) { continue }
    $va = if ($res[$k].ContainsKey($LabelA)) { @($res[$k][$LabelA].Keys | Sort-Object | ForEach-Object { $res[$k][$LabelA][$_] }) } else { @() }
    $vb = if ($res[$k].ContainsKey($LabelB)) { @($res[$k][$LabelB].Keys | Sort-Object | ForEach-Object { $res[$k][$LabelB][$_] }) } else { @() }
    $a = if ($va.Count) { ($va | Measure-Object -Minimum).Minimum } else { [double]::NaN }
    $b = if ($vb.Count) { ($vb | Measure-Object -Minimum).Minimum } else { [double]::NaN }
    $ra = ($va | ForEach-Object { "{0:F2}" -f $_ }) -join "/"
    $rb = ($vb | ForEach-Object { "{0:F2}" -f $_ }) -join "/"
    # Pair the two runs of each round BY ROUND NUMBER -- they sit adjacent in time, so a
    # drift that is linear over the round cancels when both orders are averaged together.
    # A round that lost either run contributes nothing rather than shifting everything
    # after it by one slot.
    $pairs = @()
    if ($res[$k].ContainsKey($LabelA) -and $res[$k].ContainsKey($LabelB)) {
        foreach ($rr in ($res[$k][$LabelA].Keys | Sort-Object)) {
            if ($rr -le $Discard) { continue }
            if ($res[$k][$LabelB].ContainsKey($rr)) {
                $pairs += ($res[$k][$LabelB][$rr] - $res[$k][$LabelA][$rr])
            }
        }
    }
    $paired = if ($pairs.Count) { ($pairs | Measure-Object -Average).Average } else { [double]::NaN }
    # The within-configuration spread is the honest noise floor: a PAIRED smaller than this
    # is not a result, however confident its sign looks.
    $spread = if ($va.Count -gt 1) { ($va | Measure-Object -Maximum).Maximum - $a } else { [double]::NaN }
    Write-Output ("{0,-10} {1,10:F2} {2,10:F2} {3,10:F2} {4,10:F2} {5,7} {6,8:F2}   {7}  vs  {8}" -f `
                  $k, $a, $b, ($b - $a), $paired, $pairs.Count, $spread, $ra, $rb)
}
