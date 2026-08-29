# ncnn on this device: latency for CPU and Vulkan, and fp16 accuracy.  Roadmap section 6.
#
# Two tools, because one cannot do both jobs:
#   benchncnn   latency.  Runs from a .param alone on random weights (DataReaderFromEmpty),
#               so timing costs no weight transfer -- and tells you nothing about accuracy.
#   ncnn_run    accuracy.  Real weights, a deterministic seeded input, output dumped as
#               fp32 so the modes are directly comparable.  work/device/ncnn_run.cpp.
#
# ⚠ -Mode vulkan uses num_threads=1.  That is deliberate and it is the control: a silent
# per-layer CPU fallback would then show up as the 1-thread CPU timing rather than hiding
# inside an 8-thread one.  Vulkan beating CPU-1t by 1.9-4.6x is how we know the GPU is
# actually doing the work.
#
# ⚠ Compare latencies only WITHIN one run of this script (trap #10).  cooling_down=1 is on.
#
# Build (WSL):
#   ~/ncnn/build-android-vulkan  -- cmake with -DNCNN_VULKAN=ON -DNCNN_BUILD_BENCHMARK=ON,
#   then `make benchncnn ncnn_run`.  ncnn_run.cpp is dropped into ncnn/benchmark/ and added
#   to its CMakeLists, so it inherits benchncnn's Vulkan link line.
# Convert (WSL):
#   pnnx <model>.onnx "inputshape=[...]" -- prebuilt, and it takes ONNX directly, which
#   avoids onnx2ncnn's protobuf dependency (no root on this box).
#
#   .\work\device\bench_ncnn.ps1                 # latency, CPU + Vulkan + the 1t control
#   .\work\device\bench_ncnn.ps1 -Accuracy       # fp16 vs fp32 on hyperswap
#   .\work\device\bench_ncnn.ps1 -Contention     # CPU cost per inference, and throttling
param([switch]$Accuracy, [switch]$Contention, [int]$Loops = 10)

$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false
$adb = if ($env:FF_ADB) { $env:FF_ADB } else { "adb" }
$serial = $env:FF_ADB_SERIAL
if (-not $serial) {
    # -s always: this phone attaches twice, USB and Tailscale (trap #14).  USB first, it is
    # the one that carried 800 MB without dropping.
    $serial = (& $adb devices | Select-String '\sdevice$' |
               ForEach-Object { ($_ -split '\s+')[0] } |
               Sort-Object { $_ -match ':' } | Select-Object -First 1)
}
if (-not $serial) { Write-Output "no adb device"; exit 1 }
$R = "/data/local/tmp/ffncnn"

# name -> ncnn Mat shape, which is w,h,c and NOT NCHW.  hyperswap takes source[512] BEFORE
# target[256x256x3]; reversing them converts, runs, and measures nothing.
$models = [ordered]@{
    "yoloface_8n_b1"        = "[640,640,3]"
    "2dfan4_heatmaps"       = "[256,256,3]"
    "arcface_w600k_r50_b1"  = "[112,112,3]"
    "hyperswap_1a_256_fp32" = "[512],[256,256,3]"
}

if ($Contention) {
    # Why the wall-clock table is not the whole answer.  Two backends can tie on ms/frame
    # and still differ completely in what they leave the CPU free to do -- and in whether
    # they hold that speed for the 300 frames of a 10 s clip.
    #
    # DIFFERENTIAL, two loop counts.  A single run's CPU time is dominated by setup: Vulkan
    # spends ~7.6 s compiling shaders and uploading 402 MB before the first inference, which
    # made a naive measurement read ~0.77 cores instead of 0.06.  Subtracting the small run
    # from the large one cancels setup exactly.
    #
    # cooling_down=0 on purpose: the throttling IS the measurement here.
    $small = 20; $large = 120
    foreach ($cfg in @(
        @{ label = "VULKAN"; gpu = 0; threads = 1 },
        @{ label = "CPU-8t"; gpu = -1; threads = 8 })) {
        $stats = @{}
        foreach ($n in @($small, $large)) {
            # The subshell's stderr is redirected, not benchncnn's: `time` is a shell
            # builtin and writes to the SHELL's stderr, so a redirect inside the pipeline
            # captures the timings of nothing.  benchncnn also reports on stderr.
            $raw = & $adb -s $serial shell "cd $R && ( time ./benchncnn $n $($cfg.threads) 0 $($cfg.gpu) 0 param=hyperswap_1a_256_fp32.ncnn.param shape=[512],[256,256,3] ) 2>&1 | grep -E 'min =|real|user|sys'"
            $t = ($raw -join ' ')
            # toybox `time` prints  0m11.38s real  0m05.01s user  0m03.76s system
            $sec = [regex]::Matches($t, '(\d+)m([\d.]+)s') | ForEach-Object {
                [double]$_.Groups[1].Value * 60 + [double]$_.Groups[2].Value }
            $avg = if ($t -match 'avg\s*=\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
            $mx  = if ($t -match 'max\s*=\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
            $stats[$n] = @{ real = $sec[0]; cpu = $sec[1] + $sec[2]; avg = $avg; max = $mx }
            Write-Output ("{0,-7} n={1,-4} avg {2,7:N2} ms  max {3,7:N2} ms  real {4,6:N2}s  cpu {5,7:N2}s" -f
                          $cfg.label, $n, $avg, $mx, $sec[0], ($sec[1] + $sec[2]))
        }
        $dn   = $large - $small
        $wall = ($stats[$large].real - $stats[$small].real) / $dn * 1000
        $cpu  = ($stats[$large].cpu  - $stats[$small].cpu)  / $dn * 1000
        Write-Output ("  -> per inference: {0,7:N1} ms wall   {1,8:N1} ms CPU   {2,5:N2} cores busy" -f
                      $wall, $cpu, ($cpu / $wall))
        Write-Output ("  -> sustained:     avg {0,6:N1} -> {1,6:N1}   max {2,6:N1} -> {3,6:N1}" -f
                      $stats[$small].avg, $stats[$large].avg, $stats[$small].max, $stats[$large].max)
        Write-Output ""
    }
    Write-Output "Expect: Vulkan a fraction of one core and flat under load; CPU-8t all eight"
    Write-Output "cores and degrading badly.  The ABSOLUTE numbers move a lot with how hot the"
    Write-Output "phone already is, and the CPU path is what moves -- 2026-08-30 measured 0.06"
    Write-Output "cores / +52% worst frame cold, and 0.13 cores / +142% on an already-warm"
    Write-Output "device.  Read the gap between the two rows, not the digits (trap #10)."
    return
}

if (-not $Accuracy) {
    foreach ($cfg in @(
        @{ label = "CPU  8 threads"; gpu = -1; threads = 8 },
        @{ label = "CPU  1 thread (fallback control)"; gpu = -1; threads = 1 },
        @{ label = "VULKAN"; gpu = 0; threads = 1 })) {
        Write-Output ""
        Write-Output ("###### " + $cfg.label + " ######")
        foreach ($m in $models.Keys) {
            $line = & $adb -s $serial shell "cd $R && ./benchncnn $Loops $($cfg.threads) 0 $($cfg.gpu) 1 param=$m.ncnn.param shape=$($models[$m]) 2>&1 | grep 'min ='"
            Write-Output ("  {0,-24} {1}" -f $m, (($line -join ' ').Trim()))
        }
    }
} else {
    # hyperswap only: it is 47% of the frame budget and the one that was fp32-demoted on
    # HTP, so it is where a precision problem would appear first.
    #   _fp32 = pnnx default (fp16 WEIGHTS), _w32 = pnnx fp16=0 (fp32 weights)
    # The pair isolates storage from arithmetic; the .ncnn.param files are byte-identical.
    Write-Output "running hyperswap in four configurations ..."
    $runs = @(
        @{ out = "w32_fp32";   param = "hyperswap_w32";        mode = "fp32"   },
        @{ out = "fp32";       param = "hyperswap_1a_256_fp32"; mode = "fp32"   },
        @{ out = "fp16";       param = "hyperswap_1a_256_fp32"; mode = "fp16"   },
        @{ out = "vulkan";     param = "hyperswap_1a_256_fp32"; mode = "vulkan" },
        @{ out = "w32_vulkan"; param = "hyperswap_w32";        mode = "vulkan" }
    )
    foreach ($r in $runs) {
        $o = & $adb -s $serial shell "cd $R && ./ncnn_run $($r.param).ncnn.param $($r.param).ncnn.bin $($r.mode) out_$($r.out).raw 512 256x256x3 2>&1 | grep -v Adreno"
        Write-Output ("  {0,-12} {1}" -f $r.out, (($o -join ' ').Trim()))
    }
    Write-Output ""
    Write-Output "pull with:  adb -s $serial pull $R/out_<name>.raw"
    Write-Output "score against out_w32_fp32.raw (fp32 weights AND fp32 arithmetic)."
}
