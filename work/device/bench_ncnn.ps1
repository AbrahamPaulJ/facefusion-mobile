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
param([switch]$Accuracy, [int]$Loops = 10)

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
