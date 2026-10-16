param(
    [ValidateSet("cpu", "vulkan", "cuda")]
    [string]$Backend = "vulkan",
    [switch]$Release,
    [switch]$Native,
    # Leave out the NVIDIA runtime (`cudart64_*`, `cublas64_*`, `cublasLt64_*`),
    # which is 477 MB of the CUDA backend's 643 MB — nearly all of it
    # `cublasLt64_*.dll`. The lite installer ships this way and needs the target
    # machine to already have the CUDA runtime; the full installer bundles it so
    # a machine with only the display driver works.
    [switch]$NoCudaRuntime
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# Isolate build outputs per backend so switching features never leaves stale DLLs
# in a shared target directory.
$targetDir = Join-Path $root "target\backend-$Backend"
$env:CARGO_TARGET_DIR = $targetDir

$cargoArgs = @("build", "-p", "infer-daemon")
if ($Release) { $cargoArgs += "--release" }
$features = @()
if ($Backend -ne "cpu") { $features += $Backend }
if ($Native) { $features += "native" }
if ($features.Count -gt 0) { $cargoArgs += @("--features", ($features -join ",")) }

Write-Host "==> backend: $Backend (features: $(if ($features) { $features -join ',' } else { 'none' }))"
& (Join-Path $PSScriptRoot "build.ps1") @cargoArgs
if ($LASTEXITCODE -ne 0) { throw "cargo build failed for $Backend" }

$profile = if ($Release) { "release" } else { "debug" }
$outDir = Join-Path $targetDir $profile
$src = Join-Path $outDir "infer-daemon.exe"
if (-not (Test-Path -LiteralPath $src)) { throw "daemon binary not found at $src" }

$dstDir = Join-Path $root "crates\app\backends\$Backend"
if (Test-Path -LiteralPath $dstDir) { Remove-Item -Recurse -Force $dstDir }
New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
Copy-Item -LiteralPath $src -Destination (Join-Path $dstDir "infer-daemon.exe") -Force

# llama-common.dll carries common_sampler (speculative verification) and the
# n-gram drafters; mtmd.dll carries the image projector. Both are called at
# runtime, so neither is dead weight.
$runtimeDlls = @("llama.dll", "llama-common.dll", "mtmd.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll")
switch ($Backend) {
    "cuda" { $runtimeDlls += "ggml-cuda.dll" }
    "vulkan" { $runtimeDlls += "ggml-vulkan.dll" }
}

foreach ($name in $runtimeDlls) {
    $lib = Join-Path $outDir $name
    if (Test-Path -LiteralPath $lib) {
        Copy-Item -LiteralPath $lib -Destination (Join-Path $dstDir $name) -Force
    } else {
        Write-Warning "missing runtime library: $name (backend $Backend)"
    }
}

# The CUDA runtime is not part of the display driver — it ships with the CUDA
# Toolkit. A machine with only the NVIDIA driver installed has no `cudart64_*` or
# `cublas*`, so a cuda backend that leans on them dies at load. Bundle the three
# the backend links against so it is self-contained on any NVIDIA machine.
#
# `-NoCudaRuntime` leaves them out for the lite installer, which expects the
# target machine to provide them; it is 477 MB smaller and fails to load the
# cuda backend where they are missing.
if ($Backend -eq "cuda" -and -not $NoCudaRuntime) {
    $cudaBins = @()
    if ($env:CUDA_PATH) { $cudaBins += (Join-Path $env:CUDA_PATH "bin") }
    Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object { $cudaBins += (Join-Path $_.FullName "bin") }

    foreach ($pattern in @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")) {
        $found = $null
        foreach ($bin in $cudaBins) {
            foreach ($sub in @("", "x64")) {
                $dir = Join-Path $bin $sub
                if (-not (Test-Path -LiteralPath $dir)) { continue }
                $found = Get-ChildItem $dir -Filter $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
                if ($found) { break }
            }
            if ($found) { break }
        }
        if ($found) {
            Copy-Item -LiteralPath $found.FullName -Destination (Join-Path $dstDir $found.Name) -Force
            Write-Host "cuda runtime: $($found.Name)"
        } else {
            Write-Warning "CUDA runtime '$pattern' not found; installing the cuda backend will need the CUDA Toolkit on this build machine"
        }
    }
} elseif ($Backend -eq "cuda") {
    Write-Host "cuda runtime not bundled: the target machine must already provide cudart64_*, cublas64_* and cublasLt64_*"
}

foreach ($crt in @("MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll")) {
    $lib = Join-Path $env:WINDIR "System32\$crt"
    if (Test-Path -LiteralPath $lib) {
        Copy-Item -LiteralPath $lib -Destination (Join-Path $dstDir $crt) -Force
    }
}

$count = (Get-ChildItem $dstDir -File).Count
$sizeMb = [math]::Round(((Get-ChildItem $dstDir -File | Measure-Object Length -Sum).Sum / 1MB), 1)
Write-Host "backend ready: $dstDir ($count files, $sizeMb MB)"
