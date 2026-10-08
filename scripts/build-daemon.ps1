param(
    [ValidateSet("cpu", "vulkan", "cuda")]
    [string]$Backend = "vulkan",
    [switch]$Release,
    [switch]$Native
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

foreach ($crt in @("MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll")) {
    $lib = Join-Path $env:WINDIR "System32\$crt"
    if (Test-Path -LiteralPath $lib) {
        Copy-Item -LiteralPath $lib -Destination (Join-Path $dstDir $crt) -Force
    }
}

$count = (Get-ChildItem $dstDir -File).Count
$sizeMb = [math]::Round(((Get-ChildItem $dstDir -File | Measure-Object Length -Sum).Sum / 1MB), 1)
Write-Host "backend ready: $dstDir ($count files, $sizeMb MB)"
