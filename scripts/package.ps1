param(
    # Which installers to produce.
    #
    #   full  cpu + cuda with NVIDIA's runtime bundled (~486 MB). Works on any
    #         machine: an NVIDIA card runs on the GPU with no CUDA Toolkit to
    #         install, and everything else falls back to the CPU.
    #   lite  cpu + cuda without that runtime (~180 MB). Needs the target machine
    #         to already have the CUDA runtime, and is otherwise the same app.
    #   both  one after the other, from a single daemon build.
    [ValidateSet("full", "lite", "both")]
    [string]$Edition = "full",
    # Defaults to cpu + cuda, plus vulkan when the Vulkan SDK is present, so one
    # build covers NVIDIA, AMD/Intel and CPU wherever it can.
    [string[]]$Backends,
    # Off by default: a shipped backend must run on any CPU, so it is built
    # against a baseline, not the machine that happened to build it. `GGML_NATIVE`
    # (the `native` feature) bakes the build host's instruction set in, which
    # crashes elsewhere. Pass -Native only for a personal, same-machine build.
    [switch]$Native,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
if ($env:Path -notlike "*$cargoBin*") { $env:Path = "$cargoBin;$env:Path" }

# The vulkan backend is compiled by glslc, which ships with the LunarG Vulkan
# SDK. Absent it, the build would fail, so it is only added when the SDK is here.
function Test-VulkanSdk {
    if ($env:VULKAN_SDK -and (Test-Path -LiteralPath $env:VULKAN_SDK)) { return $true }
    if (Get-Command glslc -ErrorAction SilentlyContinue) { return $true }
    if (Test-Path -LiteralPath 'C:\VulkanSDK') { return $true }
    return $false
}

if (-not $Backends -or $Backends.Count -eq 0) {
    $Backends = @("cpu", "cuda")
    if (Test-VulkanSdk) {
        $Backends += "vulkan"
    } else {
        Write-Host "note: no Vulkan SDK found (glslc/VULKAN_SDK); skipping the vulkan backend - AMD/Intel GPUs will use CPU unless it is fetched at runtime"
    }
}

$cudaDir = Join-Path $root "crates\app\backends\cuda"
$bundleDir = Join-Path $root "crates\app\target\release\bundle\nsis"

# The NVIDIA runtime the lite edition leaves out. Named by pattern because the
# toolkit's version suffix moves; `cublasLt64_*.dll` is 428 MB on its own, which
# is what makes the difference between the two installers worth offering.
$cudaRuntime = @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")

function Test-CudaRuntimeStaged {
    if (-not (Test-Path -LiteralPath $cudaDir)) { return $false }
    foreach ($pattern in $cudaRuntime) {
        if (-not (Get-ChildItem $cudaDir -Filter $pattern -ErrorAction SilentlyContinue)) {
            return $false
        }
    }
    return $true
}

function Invoke-TauriBuild {
    $tauri = Join-Path $root "frontend\node_modules\.bin\tauri.cmd"
    if (-not (Test-Path -LiteralPath $tauri)) {
        throw "Tauri CLI not found. Run: npm --prefix frontend install"
    }
    # build-daemon.ps1 exports CARGO_TARGET_DIR per backend so the backends do not
    # share a target directory. The app must not inherit it: the bundle would land
    # under `target/backend-cuda/release/bundle` instead of the app's own target,
    # where nothing looks for it. Cleared here, per build, because restaging a
    # backend inside the edition loop sets it again.
    Remove-Item Env:\CARGO_TARGET_DIR -ErrorAction SilentlyContinue
    Push-Location (Join-Path $root "crates\app")
    try {
        & $tauri build
        if ($LASTEXITCODE -ne 0) { throw "tauri build failed" }
    } finally {
        Pop-Location
    }
}

# Tauri names the installer after the product, so the two editions cannot both
# keep the generated name. The suffix is added to the file the user downloads;
# both installers keep the same product name, identifier and install location, so
# installing one over the other replaces it rather than sitting beside it.
#
# `-Since` is when this edition's build started: an installer left over from an
# earlier run has an older timestamp, and picking "the newest" alone would be a
# guess on a machine that also built one ten minutes ago.
function Publish-Installer {
    param([string]$Edition, [datetime]$Since)
    if (-not (Test-Path -LiteralPath $bundleDir)) {
        throw "no installer produced in $bundleDir"
    }
    $built = Get-ChildItem $bundleDir -Filter *.exe |
        Where-Object { $_.LastWriteTime -ge $Since } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $built) { throw "no installer produced in $bundleDir by this build" }
    $suffix = if ($Edition -eq "lite") { "-lite" } else { "" }
    $name = $built.Name -replace '-setup\.exe$', "$suffix-setup.exe"
    if ($name -ne $built.Name) {
        if (Test-Path -LiteralPath (Join-Path $bundleDir $name)) {
            Remove-Item -LiteralPath (Join-Path $bundleDir $name) -Force
        }
        Move-Item -LiteralPath $built.FullName -Destination (Join-Path $bundleDir $name) -Force
    }
    $final = Get-Item (Join-Path $bundleDir $name)
    Write-Host "installer: $($final.FullName) ($([math]::Round($final.Length/1MB,2)) MiB)"
}

# One daemon build per backend. The lite edition only changes what is *staged*
# beside it, so both editions come out of this same build.
foreach ($backend in $Backends) {
    $daemonArgs = @{ Backend = $backend }
    if ($Native) { $daemonArgs.Native = $true }
    if (-not $Debug) { $daemonArgs.Release = $true }
    & (Join-Path $PSScriptRoot "build-daemon.ps1") @daemonArgs
    if ($LASTEXITCODE -ne 0) { throw "daemon build failed for $backend" }
}

# build-daemon.ps1 sets CARGO_TARGET_DIR per backend; clear it so the desktop app
# builds into its own workspace target directory.
Remove-Item Env:\CARGO_TARGET_DIR -ErrorAction SilentlyContinue
$env:CARGO_TARGET_DIR = $null

# Lite first, then full. The installer's file name is generated from the product
# name, so both editions want the same path: publishing lite moves its file to a
# `-lite` name, which frees the plain one for the full build to write into. Doing
# it the other way round would have the second makensis overwrite the first
# installer, and the first would be gone by the time it was renamed.
$editions = if ($Edition -eq "both") { @("lite", "full") } else { @($Edition) }

foreach ($want in $editions) {
    if ($Backends -contains "cuda") {
        if ($want -eq "lite") {
            # Leave out the runtime the target machine is expected to have.
            foreach ($pattern in $cudaRuntime) {
                Get-ChildItem $cudaDir -Filter $pattern -ErrorAction SilentlyContinue |
                    Remove-Item -Force
            }
            Write-Host "== edition: lite (cuda runtime not bundled)"
        } elseif (-not (Test-CudaRuntimeStaged)) {
            # A previous lite staging removed it; put it back, or this "full"
            # build would quietly ship as a lite one.
            Write-Host "== edition: full (restaging the cuda runtime)"
            & (Join-Path $PSScriptRoot "build-daemon.ps1") -Backend cuda -Release:(-not $Debug) -Native:$Native
            if ($LASTEXITCODE -ne 0) { throw "daemon restage failed for cuda" }
        } else {
            Write-Host "== edition: full"
        }
    } else {
        Write-Host "== edition: $want (no cuda backend in this build)"
    }

    $started = Get-Date
    Invoke-TauriBuild
    Publish-Installer -Edition $want -Since $started
}
