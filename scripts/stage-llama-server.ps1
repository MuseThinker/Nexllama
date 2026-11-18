param(
    # Which backend folder to stage into.
    [Parameter(Mandatory = $true)]
    [ValidateSet("cpu", "cuda", "vulkan")]
    [string]$Backend,
    # Folder holding llama-server.exe and the DLLs it loads (a llama.cpp build's
    # `bin` directory). Not in the repository: llama-server is built from the
    # llama.cpp source tree at the commit the crate pins.
    [Parameter(Mandatory = $true)]
    [string]$From
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path -LiteralPath (Join-Path $From "llama-server.exe"))) {
    throw "no llama-server.exe in $From"
}

$dst = Join-Path $root "crates\app\backends\$Backend\server"
if (-not (Test-Path -LiteralPath (Join-Path $root "crates\app\backends\$Backend\infer-daemon.exe"))) {
    throw "backend '$Backend' is not staged yet: run scripts/build-daemon.ps1 -Backend $Backend -Release first"
}

if (Test-Path -LiteralPath $dst) { Remove-Item -LiteralPath $dst -Recurse -Force }
New-Item -ItemType Directory -Force -Path $dst | Out-Null

# llama-server is built with shared libraries, so the exe is a thin launcher:
# it needs its own llama/ggml DLLs, and they must not be mixed with the daemon's
# copies (different CMake runs, same names). A `server` subfolder keeps them
# apart; the daemon adds the parent folder to PATH for the CUDA runtime.
Copy-Item -Path (Join-Path $From "llama-server*") -Destination $dst -Force
Copy-Item -Path (Join-Path $From "llama*.dll") -Destination $dst -Force -ErrorAction SilentlyContinue
Copy-Item -Path (Join-Path $From "ggml*.dll") -Destination $dst -Force -ErrorAction SilentlyContinue
Copy-Item -Path (Join-Path $From "mtmd*.dll") -Destination $dst -Force -ErrorAction SilentlyContinue

$files = Get-ChildItem -LiteralPath $dst -File
$size = [math]::Round((($files | Measure-Object Length -Sum).Sum / 1MB), 1)
Write-Host ("server: staged {0} files, {1} MiB into crates\app\backends\{2}\server" -f $files.Count, $size, $Backend)

# The version string says which llama.cpp this is; it should be the commit the
# crate pins, or the server and the daemon are not the same engine.
# (llama-server writes this to stderr, which PowerShell treats as an error.)
$ErrorActionPreference = "Continue"
$version = & (Join-Path $dst "llama-server.exe") --version 2>&1 | Select-Object -First 1
$ErrorActionPreference = "Stop"
Write-Host "server: $version"
