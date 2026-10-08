param(
    [string[]]$Backends = @("cpu", "cuda"),
    [switch]$NoNative,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
if ($env:Path -notlike "*$cargoBin*") { $env:Path = "$cargoBin;$env:Path" }

foreach ($backend in $Backends) {
    $daemonArgs = @{ Backend = $backend }
    if (-not $NoNative) { $daemonArgs.Native = $true }
    if (-not $Debug) { $daemonArgs.Release = $true }
    & (Join-Path $PSScriptRoot "build-daemon.ps1") @daemonArgs
    if ($LASTEXITCODE -ne 0) { throw "daemon build failed for $backend" }
}

# build-daemon.ps1 sets CARGO_TARGET_DIR per backend; clear it so the desktop app
# builds into its own workspace target directory.
Remove-Item Env:\CARGO_TARGET_DIR -ErrorAction SilentlyContinue
$env:CARGO_TARGET_DIR = $null

$tauri = Join-Path $root "frontend\node_modules\.bin\tauri.cmd"
if (-not (Test-Path -LiteralPath $tauri)) {
    throw "Tauri CLI not found. Run: npm --prefix frontend install"
}

Push-Location (Join-Path $root "crates\app")
try {
    & $tauri build
    if ($LASTEXITCODE -ne 0) { throw "tauri build failed" }
} finally {
    Pop-Location
}

$bundle = Join-Path $root "crates\app\target\release\bundle\nsis"
if (Test-Path -LiteralPath $bundle) {
    Get-ChildItem $bundle -Filter *.exe | ForEach-Object {
        Write-Host "installer: $($_.FullName) ($([math]::Round($_.Length/1MB,2)) MiB)"
    }
}
