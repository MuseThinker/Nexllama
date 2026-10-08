param(
    [Parameter(Mandatory = $true)]
    [string]$Version,

    [string]$BaseUrl = "",
    [string]$LlamaCpp = "",
    [string]$BackendsDir = "",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $BackendsDir) { $BackendsDir = Join-Path $root "crates\app\backends" }
if (-not $OutDir) { $OutDir = Join-Path $root "dist\backends" }
if (-not $BaseUrl) { $BaseUrl = "https://github.com/OWNER/REPO/releases/download/v$Version" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$backends = @{}
foreach ($dir in Get-ChildItem $BackendsDir -Directory) {
    $name = $dir.Name
    $exe = Join-Path $dir.FullName "infer-daemon.exe"
    if (-not (Test-Path -LiteralPath $exe)) {
        Write-Warning "skipping $name (no infer-daemon.exe)"
        continue
    }

    $zipName = "nexllama-backend-$name-$Version.zip"
    $zipPath = Join-Path $OutDir $zipName
    if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }

    # Zip the directory *contents* so files sit at the archive root.
    Compress-Archive -Path (Join-Path $dir.FullName "*") -DestinationPath $zipPath -Force

    $hash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLower()
    $backends[$name] = @{ url = "$BaseUrl/$zipName"; sha256 = $hash }
    Write-Host ("packed {0,-8} -> {1} ({2:N1} MB, sha256 {3})" -f $name, $zipName, ((Get-Item $zipPath).Length / 1MB), $hash.Substring(0, 12))
}

$manifest = [ordered]@{
    version   = $Version
    llama_cpp = $LlamaCpp
    backends  = $backends
}
$manifestPath = Join-Path $OutDir "manifest.json"
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host "manifest: $manifestPath"
Write-Host "upload manifest.json + the .zip files to: $BaseUrl"
