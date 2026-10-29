param(
    # Release tag, e.g. v0.1.3.
    [Parameter(Mandatory = $true)]
    [string]$Tag,
    [string]$Notes = "",
    [string]$Repository = "MuseThinker/Nexllama"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $root "crates\app\tauri.conf.json"
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$version = [string]$config.version

if ($Tag.TrimStart("v") -ne $version) {
    throw "release tag $Tag does not match tauri.conf.json version $version"
}

$bundleDir = Join-Path $root "crates\app\target\release\bundle\nsis"
$full = Join-Path $bundleDir "Nexllama_${version}_x64-setup.exe"
$lite = Join-Path $bundleDir "Nexllama_${version}_x64-lite-setup.exe"

function Get-UpdaterEntry {
    param([string]$Installer, [string]$FileName)
    if (-not (Test-Path -LiteralPath $Installer)) {
        throw "installer not found: $Installer"
    }
    $signaturePath = "$Installer.sig"
    if (-not (Test-Path -LiteralPath $signaturePath)) {
        throw "updater signature not found: $signaturePath (build with TAURI_SIGNING_PRIVATE_KEY set)"
    }
    $signature = (Get-Content -LiteralPath $signaturePath -Raw).Trim()
    if (-not $signature) {
        throw "updater signature is empty: $signaturePath"
    }
    return [ordered]@{
        signature = $signature
        url       = "https://github.com/$Repository/releases/download/$Tag/$FileName"
    }
}

$fullName = Split-Path -Leaf $full
$liteName = Split-Path -Leaf $lite
$manifest = [ordered]@{
    version   = $version
    notes     = $Notes
    pub_date  = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    platforms = [ordered]@{
        "windows-x86_64"      = Get-UpdaterEntry -Installer $full -FileName $fullName
        "windows-x86_64-lite" = Get-UpdaterEntry -Installer $lite -FileName $liteName
    }
}

$output = Join-Path $bundleDir "latest.json"
$json = $manifest | ConvertTo-Json -Depth 6
[IO.File]::WriteAllText($output, $json, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "wrote $output"
Write-Host "version: $version"
Write-Host "full: $fullName"
Write-Host "lite: $liteName"
