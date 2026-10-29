param(
    # Release tag, e.g. v0.1.3.
    [Parameter(Mandatory = $true)]
    [string]$Tag,
    [string]$Notes = "",
    [string]$Repository = "MuseThinker/Nexllama",
    # Defaults to the full installer for the version in tauri.conf.json.
    [string]$Installer
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $root "crates\app\tauri.conf.json"
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$version = [string]$config.version

if ($Tag.TrimStart("v") -ne $version) {
    throw "release tag $Tag does not match tauri.conf.json version $version"
}
if (-not $Installer) {
    $Installer = Join-Path $root "crates\app\target\release\bundle\nsis\Nexllama_${version}_x64-setup.exe"
}
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

$fileName = Split-Path -Leaf $Installer
$url = "https://github.com/$Repository/releases/download/$Tag/$fileName"
$manifest = [ordered]@{
    version   = $version
    notes     = $Notes
    pub_date  = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    platforms = [ordered]@{
        "windows-x86_64" = [ordered]@{
            signature = $signature
            url       = $url
        }
    }
}

$output = Join-Path (Split-Path -Parent $Installer) "latest.json"
$json = $manifest | ConvertTo-Json -Depth 6
[IO.File]::WriteAllText($output, $json, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "wrote $output"
Write-Host "version: $version"
Write-Host "url: $url"
