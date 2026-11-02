param(
    # llama.cpp source tree (the same revision as the backend DLLs).
    [Parameter(Mandatory = $true)][string]$LlamaSource,
    # CMake install tree containing include\ and lib\llama.lib, mtmd.lib, ggml*.lib.
    [Parameter(Mandatory = $true)][string]$LlamaLib,
    # CMake build tree, which holds common\llama-common-base.lib.
    [Parameter(Mandatory = $true)][string]$LlamaBuild,
    [string]$Out = (Join-Path $PSScriptRoot '..\crates\app\tts\llama-tts-worker.exe')
)

$ErrorActionPreference = "Stop"
$src = Join-Path $PSScriptRoot '..\crates\app\tts\llama-tts-worker.cpp'
$outPath = [IO.Path]::GetFullPath($Out)

$args = @(
    '/nologo', '/MD', '/Brepro', '/W0', '/utf-8', '/O2', '/DNDEBUG', '/std:c++17', '/EHsc',
    '/DGGML_BACKEND_SHARED', '/DGGML_SHARED', '/DGGML_USE_CPU', '/DLLAMA_SHARED',
    '/DLLAMA_SUBPROCESS', '/D_CRT_SECURE_NO_WARNINGS',
    "/I$LlamaSource\include", "/I$LlamaSource\ggml\include", "/I$LlamaSource\common",
    "/I$LlamaSource\vendor", "/I$LlamaSource\tools\mtmd", "/I$LlamaLib\include",
    $src, "/Fe:$outPath", '/link', '/SUBSYSTEM:CONSOLE', '/INCREMENTAL:NO',
    (Join-Path $LlamaBuild 'common\llama-common.lib'),
    (Join-Path $LlamaLib 'lib\mtmd.lib'),
    (Join-Path $LlamaBuild 'common\llama-common-base.lib'),
    (Join-Path $LlamaLib 'lib\llama.lib'),
    (Join-Path $LlamaLib 'lib\ggml.lib'),
    (Join-Path $LlamaLib 'lib\ggml-cpu.lib'),
    (Join-Path $LlamaLib 'lib\ggml-base.lib'),
    'kernel32.lib', 'user32.lib', 'gdi32.lib', 'winspool.lib', 'shell32.lib',
    'ole32.lib', 'oleaut32.lib', 'uuid.lib', 'comdlg32.lib', 'advapi32.lib'
)

$quoted = $args | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
cmd /c ('"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul && cl.exe ' + ($quoted -join ' '))
if ($LASTEXITCODE -ne 0) { throw "cl failed with $LASTEXITCODE" }
Write-Host "built $outPath"
