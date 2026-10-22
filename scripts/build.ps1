param()

$ErrorActionPreference = "Stop"

$CargoArgs = $args

function Find-VsInstall {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $path = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($path) { return $path.Trim() }
    }
    foreach ($p in @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community"
    )) {
        if (Test-Path -LiteralPath (Join-Path $p "VC\Auxiliary\Build\vcvars64.bat")) { return $p }
    }
    throw "Visual Studio C++ build tools not found. Install 'Desktop development with C++'."
}

$vs = Find-VsInstall
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$ninja = Join-Path $vs "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
$llvm = "C:\Program Files\LLVM\bin"
$cmake = "C:\Program Files\CMake\bin"
$cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"

foreach ($d in @($cmake, $ninja, $cargoBin)) {
    if ((Test-Path -LiteralPath $d) -and ($env:Path -notlike "*$d*")) {
        $env:Path = "$d;$env:Path"
    }
}
$env:LIBCLANG_PATH = $llvm
$env:CFLAGS = "/utf-8 /O2"
$env:CXXFLAGS = "/utf-8 /O2"
$env:CL = "/utf-8 /O2"
$env:VSLANG = "1033"

if (-not $CargoArgs -or $CargoArgs.Count -eq 0) { $CargoArgs = @("build") }

# vcvars64 puts Visual Studio's old LLVM in PATH, whose clang 19 headers break
# bindgen. Select the standalone libclang explicitly after vcvars has run, but
# keep its clang out of PATH: CMake subprojects otherwise pick it up and pass
# MSVC-style flags such as /utf-8 to it.
$inner = "call `"$vcvars`" >nul 2>&1" +
    " && set `"CMAKE_GENERATOR=Ninja`"" +
    " && set `"LIBCLANG_PATH=$llvm`"" +
    " && set `"CLANG_PATH=$llvm\clang.exe`"" +
    " && set `"CC=cl`"" +
    " && set `"CXX=cl`"" +
    " && cargo " + ($CargoArgs -join " ")
Write-Host "> $inner"
cmd.exe /c $inner
exit $LASTEXITCODE
