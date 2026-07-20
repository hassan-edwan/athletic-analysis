@echo off
rem Run swift with the MSVC + Swift toolchain environment on Windows.
rem Usage:  ios\swift-env.bat build | test | run …
rem (Needed until a fresh login shell picks up the installer's env vars;
rem  vcvarsall provides the MSVC CRT link libraries Swift requires.)
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
set "SWIFT_ROOT=%LOCALAPPDATA%\Programs\Swift"
set "SDKROOT=%SWIFT_ROOT%\Platforms\6.3.3\Windows.platform\Developer\SDKs\Windows.sdk"
set "PATH=%SWIFT_ROOT%\Toolchains\6.3.3+Asserts\usr\bin;%SWIFT_ROOT%\Runtimes\6.3.3\usr\bin;%PATH%"
cd /d "%~dp0AthleticAnalysisCore"
swift %*
