@echo off

echo.
echo "Download and unzip ICU 67.1 for Windows x64 MSVC2017"

mkdir icu
cscript //Nologo helpers/demo.cmd.wget.js https://github.com/unicode-org/icu/releases/download/release-67-1/icu4c-67_1-Win64-MSVC2017.zip %cd%\icu\icu4c-67_1-Win64-MSVC2017.zip
cscript //Nologo helpers/demo.cmd.unzip.vbs %cd%\icu %cd%\icu\icu4c-67_1-Win64-MSVC2017.zip
del /f %cd%\icu\icu4c-67_1-Win64-MSVC2017.zip
copy %cd%\icu\bin64\icuuc67.dll .
copy %cd%\icu\bin64\icudt67.dll .
