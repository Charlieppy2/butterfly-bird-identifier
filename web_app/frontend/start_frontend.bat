@echo off
echo Starting React Frontend...
cd /d %~dp0
set "PATH=C:\Program Files\nodejs;%PATH%"
call "C:\Program Files\nodejs\npm.cmd" start
pause

