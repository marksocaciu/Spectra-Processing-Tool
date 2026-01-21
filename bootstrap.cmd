@echo off
setlocal
REM Uses Windows Python launcher if available; falls back to python
where py >nul 2>nul
if %errorlevel%==0 (
  py "%~dp0bootstrap.py" %*
) else (
  python "%~dp0bootstrap.py" %*
)
endlocal
