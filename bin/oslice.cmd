@echo off
setlocal
set "ROOT_DIR=%~dp0.."
"%ROOT_DIR%\.venv\Scripts\python.exe" "%ROOT_DIR%\oslice_cli.py" %*
endlocal
