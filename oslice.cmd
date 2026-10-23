@echo off
setlocal
"%~dp0.venv\Scripts\python.exe" "%~dp0oslice_cli.py" %*
endlocal
