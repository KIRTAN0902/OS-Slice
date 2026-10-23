$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPy = Join-Path $RootDir ".venv\Scripts\python.exe"
$Cli = Join-Path $RootDir "oslice_cli.py"

if (-not (Test-Path $VenvPy)) {
    $VenvPy = (Get-Command python).Source
}

& $VenvPy $Cli @args
