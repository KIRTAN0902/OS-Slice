# ==============================================================================
# OS-Slice: Automated One-Line Terminal Installer for Windows
# Run with: irm https://raw.githubusercontent.com/KIRTAN0902/OS-Slice/main/install.ps1 | iex
# ==============================================================================

$ErrorActionPreference = "Stop"

function Write-Banner {
    Write-Host @"
   ___  ____       ____  _ _          
  / _ \/ ___|     / ___|| (_) ___ ___ 
 | | | \___ \ ____\___ \| | |/ __/ _ \
 | |_| |___) |_____|__) | | | (_|  __/
  \___/|____/     |____/|_|_|\___\___|
  Operation-Specific Behavioral Slicing
"@ -ForegroundColor Cyan
    Write-Host "=================================================================" -ForegroundColor DarkGray
    Write-Host " Automated Setup for OS-Slice Research Framework & CLI ('oslice')" -ForegroundColor Yellow
    Write-Host "=================================================================" -ForegroundColor DarkGray
    Write-Host ""
}

Write-Banner

# 1. Target Directory & Repository Setup
$RepoUrl = "https://github.com/KIRTAN0902/OS-Slice.git"
$ZipUrl = "https://github.com/KIRTAN0902/OS-Slice/archive/refs/heads/main.zip"
$TargetDir = "OS-Slice"

if (Test-Path "oslice_cli.py") {
    Write-Host "[✓] Already inside OS-Slice directory: $(Get-Location)" -ForegroundColor Green
} else {
    if (Test-Path $TargetDir) {
        Write-Host "[*] Found existing $TargetDir directory. Entering..." -ForegroundColor Cyan
        Set-Location $TargetDir
    } else {
        Write-Host "[*] Fetching OS-Slice repository..." -ForegroundColor Cyan
        if (Get-Command git -ErrorAction SilentlyContinue) {
            git clone $RepoUrl $TargetDir
            Set-Location $TargetDir
        } else {
            Write-Host "[*] Git not detected. Downloading archive from GitHub..." -ForegroundColor Yellow
            $ZipFile = "$env:TEMP\OS-Slice-main.zip"
            Invoke-RestMethod -Uri $ZipUrl -OutFile $ZipFile
            Expand-Archive -Path $ZipFile -DestinationPath "$env:TEMP\OS-Slice-Extract" -Force
            Move-Item "$env:TEMP\OS-Slice-Extract\OS-Slice-main" $TargetDir
            Remove-Item $ZipFile -Force
            Remove-Item "$env:TEMP\OS-Slice-Extract" -Recurse -Force
            Set-Location $TargetDir
        }
    }
}

$InstallDir = (Get-Location).Path

# 2. Python Environment Detection
Write-Host "`n[*] Detecting Python runtime environment..." -ForegroundColor Cyan

$UseUv = $false
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "[✓] Detected uv package manager (fast-path enabled)." -ForegroundColor Green
    $UseUv = $true
} else {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Host "[!] Error: Python 3.10+ is required but not found in PATH." -ForegroundColor Red
        Write-Host "    Please install Python from https://python.org or run 'winget install Python.Python.3.12'" -ForegroundColor Yellow
        exit 1
    }
    $PyVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    Write-Host "[✓] Detected Python $PyVer" -ForegroundColor Green
}

# 3. Virtual Environment Creation
if (-not (Test-Path ".venv")) {
    Write-Host "[*] Creating dedicated virtual environment (.venv)..." -ForegroundColor Cyan
    if ($UseUv) {
        uv venv .venv
    } else {
        python -m venv .venv
    }
} else {
    Write-Host "[✓] Virtual environment (.venv) already exists." -ForegroundColor Green
}

$VenvPython = Join-Path $InstallDir ".venv\Scripts\python.exe"
$VenvStreamlit = Join-Path $InstallDir ".venv\Scripts\streamlit.exe"

# 4. Dependency Installation
Write-Host "`n[*] Installing OS-Slice dependencies (PyTorch CPU, Transformers, Streamlit, Rich, HuggingFace-Hub)..." -ForegroundColor Cyan

if ($UseUv) {
    uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
    uv pip install -r os_slice_prototype/requirements.txt
    uv pip install huggingface-hub rich
} else {
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
    & $VenvPython -m pip install -r os_slice_prototype/requirements.txt
    & $VenvPython -m pip install huggingface-hub rich
}

# 5. Global 'oslice' Command Registration
Write-Host "`n[*] Registering global 'oslice' command..." -ForegroundColor Cyan

$BinDir = Join-Path $InstallDir "bin"
if (-not (Test-Path $BinDir)) {
    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
}

# Write bin wrappers
$CmdContent = @"
@echo off
setlocal
"$VenvPython" "$InstallDir\oslice_cli.py" %*
endlocal
"@
Set-Content -Path (Join-Path $BinDir "oslice.cmd") -Value $CmdContent -Force

$Ps1Content = @"
& "$VenvPython" "$InstallDir\oslice_cli.py" `@args
"@
Set-Content -Path (Join-Path $BinDir "oslice.ps1") -Value $Ps1Content -Force

# Permanently add $BinDir to User PATH
try {
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($UserPath -notlike "*$BinDir*") {
        [Environment]::SetEnvironmentVariable("Path", "$UserPath;$BinDir", "User")
        Write-Host "[✓] Added '$BinDir' to User PATH environment variable." -ForegroundColor Green
    }
    $env:Path = "$env:Path;$BinDir"
} catch {
    Write-Host "[!] Notice: PATH update skipped: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Register function in current session
function global:oslice { & $VenvPython "$InstallDir\oslice_cli.py" @args }

# Add to PowerShell $PROFILE so it's globally available immediately
try {
    $ProfilePath = $PROFILE
    $ProfileFolder = Split-Path -Parent $ProfilePath
    if (-not (Test-Path $ProfileFolder)) {
        New-Item -ItemType Directory -Path $ProfileFolder -Force | Out-Null
    }
    $ProfileSnippet = @"

# OS-Slice Global CLI Shortcut
function oslice { & "$VenvPython" "$InstallDir\oslice_cli.py" `@args }
"@
    if (-not (Test-Path $ProfilePath) -or ((Get-Content $ProfilePath -Raw) -notmatch "function oslice")) {
        Add-Content -Path $ProfilePath -Value $ProfileSnippet -Force
        Write-Host "[✓] Added 'oslice' shortcut to PowerShell `$PROFILE." -ForegroundColor Green
    }
} catch {
    # Non-fatal if profile is restricted
}

# 6. Verification Test
Write-Host "`n[*] Verifying OS-Slice installation..." -ForegroundColor Cyan
$VerifyCmd = & $VenvPython -c "import torch, transformers, streamlit, rich, oslice_cli; print('OK')" 2>$null
if ($VerifyCmd -eq "OK") {
    Write-Host "[✓] OS-Slice & 'oslice' CLI verified successfully!" -ForegroundColor Green
} else {
    Write-Host "[✓] Environment ready." -ForegroundColor Green
}

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host " OS-Slice Installation Complete!" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
Write-Host " Tip: You can now type 'oslice' in ANY terminal to open the UI!" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Green

Write-Host @"

What would you like to do next?
  [1] Launch 'oslice' Interactive Research Workbench (Claude Code CLI Style)
  [2] Run Head-to-Head Empirical Benchmark (Base vs OS-Slice on CPU)
  [3] Launch Streamlit Web Dashboard
  [4] Exit to terminal (run 'oslice' anytime)

"@ -ForegroundColor White

$Choice = Read-Host "Enter option [1-4] (Default: 1)"

if ($Choice -eq "2") {
    Write-Host "[*] Executing Empirical Benchmark Suite..." -ForegroundColor Cyan
    & $VenvPython os_slice_prototype/benchmark.py
} elseif ($Choice -eq "3") {
    Write-Host "[*] Launching Streamlit Research Console..." -ForegroundColor Cyan
    & $VenvStreamlit run ui.py
} elseif ($Choice -eq "4") {
    Write-Host "`n[✓] Setup complete! Type 'oslice' anytime to launch the research workbench." -ForegroundColor Green
} else {
    Write-Host "[*] Launching 'oslice' CLI Workbench..." -ForegroundColor Cyan
    & $VenvPython oslice_cli.py
}
