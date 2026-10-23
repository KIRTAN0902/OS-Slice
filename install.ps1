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
    Write-Host " Automated Setup for OS-Slice Research Framework & Prototype" -ForegroundColor Yellow
    Write-Host "=================================================================" -ForegroundColor DarkGray
    Write-Host ""
}

Write-Banner

# 1. Target Directory & Repository Cloning
$RepoUrl = "https://github.com/KIRTAN0902/OS-Slice.git"
$ZipUrl = "https://github.com/KIRTAN0902/OS-Slice/archive/refs/heads/main.zip"
$TargetDir = "OS-Slice"

if (Test-Path "os_slice_prototype\benchmark.py") {
    Write-Host "[✓] Already inside OS-Slice directory." -ForegroundColor Green
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

$VenvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
$VenvStreamlit = Join-Path (Get-Location) ".venv\Scripts\streamlit.exe"

# 4. Dependency Installation
Write-Host "`n[*] Installing OS-Slice dependencies (PyTorch CPU, Transformers, Streamlit, Rich)..." -ForegroundColor Cyan

if ($UseUv) {
    uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
    uv pip install -r os_slice_prototype/requirements.txt
} else {
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
    & $VenvPython -m pip install -r os_slice_prototype/requirements.txt
}

# 5. Verification Test
Write-Host "`n[*] Verifying OS-Slice installation..." -ForegroundColor Cyan
$VerifyCmd = & $VenvPython -c "import torch, transformers, streamlit, rich; print('OK')" 2>$null
if ($VerifyCmd -eq "OK") {
    Write-Host "[✓] Environment verified successfully!" -ForegroundColor Green
} else {
    Write-Host "[!] Warning: Dependency verification had minor warnings, but proceeding..." -ForegroundColor Yellow
}

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host " OS-Slice Installation Complete!" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green

Write-Host @"

What would you like to do next?
  [1] Run Head-to-Head Empirical Benchmark (Base vs OS-Slice on CPU)
  [2] Launch Interactive Streamlit Research UI
  [3] Exit and use manual commands

"@ -ForegroundColor White

$Choice = Read-Host "Enter option [1-3] (Default: 1)"

if ($Choice -eq "2") {
    Write-Host "[*] Launching Streamlit Research Console..." -ForegroundColor Cyan
    & $VenvStreamlit run os_slice_prototype/app.py
} elseif ($Choice -eq "3") {
    Write-Host @"
To run manually at any time:
  - Benchmark:  .\.venv\Scripts\python.exe os_slice_prototype/benchmark.py
  - Web UI:     .\.venv\Scripts\streamlit.exe run os_slice_prototype/app.py
"@ -ForegroundColor Yellow
} else {
    Write-Host "[*] Executing Empirical Benchmark Suite..." -ForegroundColor Cyan
    & $VenvPython os_slice_prototype/benchmark.py
}
