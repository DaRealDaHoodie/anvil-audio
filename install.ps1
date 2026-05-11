# install.ps1 — Anvil Audio self-contained installer for Windows
# Run with: powershell -ExecutionPolicy Bypass -File install.ps1
param(
    [switch]$SkipAceStep
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$VenvDir = ".venv"
$PythonExe = "python"

Write-Host "=== Anvil Audio Installer ===" -ForegroundColor Cyan
Write-Host "Platform: Windows x64"
Write-Host ""

# ── Python version check ──────────────────────────────────────────────────────
$PyVer = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python not found. Install Python 3.12 or 3.13 from https://python.org"
    exit 1
}
$Parts = $PyVer.Split(".")
if ([int]$Parts[0] -lt 3 -or ([int]$Parts[0] -eq 3 -and [int]$Parts[1] -lt 12)) {
    Write-Error "Python 3.12 or 3.13 required (found $PyVer). Download from https://python.org"
    exit 1
}
Write-Host "Python $PyVer — OK"

# ── Virtual environment ───────────────────────────────────────────────────────
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment at $VenvDir..."
    & $PythonExe -m venv $VenvDir
}
$Pip = "$VenvDir\Scripts\pip.exe"
Write-Host "Using venv: $VenvDir"
& $Pip install --upgrade pip --quiet

# ── PyTorch (CUDA) ───────────────────────────────────────────────────────────
Write-Host "Installing PyTorch for Windows CUDA..."
& $Pip install torch torchaudio `
    --index-url https://download.pytorch.org/whl/cu128 --quiet

# ── Anvil Audio ───────────────────────────────────────────────────────────────
Write-Host "Installing anvil-audio..."
& $Pip install -e . --quiet

# ── ACE-Step (optional) ───────────────────────────────────────────────────────
if (-not $SkipAceStep) {
    $Answer = Read-Host "Install ACE-Step for music generation? [y/N]"
    if ($Answer -match "^[Yy]$") {
        Write-Host "Installing nano-vllm (required on Windows)..."
        & $Pip install `
            "git+https://github.com/ace-step/ACE-Step.git#subdirectory=acestep/third_parts/nano-vllm" `
            --quiet

        Write-Host "Installing ACE-Step..."
        & $Pip install `
            "ace-step @ git+https://github.com/ace-step/ACE-Step.git" `
            --quiet
        Write-Host "ACE-Step installed."
    }
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Installation complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Activate your environment:"
Write-Host "  $VenvDir\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Check your setup:"
Write-Host "  anvil setup"
Write-Host ""
Write-Host "Generate audio:"
Write-Host "  anvil generate --model stable-audio-open-1.0 --prompt ""rain on a tin roof"""
