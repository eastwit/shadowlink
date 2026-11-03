# Create a local virtual environment and install ShadowLink in editable mode.
# Usage: .\scripts\bootstrap_dev.ps1 [-WithHamer]
param(
    [switch]$WithHamer
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$python = $env:PYTHON
if (-not $python) {
    $python = "python"
}

& $python -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"

if (-not (Test-Path "third_party/hamer/hamer")) {
    Write-Warning "third_party/hamer is not checked out. Run: git submodule update --init"
}

& $python -m venv .venv
$venvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip

if ($WithHamer) {
    & $venvPython -m pip install -e ".[dev,hamer]"
    if (Test-Path "third_party/hamer/hamer") {
        & $venvPython -m pip install -e third_party/hamer --no-deps
    } else {
        throw "Install the submodule first: git submodule update --init"
    }
} else {
    & $venvPython -m pip install -e ".[dev]"
}

Write-Host ""
Write-Host "Installed. Next steps:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  shadowlink setup-models"
Write-Host "  # register at https://mano.is.tue.mpg.de and place MANO_RIGHT.pkl"
Write-Host "  shadowlink doctor --strict-models"
