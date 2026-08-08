$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = (Resolve-Path (Join-Path $scriptDir "..")).Path
$distDir = Join-Path $root "dist"
$stagingDir = Join-Path $distDir "ml-addon"
$zipPath = Join-Path $distDir "ml-engine.zip"
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }

# The embedded runtime built by build_pipeline.ps1 is CPython 3.10. Native
# PyTorch wheels are ABI-specific, so packaging from (for example) a local
# 3.11 venv would create an engine the installed application cannot import.
$pythonVersion = & $pythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $pythonVersion.Trim() -ne "3.10") {
  throw "ML add-on must be built with CPython 3.10 (bundled runtime ABI); found '$pythonVersion' via '$pythonExe'."
}

# Some development venvs are created without pip. Bootstrap it rather than
# failing later with a cryptic 'No module named pip' error.
& $pythonExe -m ensurepip --upgrade | Out-Host
if ($LASTEXITCODE -ne 0) {
  throw "Could not bootstrap pip in build venv: $pythonExe"
}

Write-Host "[1/5] Preparing staging directory..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $distDir | Out-Null
if (Test-Path $stagingDir) {
  Remove-Item -LiteralPath $stagingDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stagingDir | Out-Null

Write-Host "[2/5] Installing ML engine packages into staging..." -ForegroundColor Cyan
# Do not install `.[ml] --no-deps`: --no-deps also skips the engine's required
# transitive packages. Resolve the full engine closure, while constraining the
# shared libraries to the same compatibility bands as the light installer.
$constraintsPath = Join-Path $stagingDir "engine-constraints.txt"
@(
  "numpy>=1.19.0,<2.0.0",
  "pillow>=9.0.0,<11.0.0",
  "PyYAML>=6.0.0,<7.0.0"
) | Set-Content -LiteralPath $constraintsPath -Encoding ascii
& $pythonExe -m pip install `
  "torch>=2.0.0,<3.0.0" `
  "torchvision>=0.15.0,<1.0.0" `
  "timm>=0.9.0,<2.0.0" `
  --target $stagingDir `
  --constraint $constraintsPath `
  --only-binary=:all:
if ($LASTEXITCODE -ne 0) {
  throw "pip install failed while building the ML add-on package."
}

Write-Host "[3/5] Validating and cleaning package cache..." -ForegroundColor Cyan
if (-not (Test-Path (Join-Path $stagingDir "torch")) -or -not (Test-Path (Join-Path $stagingDir "torchvision")) -or -not (Test-Path (Join-Path $stagingDir "timm"))) {
  throw "ML add-on staging is missing torch, torchvision, or timm."
}
Get-ChildItem -LiteralPath $stagingDir -Recurse -Force -Directory -Filter "__pycache__" |
  Remove-Item -Recurse -Force
# Keep *.dist-info: torch and third-party extensions can consult installed
# distribution metadata at runtime.
Remove-Item -LiteralPath $constraintsPath -Force

# Import from staging with `-S` so this cannot accidentally pass by resolving
# packages from the build venv. This is the closest local proof that the zip
# will contain a self-sufficient ML engine when merged with the light runtime.
$previousPythonPath = $env:PYTHONPATH
try {
  $env:PYTHONPATH = $stagingDir
  & $pythonExe -S -c "import torch, torchvision, timm; print('ML engine imports OK', torch.__version__)"
  if ($LASTEXITCODE -ne 0) {
    throw "ML add-on staging cannot import torch, torchvision, and timm by itself."
  }
} finally {
  $env:PYTHONPATH = $previousPythonPath
}

Write-Host "[4/5] Compressing add-on archive..." -ForegroundColor Cyan
if (Test-Path $zipPath) {
  Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path (Join-Path $stagingDir "*") -DestinationPath $zipPath -Force

Write-Host "[5/5] Done." -ForegroundColor Green
Write-Host "Created ML add-on package: $zipPath"
