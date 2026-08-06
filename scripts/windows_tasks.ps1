param(
  [Parameter(Position = 0)]
  [string]$Task = "help"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

function Invoke-Uv {
  param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
  )

  & uv @Args
  if ($LASTEXITCODE -ne 0) {
    throw "uv command failed: uv $($Args -join ' ')"
  }
}

function Invoke-Checked {
  param(
    [string]$FilePath,
    [string[]]$Arguments = @()
  )

  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed: $FilePath $($Arguments -join ' ')"
  }
}

function Ensure-Venv {
  $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
  if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment with Python 3.10..." -ForegroundColor Cyan
    Invoke-Uv "venv" "--python" "3.10"
  }
}

function Ensure-Cargo {
  $cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
  $cargoExe = Join-Path $cargoBin "cargo.exe"

  if ((-not (Get-Command cargo -ErrorAction SilentlyContinue)) -and (Test-Path $cargoExe)) {
    $env:PATH = "$cargoBin;$env:PATH"
  }

  if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "Rust/Cargo is not installed or not on PATH. Install Rust from https://rustup.rs/ and reopen PowerShell."
  }
}

function Ensure-TauriCli {
  Ensure-Cargo

  & cargo tauri --version | Out-Null
  if ($LASTEXITCODE -ne 0) {
    throw "The Tauri CLI is not available. Run 'cargo install tauri-cli', then reopen PowerShell and try again."
  }
}

function Invoke-InstallAll {
  Ensure-Venv
  Invoke-Uv "pip" "install" "-e" ".[all]"
}

switch ($Task.ToLowerInvariant()) {
  "help" {
    @(
      "Windows task runner for BOT-MMORPG-AI",
      "",
      "Usage:",
      "  powershell -ExecutionPolicy Bypass -File .\scripts\windows_tasks.ps1 <task>",
      "",
      "Tasks:",
      "  install-all       Install all Python dependencies into .venv",
      "  dev               Run the Tauri desktop app from source",
      "  run               Alias for dev",
      "  dev-sidecar       Run only the Python sidecar",
      "  dev-sidecar-test  Smoke-test the Python sidecar",
      "  doctor            Run the backend health ladder",
      "  run-backend       Run backend/main_backend.py directly",
      "  build-installer   Build the Windows installer",
      "  verify-installer  Verify the generated installer",
      "  test-installer    Run the installer test script",
      "  clean-installer   Remove installer build artifacts",
      "  install-drivers   Run the Windows driver installer helper"
    ) | ForEach-Object { Write-Host $_ }
  }
  "install-all" {
    Invoke-InstallAll
  }
  "dev" {
    Invoke-InstallAll
    Ensure-TauriCli
    Invoke-Checked "python" @("scripts/stamp_ui_build_tag.py")
    Push-Location (Join-Path $repoRoot "src-tauri")
    try {
      Invoke-Checked "cargo" @("tauri", "dev")
    } finally {
      Pop-Location
    }
  }
  "run" {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $repoRoot "scripts\windows_tasks.ps1") "dev"
    exit $LASTEXITCODE
  }
  "dev-sidecar" {
    Invoke-InstallAll
    $dataRoot = Join-Path $repoRoot ".dev-data"
    New-Item -ItemType Directory -Force -Path $dataRoot | Out-Null
    Invoke-Uv "run" "python" "backend/entry_main.py" "--port" "0" "--token" "devtoken" "--resource-root" $repoRoot "--data-root" $dataRoot
  }
  "dev-sidecar-test" {
    Invoke-InstallAll
    Invoke-Uv "run" "python" "scripts/dev_sidecar_smoke.py"
  }
  "doctor" {
    Invoke-InstallAll
    Invoke-Uv "run" "python" "-c" "import sys; import torch; import fastapi; import uvicorn; import modelhub; import modelhub.tauri; print(f'  -> python {sys.version.split()[0]} | torch {torch.__version__} | fastapi {fastapi.__version__} | modelhub.tauri {modelhub.tauri.__file__!r}')"
    Invoke-Uv "run" "python" "scripts/dev_sidecar_smoke.py"
    Invoke-Uv "run" "python" "-m" "pytest" "tests/test_backend_startup.py" "tests/test_jobs_routes.py" "tests/test_jobs_runner.py" "tests/test_diagnostics_smoke.py" "tests/test_health_probe_smoke.py" "tests/test_runtime_doctor.py" "--no-cov" "-q"
  }
  "run-backend" {
    Invoke-InstallAll
    Invoke-Uv "run" "python" "backend/main_backend.py"
  }
  "build-installer" {
    Invoke-InstallAll
    Ensure-TauriCli
    Invoke-Checked "python" @("scripts/stamp_ui_build_tag.py")
    Invoke-Checked "powershell" @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/build_pipeline.ps1")
  }
  "verify-installer" {
    Invoke-Checked "powershell" @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/verify_installer.ps1")
  }
  "test-installer" {
    Invoke-Checked "powershell" @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/test_installer.ps1")
  }
  "clean-installer" {
    $targets = @("dist", "build", "src-tauri\target", "src-tauri\binaries", "*.spec")
    foreach ($target in $targets) {
      Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $target
    }
  }
  "install-drivers" {
    Invoke-Checked "powershell" @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/install_windows.ps1")
  }
  default {
    throw "Unknown task '$Task'. Run '.\scripts\windows_tasks.ps1 help' for the task list."
  }
}
