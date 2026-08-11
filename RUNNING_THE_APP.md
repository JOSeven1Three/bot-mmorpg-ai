# Running BOT MMORPG AI - Complete Guide

> **Windows 10/11 required.** The desktop UI requires [Rust](https://rustup.rs/) and the Tauri CLI (`cargo install tauri-cli`).
>
> **Windows note:** PowerShell usually does not include `make`. Use `.\scripts\windows_tasks.ps1` for the Windows-native commands in this guide. The wrapper also checks for both `cargo` and the Tauri CLI before trying to launch the desktop app.

## Quick Start

### 1. Install Dependencies

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_tasks.ps1 install-all
```

This installs:
- Python dependencies (via uv)
- Creates virtual environment
- Installs all required packages

> **Note:** A virtual environment (`.venv/`) is created automatically. You do NOT need to activate it manually.

### 2. Run the Application

**Requires:** Rust + Tauri CLI (for the desktop UI). If you only need the Python pipeline, skip to "Run Without Desktop UI" below.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_tasks.ps1 run
```

**OR** (same thing):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_tasks.ps1 dev
```

This will:
- Start the Tauri development server
- Launch the desktop application window
- Automatically start the Python backend as a sidecar
- Open the UI at `tauri://localhost`

---

## Important: This is NOT a Node.js Project!

### ❌ Common Mistake

```bash
npm run dev  # ❌ This WILL NOT work!
```

**Error you'll see:**
```
npm error enoent Could not read package.json: Error: ENOENT: no such file or directory
```

### ✅ Correct Way

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_tasks.ps1 run
```

---

## Project Structure

```
BOT-MMORPG-AI/
├── tauri-ui/              # ← Frontend (Plain HTML/CSS/JavaScript)
│   ├── index.html         # Main UI
│   └── main.js            # JavaScript (ES modules)
│
├── backend/               # ← Python Backend
│   └── entry_main.py    # HTTP API server
│
├── src-tauri/             # ← Tauri (Rust) Desktop Framework
│   ├── src/main.rs        # Rust entry point
│   ├── tauri.conf.json    # Tauri configuration
│   └── Cargo.toml         # Rust dependencies
│
├── src/bot_mmorpg/        # ← Python Package
│   └── scripts/           # AI scripts (collect, train, play)
│
└── Makefile               # ← Commands for running the app
```

---

## Architecture

This is a **Tauri desktop application**:

```
┌────────────────────────────────────────────┐
│     Tauri Desktop Window                   │
│  ┌──────────────────────────────────────┐  │
│  │  Frontend (tauri-ui/)                │  │
│  │  - Plain HTML/CSS/JavaScript         │  │
│  │  - No npm, no webpack, no build step │  │
│  │  - Direct ES modules                 │  │
│  └──────────────────────────────────────┘  │
│              │                              │
│              │ HTTP/JSON                    │
│              ↓                              │
│  ┌──────────────────────────────────────┐  │
│  │  Backend (Python Sidecar)            │  │
│  │  - Starts automatically              │  │
│  │  - HTTP server on random port        │  │
│  │  - API endpoints for AI functions    │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

**Key Points:**
- ✅ Frontend uses **plain HTML/JavaScript** (no npm needed)
- ✅ Backend starts **automatically** when app launches
- ✅ Communication via **HTTP/JSON** on localhost
- ✅ No build step for frontend (direct file serving)

---

## Why No `package.json`?

This project deliberately avoids Node.js complexity:

### Traditional Web Dev
```
npm install → webpack → babel → 1000 packages → build → dist/
```

### This Project
```
Plain HTML/JS → Tauri → Done!
```

**Benefits:**
- 🚀 No npm dependency hell
- ⚡ No build time for frontend
- 🎯 Simple, direct development
- 📦 Smaller footprint

---

## Running Different Components

### Full Application (Recommended)
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_tasks.ps1 run
```
Starts everything together.

### Backend Only (Testing)
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_tasks.ps1 run-backend
```
Runs just the Python backend server.

### Python Scripts Directly
```bash
# Collect data
make collect-data

# Train model
make train-model

# Test model
make test-model
```

---

## Development Workflow

### 1. Make Changes to UI

Edit files in `tauri-ui/`:
```bash
# Edit the HTML
vim tauri-ui/index.html

# Edit the JavaScript
vim tauri-ui/main.js
```

Changes are **automatically reloaded** when you save!

### 2. Make Changes to Backend

Edit `backend/entry_main.py`:
```bash
vim backend/entry_main.py
```

Then **restart** the app:
```bash
# Stop the app (Ctrl+C)
# Start again
make run
```

### 3. Make Changes to Python Scripts

Edit scripts in `src/bot_mmorpg/scripts/`:
```bash
vim src/bot_mmorpg/scripts/collect_data.py
```

Changes take effect on next run (no restart needed).

---

## Prerequisites

### Required

1. **Python 3.10+**
   ```bash
   python --version
   ```

2. **Rust + Cargo** (for Tauri)
   ```bash
   cargo --version
   ```

   If missing: https://rustup.rs/

3. **uv** (Python package manager)
   ```bash
   make install-uv
   ```

### Optional (Windows only)

- **Visual Studio Build Tools** (for native Python packages)
- **NSIS** (for installer builds)

---

## Troubleshooting

### "npm run dev" doesn't work

**Problem:** Wrong command for this project.

**Solution:** Use `make run` instead.

### "cargo: command not found"

**Problem:** Rust not installed.

**Solution:**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Or on Windows
# Download from: https://rustup.rs/

# Then restart terminal
cargo --version
```

### "uv: command not found"

**Problem:** uv not installed.

**Solution:**
```bash
make install-uv
# Then restart terminal
```

### Backend doesn't start

**Problem:** Python dependencies not installed.

**Solution:**
```bash
make install-all
```

### UI shows "Backend not ready"

**Problem:** Backend failed to start or crashed.

**Check:**
1. Look in terminal for Python errors
2. Verify Python 3.10+ is installed
3. Check that dependencies are installed

### Port already in use

**Problem:** Another instance is running.

**Solution:**
```bash
# Kill other instances
pkill -f main-backend

# Or on Windows
taskkill /F /IM main-backend.exe

# Then try again
make run
```

---

## Building the Installer

### Prerequisites (Windows Only)

1. Install all dependencies:
   ```bash
   make install-all
   ```

2. Install Rust:
   ```bash
   # Download from https://rustup.rs/
   ```

3. Install Tauri CLI:
   ```bash
   cargo install tauri-cli
   ```

### Build

```bash
make build-installer
```

This creates:
```
src-tauri/target/release/bundle/nsis/BOT-MMORPG-AI_0.1.5_x64-setup.exe
```

### Verify

```bash
make verify-installer
```

### Test

```bash
make test-installer
```

---

## Common Commands Reference

| Command | Description |
|---------|-------------|
| `.\scripts\windows_tasks.ps1 help` | Show Windows-native task commands |
| `.\scripts\windows_tasks.ps1 install-all` | Install all dependencies |
| `.\scripts\windows_tasks.ps1 run` | Run the application (dev mode) |
| `.\scripts\windows_tasks.ps1 dev` | Same as `run` |
| `.\scripts\windows_tasks.ps1 run-backend` | Run only backend (testing) |
| `uv run python versions/0.01/1-collect_data.py` | Run data collection |
| `uv run python versions/0.01/2-train_model.py` | Train AI model |
| `uv run python versions/0.01/3-test_model.py` | Test trained model |
| `.\scripts\windows_tasks.ps1 build-installer` | Build Windows installer |
| `.\scripts\windows_tasks.ps1 verify-installer` | Verify installer build |
| `.\scripts\windows_tasks.ps1 test-installer` | Test installer package |
| `python -m compileall src backend modelhub launcher` | Quick syntax validation |

---

## FAQ

### Q: Can I use npm/webpack/vite?

**A:** Not needed! The frontend is intentionally simple (plain HTML/JS) to avoid build complexity.

### Q: How do I add a new UI feature?

**A:** Just edit `tauri-ui/index.html` or `tauri-ui/main.js`. Changes reload automatically.

### Q: How do I add a new backend endpoint?

**A:** Edit `backend/entry_main.py`, add your endpoint, restart the app.

### Q: Can I use TypeScript?

**A:** Not currently set up. The project uses plain JavaScript for simplicity.

### Q: Where are the AI models?

**A:** Models are in `artifacts/model/` (created by training) or downloaded to `models/` by the installer.

### Q: How do I debug?

**A:**
- Frontend: Browser DevTools (F12 in the app window)
- Backend: Check terminal output for Python errors
- Rust: Check terminal for Tauri/Rust errors

---

## Production Build

### For Distribution

```bash
# Windows
make artifact

# This creates a full installer at:
# src-tauri/target/release/bundle/nsis/BOT-MMORPG-AI_0.1.5_x64-setup.exe
```

The installer includes:
- ✅ Tauri desktop application
- ✅ Python backend (bundled)
- ✅ All dependencies
- ✅ Driver installers
- ✅ Component selection wizard
- ✅ Professional UI

Users can just run the installer - no Python, Rust, or dependencies needed!

---

## Summary

**To run the app:**
```bash
make install-all  # First time only
make run          # Start the app
```

**NOT this:**
```bash
npm run dev  # ❌ Wrong! This is not a Node.js project
```

**Frontend location:**
- ✅ `tauri-ui/` (the Tauri desktop UI -- this is the "frontend" for the app)
- ⚠️ `frontend/` does exist, but it's unrelated legacy input-recording tooling
  (AutoHotPy backups, keyboard/mouse capture scripts), not a web frontend.
  Don't confuse it with `tauri-ui/`.

**Frontend type:**
- ✅ Plain HTML/CSS/JavaScript
- ❌ NOT React/Vue/npm-based

**Backend:**
- ✅ Starts automatically as Tauri sidecar
- ✅ Python HTTP server
- ✅ No manual startup needed

---

**Happy Coding!** 🎮✨

If you have questions, check:
- `NOTES.md` - Architecture and build-pipeline details
- `TWO_UIS_EXPLAINED.md` - Launcher vs. Tauri UI
- `INSTALLER.md` - Installer features and build system
- `docs/installer/` - Deeper installer runtime-flow and debugging docs
