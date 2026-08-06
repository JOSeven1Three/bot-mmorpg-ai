# Setup and Installation Guide - BOT MMORPG AI

## Current Issues and Solutions

### Issue 1: `make run` fails with "cargo is not recognized"

**Error:**
```
'cargo' is not recognized as an internal or external command
make: *** [Makefile:187: run] Error 1
```

**Root Cause:** Rust/Cargo is not installed on your system.

**Solution:** Install Rust before running the application. On Windows PowerShell, use `.\scripts\windows_tasks.ps1 run` instead of depending on `make`.

---

### Issue 2: `npm run dev` fails with missing package.json

**Error:**
```
npm error enoent Could not read package.json
```

**Root Cause:** This is **NOT a Node.js project**. There is no `frontend/` directory with npm.

**Solution:** Use `.\scripts\windows_tasks.ps1 run` instead (after installing Rust).

---

### Issue 3: Installer is only 0.1 MB

**Error:** Installer builds but is tiny (99 KB instead of 20+ MB).

**Root Cause:** NSIS template wasn't including the main .exe file.

**Solution:** ✅ **FIXED** in latest commit - template now includes main executable.

---

## Complete Setup Instructions

### Prerequisites

You need to install these tools **BEFORE** running the application:

#### 1. Python 3.10 or 3.11 ✅ (You have this)

```powershell
python --version
# Should show: Python 3.10.x or 3.11.x
```

#### 2. Rust and Cargo ❌ (You need to install this!)

**Installation (Windows):**

```powershell
# Download and run the Rust installer
# Visit: https://rustup.rs/

# Or use this direct link:
# https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe

# After installation, close and reopen PowerShell/Terminal
# Then verify:
cargo --version
# Should show: cargo 1.xx.x
```

**Alternative method (using winget):**

```powershell
winget install Rustlang.Rustup
```

**After installation:**
- Close and reopen your terminal
- Verify: `cargo --version`

#### 2b. Tauri CLI âŒ (Also required for the desktop app)

```powershell
cargo install tauri-cli
cargo tauri --version
```

If `cargo` works but `cargo tauri --version` fails, the desktop UI still will not start.

#### 3. uv (Python package manager) ⚠️ (Optional but recommended)

```powershell
# Install via pip
python -m pip install uv

# Or install dependencies through the Windows task wrapper
powershell -ExecutionPolicy Bypass -File .\scripts\windows_tasks.ps1 install-all
```

---

## Step-by-Step Setup

### Step 1: Install Rust (Required!)

```powershell
# Download from: https://rustup.rs/
# Run the installer
# Close and reopen terminal

# Verify installation
cargo --version
rustc --version
```

### Step 2: Install Python Dependencies

```powershell
# From project root
powershell -ExecutionPolicy Bypass -File .\scripts\windows_tasks.ps1 install-all
```

This will:
- Install `uv` if not present
- Create Python virtual environment
- Install all Python dependencies

### Step 3: Run the Application

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_tasks.ps1 run
```

This will:
- Start Tauri development server
- Launch the desktop application
- Auto-start Python backend
- Open the UI window

---

## Why You Need Rust

This is a **Tauri desktop application**:

```
BOT MMORPG AI
├─ Tauri Framework (Rust) ← Desktop window, WebView
│  └─ cargo tauri dev     ← Requires Rust/Cargo
├─ Frontend (HTML/JS)     ← UI in tauri-ui/
└─ Backend (Python)       ← API server (auto-starts)
```

**Tauri** is written in Rust, so you need Rust/Cargo to:
- Run development server (`.\scripts\windows_tasks.ps1 run`)
- Build the application
- Create installers

---

## Common Errors and Solutions

### Error: "cargo is not recognized"

**Problem:** Rust not installed or not in PATH.

**Solutions:**

1. **Install Rust:**
   ```powershell
   # Visit https://rustup.rs/ and download installer
   ```

2. **Check PATH after install:**
   ```powershell
   # Restart terminal first!
   cargo --version
   ```

3. **Manual PATH check:**
   ```powershell
   # Rust binaries should be in:
   # C:\Users\<YourName>\.cargo\bin

   # Check if it's in PATH:
   $env:PATH -split ';' | Select-String "cargo"
   ```

4. **If still not working, add to PATH manually:**
   ```powershell
   # Add to system PATH:
   $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
   $cargoPath = "$env:USERPROFILE\.cargo\bin"
   [Environment]::SetEnvironmentVariable("Path", "$userPath;$cargoPath", "User")

   # Restart terminal
   ```

### Error: "npm is not recognized" or "package.json not found"

**Problem:** Trying to use npm commands on a non-Node.js project.

**Why this happens:**
- Old habit from web development
- Looking in wrong directory (`frontend/` doesn't exist)
- This project doesn't use npm at all!

**Solution:**

❌ **DON'T do this:**
```powershell
cd frontend/        # Doesn't exist!
npm install         # Not needed!
npm run dev         # Won't work!
```

✅ **DO this instead:**
```powershell
cd C:\workspace\BOT-MMORPG-AI
make install-all    # First time
make run            # Start app
```

### Error: "uv is not recognized"

**Problem:** uv package manager not installed.

**Solution:**
```powershell
python -m pip install uv

# Or
make install-uv
```

### Error: Build fails with "NSIS error" or "icon not found"

**Problem:** NSIS template issues (already fixed in latest commit).

**Solution:**
```powershell
# Pull latest changes
git pull origin claude/fix-installer-path-wizard-uql5P

# Clean and rebuild
make clean-installer
make build-installer
```

---

## Project Structure (Important!)

```
BOT-MMORPG-AI/
├─ tauri-ui/           ← Frontend (HTML/CSS/JS) - The REAL frontend!
│  ├─ index.html       ← Main UI
│  └─ main.js          ← JavaScript logic
│
├─ backend/            ← Python Backend
│  └─ main_backend.py  ← HTTP API server
│
├─ src-tauri/          ← Tauri (Rust) Framework
│  ├─ src/main.rs      ← Rust entry point
│  └─ tauri.conf.json  ← Configuration
│
├─ src/bot_mmorpg/     ← Python AI Package
│  └─ scripts/         ← AI scripts
│
└─ Makefile            ← Commands for build/run
```

**Key Points:**
- ✅ Frontend is in `tauri-ui/` (plain HTML/JS)
- ❌ There is NO `frontend/` directory
- ❌ There is NO `package.json`
- ❌ There is NO npm build process
- ✅ Uses Tauri (Rust) for desktop framework
- ✅ Uses Python for backend API

---

## Correct Workflow

### Development

```powershell
# 1. First time setup (after installing Rust!)
make install-all

# 2. Run the application
make run

# 3. Edit files (changes auto-reload)
# Edit: tauri-ui/index.html
# Edit: tauri-ui/main.js
# Edit: backend/main_backend.py

# 4. Stop with Ctrl+C
```

### Building Installer

```powershell
# 1. Make sure everything is installed
cargo --version  # Must work!
python --version # Must work!

# 2. Build
make build-installer

# 3. Verify (should be 20-50 MB, not 0.1 MB!)
ls -lh src-tauri/target/release/bundle/nsis/*.exe

# 4. Test
make verify-installer
make test-installer
```

---

## Installation Check

Run these commands to verify your setup:

```powershell
# Python check
python --version
# ✅ Should show: Python 3.10.x or 3.11.x

# Rust check
cargo --version
# ✅ Should show: cargo 1.xx.x
# ❌ If error: Install Rust from https://rustup.rs/

# uv check (optional)
uv --version
# ✅ Should show: uv x.x.x
# ⚠️ If error: Run 'make install-uv'

# Make check
make --version
# ✅ Should show: GNU Make x.x
# ⚠️ If error: Install via chocolatey or similar
```

---

## Quick Reference

### ✅ Correct Commands

```powershell
make install-all     # Install dependencies (first time)
make run             # Run the application
make dev             # Same as 'make run'
make build-installer # Build Windows installer
```

### ❌ Wrong Commands (Don't Use)

```powershell
npm install          # Not a Node.js project!
npm run dev          # Won't work!
cd frontend/         # Doesn't exist!
npm run build        # Not needed!
```

---

## Architecture Summary

```
┌─────────────────────────────────────────┐
│   Tauri Desktop Application (Rust)     │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  Frontend (tauri-ui/)             │ │
│  │  - Plain HTML/CSS/JavaScript      │ │
│  │  - No npm, no build, no webpack   │ │
│  │  - Direct file serving            │ │
│  └───────────────────────────────────┘ │
│           ↓ HTTP/JSON                   │
│  ┌───────────────────────────────────┐ │
│  │  Backend (Python Sidecar)         │ │
│  │  - Starts automatically           │ │
│  │  - HTTP API server                │ │
│  │  - AI functionality               │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

**Dependencies:**
- Rust/Cargo → Required for Tauri desktop framework
- Python 3.10+ → Required for backend
- uv → Optional (faster pip alternative)

**NOT needed:**
- ❌ Node.js
- ❌ npm
- ❌ webpack
- ❌ React/Vue/etc

---

## Troubleshooting

### "I installed Rust but cargo still not found"

1. **Close and reopen terminal** (PATH needs refresh)
2. **Check installation:**
   ```powershell
   ls $env:USERPROFILE\.cargo\bin\cargo.exe
   ```
3. **If file exists but still not found:**
   ```powershell
   # Add to PATH manually
   $env:PATH += ";$env:USERPROFILE\.cargo\bin"
   cargo --version
   ```

### "Application window opens but shows blank"

**Check:**
1. Backend started? (Look for port in terminal)
2. UI files present? (`ls tauri-ui/index.html`)
3. Browser console errors? (Press F12 in app window)

### "Backend crashes immediately"

**Check:**
1. Python dependencies installed? (`make install-all`)
2. Python version correct? (`python --version` → 3.10 or 3.11)
3. Virtual environment active? (should be automatic)

### "Installer still 0.1 MB after rebuild"

**Check:**
1. Latest code? (`git pull origin claude/fix-installer-path-wizard-uql5P`)
2. Clean build? (`make clean-installer` then `make build-installer`)
3. Backend built? (`ls src-tauri/binaries/main-backend*.exe`)

---

## Summary

**To run this application, you need:**

1. ✅ Python 3.10+ (you have this)
2. ❌ **Rust/Cargo** (you need to install this!)
3. ⚠️ uv (optional but recommended)

**Installation order:**
```powershell
# 1. Install Rust from https://rustup.rs/
# 2. Close and reopen terminal
# 3. Run: make install-all
# 4. Run: make run
```

**This is NOT:**
- ❌ A Node.js project
- ❌ An npm-based project
- ❌ Using webpack/babel/etc

**This IS:**
- ✅ A Tauri desktop app (Rust)
- ✅ With plain HTML/JS frontend
- ✅ With Python backend

**Questions?**
- Check: RUNNING_THE_APP.md
- Check: UI_BACKEND_INTEGRATION_ANALYSIS.md
- Check: BUILD_STATUS.md

---

**Last Updated:** 2026-01-12
**Status:** ✅ Installer fix committed, setup guide complete
**Next Step:** Install Rust, then run `make run`
