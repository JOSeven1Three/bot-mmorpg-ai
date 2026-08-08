## 📦 Deployment Strategy (v2.0)

To avoid Windows Installer size limits (NSIS 2GB crash) and improve update speeds, this project uses a **"Downloader Stub"** architecture.

### How it works
1.  **The Installer (`.exe`)**: Is lightweight (~300MB). It contains the App, UI, Python Runtime, and basic libraries (NumPy, OpenCV). It **does not** contain PyTorch.
2.  **The Engine (`ml-engine.zip`)**: A separate ~200MB archive containing PyTorch and torchvision (much smaller than the legacy TensorFlow setup!).

### Build Process
1.  **Build the App:**
    ```powershell
    make build-installer
    # Output: src-tauri/target/release/bundle/nsis/setup.exe
    ```
2.  **Build the Engine Add-on:**
    ```powershell
    .\scripts\package_ml_addon.ps1
    # Output: dist/ml-engine.zip
    ```
3.  **Release:**
    * Upload `setup.exe` to users.
    * Upload `ml-engine.zip` to your hosting provider.
    * Publish the asset with the exact name `ml-engine.zip`. The desktop app
      downloads it from the continuation repository's `latest/download` URL.
### Phase 1: Modify the Build Pipeline (The "Light" Installer)

We first instruct your existing build script to **ignore** the heavy ML libraries.

**Status:** Implemented in `scripts/build_pipeline.ps1` and `pyproject.toml`.
PyTorch, torchvision, and timm are optional ML dependencies, so they cannot
leak into the base project dependency list or the light installer will still
bundle them.

```powershell
# scripts/build_pipeline.ps1

# ... inside Step 1 ...
& (Join-Path $root "scripts\prepare_python_from_pyproject_embed310_target.ps1") `
  -Extras @("launcher","backend") `
  -TargetTag "win_amd64_cp310" `
  -RebuildTarget

```

**Result:** Running `make build-installer` will now produce a small, fast installer that has Python, OpenCV, and NumPy, but **no PyTorch**.

---

### Phase 2: Create the "ML Asset Package" Script

We need a new script to build the "Add-on" zip file containing just the heavy stuff.

**Status:** Implemented in `scripts/package_ml_addon.ps1`.

```powershell
<#
scripts/package_ml_addon.ps1
Builds a standalone .zip containing ONLY the ML libraries (PyTorch, torchvision, timm)
#>
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$distDir = Join-Path $root "dist"
$mlStaging = Join-Path $root "dist\ml-addon"
$mlZip = Join-Path $distDir "ml-engine.zip"

Write-Host "Building ML Add-on Package..." -ForegroundColor Cyan

# 1. Setup staging area
if (Test-Path $mlStaging) { Remove-Item -Recurse -Force $mlStaging }
New-Item -ItemType Directory -Force -Path $mlStaging | Out-Null

# 2. Use the existing build venv to drive pip
$py = Join-Path $root ".venv\Scripts\python.exe"

# 3. Install the ML engine packages and their required transitive dependencies.
# The production script constrains shared libraries to the light installer's
# supported version ranges before packaging the archive.
Write-Host "Installing ML libs..."
& $py -m pip install `
  "torch>=2.0.0,<3.0.0" `
  "torchvision>=0.15.0,<1.0.0" `
  "timm>=0.9.0,<2.0.0" `
  --target $mlStaging

# 4. Clean up cache only. Keep dist-info because runtime extensions can consult it.
Get-ChildItem $mlStaging -Filter "__pycache__" -Recurse | Remove-Item -Recurse -Force

# 5. Zip it up
Write-Host "Zipping to $mlZip..."
Compress-Archive -Path "$mlStaging\*" -DestinationPath $mlZip -Force

Write-Host "SUCCESS: ML Add-on created at $mlZip" -ForegroundColor Green

```

**Workflow:**

1. Run `.\scripts\package_ml_addon.ps1` from the CPython 3.10 build venv
   (the script refuses any other ABI).
2. It creates `dist/ml-engine.zip` (~1GB).
3. **Upload this zip** to your release server (e.g., GitHub Releases, AWS S3, or your VPS).

---

### Phase 3: Update Rust Backend (`src-tauri/src/main.rs`)

Now we teach the Rust app to check if PyTorch is missing and download it on demand.

**1. Update `src-tauri/Cargo.toml` dependencies:**

```toml
[dependencies]
reqwest = { version = "0.11", features = ["blocking", "json"] } 
zip = "0.6"

```

**2. Add commands to `src-tauri/src/main.rs`:**

```rust
// Add imports
use std::fs::File;
use std::io::{Cursor, Read, Write};
use zip::ZipArchive;

// 1. Check if ML Engine exists
#[tauri::command]
fn check_ml_status(app: AppHandle) -> bool {
    let py_dir = managed_site_packages_dir(&app);
    // If 'torch' folder exists, we assume the engine is ready
    py_dir.join("torch").exists()
}

// 2. Download and Install Command
#[tauri::command]
async fn install_ml_engine(app: AppHandle, window: Window) -> Result<String, String> {
    // REPLACE THIS WITH YOUR REAL URL
    let url = "https://github.com/JOSeven1Three/bot-mmorpg-ai/releases/latest/download/ml-engine.zip";
    let target_dir = managed_site_packages_dir(&app);

    window.emit("download_progress", "Downloading AI Engine (approx 200MB)...").unwrap();

    // A. Download (Blocking for simplicity, or use async stream for progress bar)
    let response = reqwest::get(url).await.map_err(|e| e.to_string())?;
    let content = response.bytes().await.map_err(|e| e.to_string())?;

    window.emit("download_progress", "Extracting...").unwrap();

    // B. Unzip directly into site-packages
    let reader = Cursor::new(content);
    let mut archive = ZipArchive::new(reader).map_err(|e| e.to_string())?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i).unwrap();
        // Security check: ensure file path doesn't escape directory
        let outpath = target_dir.join(file.mangled_name());

        if file.name().ends_with('/') {
            std::fs::create_dir_all(&outpath).unwrap();
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() { std::fs::create_dir_all(p).unwrap(); }
            }
            let mut outfile = File::create(&outpath).unwrap();
            std::io::copy(&mut file, &mut outfile).unwrap();
        }
    }

    window.emit("download_progress", "AI Engine Ready!").unwrap();
    Ok("Success".to_string())
}

// REMEMBER to register these in the `tauri::Builder` invoke_handler!

```

---

### Phase 4: Frontend UI Logic (JavaScript)

In your `index.html` or main JS file, update the logic for the "Start Bot" or "Train" button.

```javascript
async function onStartBotClick() {
  // 1. Check if engine is installed
  const hasEngine = await window.__TAURI__.invoke('check_ml_status');

  if (!hasEngine) {
    // 2. Not found? Trigger download flow
    const confirm = await confirm("AI Engine missing. Download (~200MB)?");
    if (!confirm) return;

    // Show a progress spinner/modal here
    document.getElementById("status").innerText = "Downloading...";
    
    try {
      await window.__TAURI__.invoke('install_ml_engine');
      alert("Installation Complete!");
      // Recursively call start to run now
      onStartBotClick();
    } catch (err) {
      alert("Download failed: " + err);
    }
    return;
  }

  // 3. Engine found, proceed normally
  window.__TAURI__.invoke('start_bot');
}

```

---

## Future Features Roadmap

### Mouse Recording (Issue #21)

**Status:** Planned for v2.1
**Priority:** High
**Requested by:** Community (Patrwer, BenoitPALISSE)

Currently the data collection system only records keyboard and gamepad inputs.
Mouse input recording is needed for games that rely heavily on mouse
movement, left/right click, and scroll actions.

**Implementation plan (enterprise-grade):**

1. **Input layer** (`src/bot_mmorpg/scripts/mouse_capture.py`):
   - Record mouse position (x, y) relative to the game window at each frame.
   - Record button states: LMB, RMB, MMB, scroll delta.
   - Use `pynput.mouse.Listener` (cross-platform) or Interception driver
     (low-level, Windows) depending on game requirements.
   - Normalize coordinates to `[0, 1]` range relative to capture region so the
     model is resolution-independent.

2. **Data format** - extend the label vector:
   ```
   [keyboard_9] + [gamepad_20] + [mouse_x, mouse_y, lmb, rmb, mmb, scroll]
   Total: 9 + 20 + 6 = 35 actions
   ```

3. **Model changes**:
   - Split head architecture: discrete actions (keyboard/gamepad) use
     cross-entropy, continuous actions (mouse x/y/scroll) use MSE loss.
   - Multi-head output: `ActionHead(29)` + `MouseHead(6)`.

4. **Replay/inference**:
   - Mouse movement via `pynput.mouse.Controller` or Interception.
   - Smooth interpolation between predicted positions to avoid jitter.

5. **Performance considerations**:
   - Mouse data is high-frequency (>60 Hz); downsample to match frame rate.
   - Ring buffer for temporal mouse trajectory (last N positions).

---

### Anti-Cheat Considerations (Issue - Tibia/BattlEye)

**Status:** Research / Documentation only
**Priority:** Informational

This project is designed for **educational and research purposes**. It operates
exclusively through screen capture (reading pixels) and does not inject into
game processes, modify game memory, or hook game APIs.

**Architecture decisions that minimize detection surface:**

- **Screen capture only**: Uses `mss` (screenshot library) which reads the
  display framebuffer, not the game process memory.
- **Input simulation**: Uses OS-level virtual input (vJoy, Interception) which
  operates at the driver level, not inside the game process.
- **No process injection**: The bot runs as a completely separate process.
- **No memory reading**: No ReadProcessMemory or similar calls.

**Important notes for users:**

- Using any form of automation in online games **may violate the game's Terms of
  Service** and can result in account bans regardless of the method used.
- Anti-cheat systems like BattlEye, EasyAntiCheat, and Vanguard may detect
  virtual input drivers (vJoy, Interception) even when used for legitimate
  purposes.
- This project does **not** provide anti-cheat bypass functionality.
- Users are responsible for understanding and complying with the terms of
  service of any game they use this software with.
- The recommended use case is **single-player games**, **private servers**, or
  **games that explicitly allow automation**.

---

### Planned Features Summary

| Feature | Version | Status |
|---|---|---|
| Mouse recording and replay | v2.1 | Planned |
| Multi-monitor support | v2.1 | Planned |
| Arduino/hardware input bridge | v2.2 | Research |
| OBS virtual camera integration | v2.2 | Research |
| Reinforcement learning mode | v3.0 | Design phase |
| Cloud training (GPU rental) | v3.0 | Design phase |
| Plugin system for custom games | v3.0 | Design phase |

