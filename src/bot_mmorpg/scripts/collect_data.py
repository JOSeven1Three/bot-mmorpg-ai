"""
Data Collection Script for BOT-MMORPG-AI

Captures screen and input data for training neural networks.
Includes comprehensive error handling for production use.

Supports keyboard, gamepad, and optional mouse recording.
Mouse recording is additive and non-destructive: when enabled it
appends 6 extra values to the action vector; when disabled the
existing keyboard+gamepad pipeline is completely unchanged.
"""

import argparse
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import helper modules with fallback
try:
    from .getgamepad import gamepad_check
    from .getkeys import key_check
    from .grabscreen import grab_screen
except ImportError:
    try:
        from getgamepad import gamepad_check
        from getkeys import key_check
        from grabscreen import grab_screen
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error(
            "Please ensure grabscreen.py, getkeys.py, and getgamepad.py are available"
        )
        grab_screen = None
        key_check = None
        gamepad_check = None

# Optional mouse capture – additive only, never breaks existing pipeline
try:
    from .mouse_capture import MouseCapture
except ImportError:
    try:
        from mouse_capture import MouseCapture
    except ImportError:
        MouseCapture = None


class DataCollectionError(Exception):
    """Custom exception for data collection errors."""

    pass


class ScreenCaptureError(DataCollectionError):
    """Raised when screen capture fails."""

    pass


class InputCaptureError(DataCollectionError):
    """Raised when input capture fails."""

    pass


def build_game_output_dir(base_out: str, game_id: str, task: str) -> str:
    """Build a timestamped output folder for game-specific capture sessions."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(base_out)

    if base.as_posix() in {"data/raw", "datasets"}:
        return str(Path("datasets") / game_id / f"{ts}_{task}")

    return str(base / game_id / f"{ts}_{task}")


def keys_to_output(keys: List[str]) -> List[int]:
    """
    One-hot encode keyboard input.

    Args:
        keys: List of pressed key characters

    Returns:
        One-hot encoded list [W, S, A, D, WA, WD, SA, SD, NOKEY]
    """
    output = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    if "W" in keys and "A" in keys:
        output[4] = 1
    elif "W" in keys and "D" in keys:
        output[5] = 1
    elif "S" in keys and "A" in keys:
        output[6] = 1
    elif "S" in keys and "D" in keys:
        output[7] = 1
    elif "W" in keys:
        output[0] = 1
    elif "S" in keys:
        output[1] = 1
    elif "A" in keys:
        output[2] = 1
    elif "D" in keys:
        output[3] = 1
    else:
        output[8] = 1

    return output


def gamepad_keys_to_output(lista: List) -> List[int]:
    """Convert gamepad input to integer list."""
    return [int(elem) for elem in lista]


def validate_dependencies() -> bool:
    """
    Validate that all required dependencies are available.

    Returns:
        True if all dependencies are available

    Raises:
        DataCollectionError: If dependencies are missing
    """
    missing = []

    if grab_screen is None:
        missing.append("grabscreen")
    if key_check is None:
        missing.append("getkeys")
    if gamepad_check is None:
        missing.append("getgamepad")

    if missing:
        raise DataCollectionError(
            f"Missing required modules: {', '.join(missing)}. "
            "Please ensure all helper modules are installed."
        )

    return True


def capture_screen(
    region: Tuple[int, int, int, int], target_size: Tuple[int, int] = (480, 270)
) -> np.ndarray:
    """
    Capture and process screen region.

    Args:
        region: Screen region (x1, y1, x2, y2)
        target_size: Target resize dimensions (width, height)

    Returns:
        Processed screen image as numpy array

    Raises:
        ScreenCaptureError: If capture fails
    """
    try:
        screen = grab_screen(region=region)

        if screen is None or screen.size == 0:
            raise ScreenCaptureError("Screen capture returned empty image")

        # Resize
        screen = cv2.resize(screen, target_size)

        # Convert color space
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        return screen

    except ScreenCaptureError:
        raise
    except Exception as e:
        raise ScreenCaptureError(f"Failed to capture screen: {e}")


def capture_input(
    mouse_capturer: Optional[object] = None,
) -> Tuple[List[int], List[int], Optional[np.ndarray]]:
    """
    Capture keyboard, gamepad, and (optionally) mouse input.

    Mouse recording is additive and non-destructive:
    - When *mouse_capturer* is ``None`` the returned mouse vector is ``None``
      and the existing keyboard+gamepad values are unchanged.
    - When a :class:`MouseCapture` instance is provided, a 6-element float32
      array ``[x, y, lmb, rmb, mmb, scroll]`` is appended.

    Args:
        mouse_capturer: Optional MouseCapture instance (or None to skip).

    Returns:
        Tuple of (keyboard_output, gamepad_output, mouse_output_or_none)

    Raises:
        InputCaptureError: If input capture fails
    """
    try:
        # Capture keyboard
        keys = key_check() if key_check else []
        keyboard_output = keys_to_output(keys)

        # Capture gamepad (with fallback for missing gamepad)
        try:
            gamepad_keys = gamepad_check() if gamepad_check else []
            gamepad_output = (
                gamepad_keys_to_output(gamepad_keys) if gamepad_keys else []
            )
        except Exception:
            # Gamepad not available - use empty output
            gamepad_output = []

        # Capture mouse (additive – only when explicitly enabled)
        mouse_output = None
        if mouse_capturer is not None:
            try:
                state = mouse_capturer.snapshot()
                mouse_output = state.to_array()  # float32, shape (6,)
            except Exception:
                # Mouse capture failure must never break recording
                mouse_output = None

        return keyboard_output, gamepad_output, mouse_output

    except Exception as e:
        raise InputCaptureError(f"Failed to capture input: {e}")


def save_training_data(data: List, path: Path) -> bool:
    """
    Save training data to file with safety checks.

    Args:
        data: Training data to save
        path: Output file path

    Returns:
        True if save successful
    """
    try:
        # Create backup if file exists
        if path.exists():
            backup_path = path.with_suffix(".npy.bak")
            path.rename(backup_path)
            logger.info(f"Created backup: {backup_path}")

        # Save data – training samples are [screen_array, label_array] pairs
        # with different shapes, so we must use dtype=object + allow_pickle.
        arr = np.array(data, dtype=object)
        np.save(path, arr, allow_pickle=True)
        logger.info(f"Saved {len(data)} frames to {path}")

        # Verify save
        if not path.exists():
            raise IOError("Save verification failed - file not created")

        return True

    except Exception as e:
        logger.error(f"Failed to save training data: {e}")
        return False


def show_preview(screen: np.ndarray, window_name: str = "Recorder Preview") -> bool:
    """
    Show preview window if display is available.

    Args:
        screen: Screen image to display
        window_name: Window title

    Returns:
        False if user pressed 'q' to quit, True otherwise
    """
    try:
        preview = cv2.resize(screen, (640, 360))
        cv2.imshow(window_name, preview)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            return False
        return True

    except Exception:
        # Display not available (headless mode)
        return True


def cleanup():
    """Clean up resources."""
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def normalized_region_to_pixels(
    normalized_region: List[float], capture_region: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """Convert normalized profile ROI to absolute pixel rectangle."""
    x1, y1, x2, y2 = capture_region
    width = x2 - x1
    height = y2 - y1
    rx, ry, rw, rh = normalized_region
    left = x1 + int(rx * width)
    top = y1 + int(ry * height)
    right = left + int(rw * width)
    bottom = top + int(rh * height)
    return left, top, right, bottom


def pixel_region_to_normalized(
    pixel_region: List[int], capture_region: Tuple[int, int, int, int]
) -> List[float]:
    """Convert absolute pixel region [x1, y1, x2, y2] to normalized [x, y, w, h]."""
    x1, y1, x2, y2 = capture_region
    width = x2 - x1
    height = y2 - y1
    left, top, right, bottom = pixel_region
    return [
        round((left - x1) / width, 6),
        round((top - y1) / height, 6),
        round((right - left) / width, 6),
        round((bottom - top) / height, 6),
    ]


def load_roi_overrides(
    overrides_path: Path,
    capture_region: Tuple[int, int, int, int],
    base_regions: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    """
    Load ROI overrides from JSON and merge them into profile regions.

    Supported formats per region:
    - normalized list: [x, y, w, h]
    - pixel box object: {"pixels": [x1, y1, x2, y2]}
    """
    payload = json.loads(overrides_path.read_text(encoding="utf-8"))
    merged = dict(base_regions)

    for name, value in payload.items():
        if isinstance(value, list) and len(value) == 4:
            merged[name] = [float(part) for part in value]
        elif isinstance(value, dict) and "pixels" in value and len(value["pixels"]) == 4:
            merged[name] = pixel_region_to_normalized(value["pixels"], capture_region)
        else:
            raise ValueError(
                f"Invalid ROI override for '{name}'. Use [x,y,w,h] or {{\"pixels\": [x1,y1,x2,y2]}}."
            )

    return merged


def annotate_profile_regions(
    screen: np.ndarray,
    important_regions: Dict[str, List[float]],
    capture_region: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, Dict[str, Dict[str, int]]]:
    """Draw profile ROIs over a captured frame and return pixel metadata."""
    annotated = screen.copy()
    x1, y1, _, _ = capture_region
    metadata: Dict[str, Dict[str, int]] = {}

    for idx, (name, normalized) in enumerate(important_regions.items()):
        left, top, right, bottom = normalized_region_to_pixels(normalized, capture_region)
        local_left = max(0, left - x1)
        local_top = max(0, top - y1)
        local_right = max(0, right - x1)
        local_bottom = max(0, bottom - y1)
        color = (
            int(80 + (idx * 53) % 175),
            int(220 - (idx * 41) % 140),
            int(120 + (idx * 67) % 135),
        )
        cv2.rectangle(annotated, (local_left, local_top), (local_right, local_bottom), color, 2)
        cv2.putText(
            annotated,
            name,
            (local_left + 4, max(18, local_top + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        metadata[name] = {
            "x1": left,
            "y1": top,
            "x2": right,
            "y2": bottom,
            "width": right - left,
            "height": bottom - top,
        }

    return annotated, metadata


def save_profile_diagnostics(
    screen: np.ndarray,
    game_profile,
    capture_region: Tuple[int, int, int, int],
    out_dir: Path,
    important_regions: Optional[Dict[str, List[float]]] = None,
) -> Path:
    """Save raw capture, annotated capture, and JSON profile-region metadata."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    diag_dir = out_dir / "diagnostics" / game_profile.id / timestamp
    diag_dir.mkdir(parents=True, exist_ok=True)

    if important_regions is None:
        important_regions = game_profile.important_regions

    annotated, region_metadata = annotate_profile_regions(
        screen, important_regions, capture_region
    )
    cv2.imwrite(str(diag_dir / "raw.png"), screen)
    cv2.imwrite(str(diag_dir / "annotated_regions.png"), annotated)
    metadata = {
        "game_id": game_profile.id,
        "game_name": game_profile.name,
        "capture_region": list(capture_region),
        "typical_resolution": game_profile.typical_resolution,
        "normalized_regions": important_regions,
        "regions": region_metadata,
    }
    (diag_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    template = {
        name: {"pixels": [box["x1"], box["y1"], box["x2"], box["y2"]]}
        for name, box in region_metadata.items()
    }
    (diag_dir / "roi_overrides.template.json").write_text(
        json.dumps(template, indent=2), encoding="utf-8"
    )
    return diag_dir


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main data collection function.

    Args:
        argv: Command line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="BOT MMORPG - Data Collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  T     - Toggle pause/resume
  Q     - Quit (in preview window)

Example:
  python collect_data.py --out data/my_session
        """,
    )
    parser.add_argument(
        "--out", default="data/raw", help="Output folder for training data"
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Screen capture region (x1,y1,x2,y2). Auto-detected when --game is used.",
    )
    parser.add_argument(
        "--game",
        default=None,
        help=(
            "Game profile ID (e.g., dragon_ball_online). "
            "Auto-detects window region and sets output folder. "
            "Use with --task to select a specific task config."
        ),
    )
    parser.add_argument(
        "--task",
        default="farming",
        help="Task type from game profile (default: farming). Used with --game.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Frames per chunk file (default: 500)",
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="Disable preview window"
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="Enable mouse recording (additive – appends 6 values to action vector)",
    )
    parser.add_argument(
        "--diagnose-profile",
        action="store_true",
        help=(
            "Capture one frame and save raw/annotated profile-region diagnostics. "
            "Requires --game."
        ),
    )
    parser.add_argument(
        "--roi-overrides",
        default=None,
        help=(
            "Optional JSON file with per-region overrides for diagnostics. "
            "Each region may be [x,y,w,h] normalized or {\"pixels\": [x1,y1,x2,y2]}."
        ),
    )
    args = parser.parse_args(argv)

    # Environment variable fallback for --mouse (used by Tauri/launcher UI)
    if not args.mouse:
        import os

        env_mouse = os.environ.get("BOTMMO_CAPTURE_MOUSE", "").lower()
        if env_mouse == "true":
            args.mouse = True

    # --- Game profile integration ---
    game_profile = None
    if args.game:
        try:
            from ..config.profile_loader import GameProfileLoader
        except ImportError:
            try:
                import sys

                sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
                from bot_mmorpg.config.profile_loader import GameProfileLoader
            except ImportError:
                logger.warning(
                    "Could not load profile_loader; --game flag ignored. "
                    "Falling back to manual --region."
                )
                GameProfileLoader = None

        if GameProfileLoader is not None:
            try:
                loader = GameProfileLoader()
                game_profile = loader.load(args.game)
                logger.info(f"Loaded game profile: {game_profile.name}")

                # Auto-configure output directory
                if args.out in {"data/raw", "datasets"}:
                    args.out = build_game_output_dir(args.out, args.game, args.task)
                    logger.info(f"Output directory set to: {args.out}")

                # Log task config if available
                task_cfg = game_profile.get_task_config(args.task)
                if task_cfg:
                    logger.info(
                        f"Task '{args.task}': {task_cfg.description} "
                        f"(fps_target={task_cfg.fps_target})"
                    )

                # Enable mouse if the game profile requires it
                if game_profile.requires_mouse and not args.mouse:
                    logger.info(
                        "Game profile requires mouse. "
                        "Enable with --mouse for best results."
                    )
            except FileNotFoundError:
                logger.error(f"Game profile '{args.game}' not found.")
                logger.error(
                    "Available profiles are listed in game_profiles/index.yaml"
                )
                return 1

    # --- Auto-detect game window region ---
    if args.region is None and args.game:
        try:
            from .grabscreen import find_game_window
        except ImportError:
            try:
                from grabscreen import find_game_window
            except ImportError:
                find_game_window = None

        if find_game_window is not None:
            detected = find_game_window(args.game)
            if detected is not None:
                args.region = ",".join(str(v) for v in detected)
                logger.info(f"Auto-detected game window region: {args.region}")
            else:
                # Fallback to game profile typical resolution
                if game_profile is not None:
                    w, h = game_profile.typical_resolution
                    args.region = f"0,0,{w},{h}"
                    logger.warning(
                        f"Game window not found. Using profile resolution: {args.region}. "
                        "Position your game window at the top-left corner of your screen, "
                        "or specify --region manually."
                    )
                else:
                    args.region = "0,0,1280,720"
                    logger.warning(
                        "Game window not found. Using default 1280x720. "
                        "Specify --region manually for best results."
                    )

    # Final fallback for region
    if args.region is None:
        args.region = "0,40,1920,1120"

    # Parse region
    try:
        region = tuple(map(int, args.region.split(",")))
        if len(region) != 4:
            raise ValueError("Region must have 4 values")
    except ValueError as e:
        logger.error(f"Invalid region format: {e}")
        logger.error("Use format: x1,y1,x2,y2 (e.g., 0,40,1920,1120)")
        return 1

    # Validate dependencies
    try:
        validate_dependencies()
    except DataCollectionError as e:
        logger.error(str(e))
        return 1

    # Setup output directory
    out_dir = Path(args.out)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {out_dir.resolve()}")
    except Exception as e:
        logger.error(f"Cannot create output directory: {e}")
        return 1

    if args.diagnose_profile:
        if game_profile is None:
            logger.error("--diagnose-profile requires --game so profile regions are known.")
            return 1
        try:
            important_regions = game_profile.important_regions
            if args.roi_overrides:
                overrides_path = Path(args.roi_overrides)
                important_regions = load_roi_overrides(
                    overrides_path, region, game_profile.important_regions
                )
                logger.info(f"Loaded ROI overrides from: {overrides_path.resolve()}")
            screen = grab_screen(region=region)
            if screen is None or screen.size == 0:
                raise ScreenCaptureError("Screen capture returned empty image")
            diag_dir = save_profile_diagnostics(
                screen,
                game_profile,
                region,
                out_dir,
                important_regions=important_regions,
            )
            logger.info(f"Saved profile diagnostics to: {diag_dir.resolve()}")
            return 0
        except Exception as e:
            logger.error(f"Profile diagnostics failed: {e}")
            return 1

    # Find starting file index
    starting_value = 1
    while True:
        file_name = out_dir / f"training_data-{starting_value}.npy"
        if file_name.exists():
            starting_value += 1
        else:
            logger.info(f"Starting with file index: {starting_value}")
            break

    training_data = []
    paused = False
    frame_count = 0
    error_count = 0
    max_consecutive_errors = 10

    # --- Optional mouse capture (additive, non-destructive) ---
    mouse_capturer = None
    if args.mouse:
        if MouseCapture is None:
            logger.warning(
                "Mouse recording requested but pynput is not installed. "
                "Continuing without mouse. Install with: pip install pynput"
            )
        else:
            mouse_capturer = MouseCapture(capture_region=region)
            mouse_capturer.start()
            logger.info(
                "Mouse recording ENABLED (adds 6 values: x, y, lmb, rmb, mmb, scroll)"
            )

    # Countdown
    logger.info("Starting in...")
    for i in range(3, 0, -1):
        logger.info(f"  {i}...")
        time.sleep(1)

    logger.info("RECORDING STARTED! Press 'T' to pause, 'Q' to quit.")

    try:
        while True:
            if not paused:
                try:
                    # Capture screen
                    screen = capture_screen(region=region)

                    # Capture input (mouse is additive / non-destructive)
                    keyboard_output, gamepad_output, mouse_output = capture_input(
                        mouse_capturer=mouse_capturer
                    )

                    # Combine inputs – mouse appended only when enabled
                    parts = [np.array(keyboard_output)]
                    if gamepad_output:
                        parts.append(np.array(gamepad_output))
                    if mouse_output is not None:
                        parts.append(mouse_output)
                    final_output = np.concatenate(parts, axis=None)

                    training_data.append([screen, final_output])
                    frame_count += 1
                    error_count = 0  # Reset on success

                    # Show preview
                    if not args.no_preview:
                        if not show_preview(screen):
                            logger.info("User requested quit via preview window")
                            break

                    # Progress logging
                    if frame_count % 100 == 0:
                        logger.info(
                            f"Captured {frame_count} frames (buffer: {len(training_data)})"
                        )

                    # Save chunks
                    if len(training_data) >= args.chunk_size:
                        save_path = out_dir / f"training_data-{starting_value}.npy"
                        if save_training_data(training_data, save_path):
                            training_data = []
                            starting_value += 1
                        else:
                            logger.warning("Save failed, will retry on next chunk")

                except ScreenCaptureError as e:
                    error_count += 1
                    logger.warning(f"Screen capture error ({error_count}): {e}")

                except InputCaptureError as e:
                    error_count += 1
                    logger.warning(f"Input capture error ({error_count}): {e}")

                # Check for too many consecutive errors
                if error_count >= max_consecutive_errors:
                    logger.error(
                        f"Too many consecutive errors ({max_consecutive_errors}). Stopping."
                    )
                    break

            # Check for pause/quit
            try:
                keys = key_check() if key_check else []

                if "T" in keys:
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    logger.info(f"Recording {status}")
                    time.sleep(0.5)  # Debounce

                if "Q" in keys:
                    logger.info("User requested quit")
                    break

            except Exception:
                pass  # Ignore key check errors during pause check

            # Small delay to prevent CPU overload when paused
            if paused:
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")

    finally:
        # Save any remaining data
        if training_data:
            logger.info(f"Saving remaining {len(training_data)} frames...")
            save_path = out_dir / f"training_data-{starting_value}.npy"
            save_training_data(training_data, save_path)

        # Stop mouse capture if it was started
        if mouse_capturer is not None:
            mouse_capturer.stop()
            logger.info("Mouse capture stopped.")

        cleanup()
        logger.info(f"Data collection complete. Total frames: {frame_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
