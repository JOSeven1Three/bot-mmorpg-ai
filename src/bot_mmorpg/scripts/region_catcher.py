"""
Interactive HUD region editor for game profiles.

Opens a screenshot or live game capture, lets the user move and resize
profile regions, and writes the updated regions back into profile.yaml.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ..config.profile_loader import GameProfileLoader
    from .collect_data import (
        annotate_profile_regions,
        load_roi_overrides,
        normalized_region_to_pixels,
        pixel_region_to_normalized,
    )
    from .grabscreen import find_game_window, grab_screen
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from bot_mmorpg.config.profile_loader import GameProfileLoader
    from bot_mmorpg.scripts.collect_data import (
        annotate_profile_regions,
        load_roi_overrides,
        normalized_region_to_pixels,
        pixel_region_to_normalized,
    )
    from bot_mmorpg.scripts.grabscreen import find_game_window, grab_screen


WINDOW_NAME = "BOT-MMORPG Region Catcher"
MIN_SIZE = 12


@dataclass
class DragState:
    region_name: Optional[str] = None
    mode: str = "move"
    anchor_x: int = 0
    anchor_y: int = 0
    original_box: Optional[Tuple[int, int, int, int]] = None


def clamp_box(
    box: Tuple[int, int, int, int], capture_region: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """Clamp a pixel box into the capture region."""
    x1, y1, x2, y2 = capture_region
    left, top, right, bottom = box

    left = max(x1, min(left, x2 - MIN_SIZE))
    top = max(y1, min(top, y2 - MIN_SIZE))
    right = max(left + MIN_SIZE, min(right, x2))
    bottom = max(top + MIN_SIZE, min(bottom, y2))
    return left, top, right, bottom


def move_box(
    box: Tuple[int, int, int, int],
    dx: int,
    dy: int,
    capture_region: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """Move a pixel box while keeping its size."""
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    x1, y1, x2, y2 = capture_region

    new_left = max(x1, min(left + dx, x2 - width))
    new_top = max(y1, min(top + dy, y2 - height))
    return new_left, new_top, new_left + width, new_top + height


def resize_box(
    box: Tuple[int, int, int, int],
    dw: int,
    dh: int,
    capture_region: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """Resize a box from its bottom-right corner."""
    left, top, right, bottom = box
    return clamp_box((left, top, right + dw, bottom + dh), capture_region)


def nearest_corner(
    box: Tuple[int, int, int, int], x: int, y: int
) -> str:
    """Return the nearest corner name for resize dragging."""
    left, top, right, bottom = box
    corners = {
        "tl": (left, top),
        "tr": (right, top),
        "bl": (left, bottom),
        "br": (right, bottom),
    }
    return min(corners, key=lambda name: (corners[name][0] - x) ** 2 + (corners[name][1] - y) ** 2)


def apply_corner_resize(
    original_box: Tuple[int, int, int, int],
    corner: str,
    current_x: int,
    current_y: int,
    capture_region: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """Resize a box using the chosen drag corner."""
    left, top, right, bottom = original_box

    if corner == "tl":
        box = (current_x, current_y, right, bottom)
    elif corner == "tr":
        box = (left, current_y, current_x, bottom)
    elif corner == "bl":
        box = (current_x, top, right, current_y)
    else:
        box = (left, top, current_x, current_y)

    return clamp_box(box, capture_region)


def point_inside_box(box: Tuple[int, int, int, int], x: int, y: int) -> bool:
    left, top, right, bottom = box
    return left <= x <= right and top <= y <= bottom


def write_regions(
    loader: GameProfileLoader,
    game_id: str,
    regions: Dict[str, List[float]],
) -> None:
    """Persist region updates to the profile file."""
    loader.save_display_regions(game_id, regions)


def render_editor(
    base_image: np.ndarray,
    capture_region: Tuple[int, int, int, int],
    regions: Dict[str, List[float]],
    selected_name: str,
    step: int,
) -> np.ndarray:
    """Draw interactive overlay for the editor."""
    annotated, _ = annotate_profile_regions(base_image, regions, capture_region)
    x1, y1, _, _ = capture_region
    selected_box = normalized_region_to_pixels(regions[selected_name], capture_region)
    local_box = (
        selected_box[0] - x1,
        selected_box[1] - y1,
        selected_box[2] - x1,
        selected_box[3] - y1,
    )
    cv2.rectangle(
        annotated,
        (local_box[0], local_box[1]),
        (local_box[2], local_box[3]),
        (0, 255, 255),
        3,
    )
    help_lines = [
        f"Selected: {selected_name}",
        "Drag inside box to move, drag near a corner to resize",
        f"WASD move | IJKL resize | TAB/N next | P prev | R refresh | step={step}",
        "O save now | +/- step | ESC/Q quit",
    ]
    for idx, line in enumerate(help_lines):
        cv2.putText(
            annotated,
            line,
            (20, 32 + idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def parse_region(region_text: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not region_text:
        return None
    values = tuple(int(part) for part in region_text.split(","))
    if len(values) != 4:
        raise ValueError("Region must be x1,y1,x2,y2")
    return values


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive profile region editor")
    parser.add_argument("--game", required=True, help="Game profile id, e.g. diablo_4")
    parser.add_argument("--screenshot", default=None, help="Optional screenshot path")
    parser.add_argument("--region", default=None, help="Manual capture region x1,y1,x2,y2")
    parser.add_argument(
        "--roi-overrides",
        default=None,
        help="Optional ROI overrides JSON to load before editing",
    )
    parser.add_argument(
        "--no-autosave",
        action="store_true",
        help="Disable write-through on every change; writes only on explicit save keys.",
    )
    args = parser.parse_args(argv)

    loader = GameProfileLoader()
    profile = loader.load(args.game)

    capture_region = parse_region(args.region)
    if capture_region is None:
        capture_region = find_game_window(args.game)
    if capture_region is None:
        w, h = profile.typical_resolution
        capture_region = (0, 0, w, h)

    regions = dict(profile.important_regions)
    if args.roi_overrides:
        regions = load_roi_overrides(Path(args.roi_overrides), capture_region, regions)

    if args.screenshot:
        base_image = cv2.imread(args.screenshot)
        if base_image is None:
            raise FileNotFoundError(f"Could not read screenshot: {args.screenshot}")
    else:
        base_image = grab_screen(region=capture_region)

    region_names = list(regions.keys())
    selected_idx = 0
    step = 2
    drag = DragState()

    def persist():
        if not args.no_autosave:
            write_regions(loader, args.game, regions)

    def on_mouse(event, x, y, _flags, _param):
        nonlocal drag
        global_x = x + capture_region[0]
        global_y = y + capture_region[1]
        selected_name = region_names[selected_idx]
        selected_box = normalized_region_to_pixels(regions[selected_name], capture_region)

        if event == cv2.EVENT_LBUTTONDOWN and point_inside_box(selected_box, global_x, global_y):
            corner = nearest_corner(selected_box, global_x, global_y)
            cx, cy = {
                "tl": (selected_box[0], selected_box[1]),
                "tr": (selected_box[2], selected_box[1]),
                "bl": (selected_box[0], selected_box[3]),
                "br": (selected_box[2], selected_box[3]),
            }[corner]
            near_corner = abs(cx - global_x) <= 18 and abs(cy - global_y) <= 18
            drag = DragState(
                region_name=selected_name,
                mode=corner if near_corner else "move",
                anchor_x=global_x,
                anchor_y=global_y,
                original_box=selected_box,
            )
        elif event == cv2.EVENT_MOUSEMOVE and drag.region_name and drag.original_box:
            if drag.mode == "move":
                box = move_box(
                    drag.original_box,
                    global_x - drag.anchor_x,
                    global_y - drag.anchor_y,
                    capture_region,
                )
            else:
                box = apply_corner_resize(
                    drag.original_box, drag.mode, global_x, global_y, capture_region
                )
            regions[drag.region_name] = pixel_region_to_normalized(list(box), capture_region)
            persist()
        elif event == cv2.EVENT_LBUTTONUP:
            drag = DragState()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        selected_name = region_names[selected_idx]
        frame = render_editor(base_image, capture_region, regions, selected_name, step)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord("q")):
            break
        if key in (9, ord("n")):
            selected_idx = (selected_idx + 1) % len(region_names)
        elif key == ord("p"):
            selected_idx = (selected_idx - 1) % len(region_names)
        elif key == ord("r") and not args.screenshot:
            base_image = grab_screen(region=capture_region)
        elif key == ord("o"):
            write_regions(loader, args.game, regions)
        elif key == ord("="):
            step = min(step + 1, 50)
        elif key == ord("-"):
            step = max(step - 1, 1)
        elif key in (ord("w"), ord("a"), ord("s"), ord("d"), ord("i"), ord("j"), ord("k"), ord("l")):
            box = normalized_region_to_pixels(regions[selected_name], capture_region)
            if key == ord("w"):
                box = move_box(box, 0, -step, capture_region)
            elif key == ord("a"):
                box = move_box(box, -step, 0, capture_region)
            elif key == ord("s"):
                box = move_box(box, 0, step, capture_region)
            elif key == ord("d"):
                box = move_box(box, step, 0, capture_region)
            elif key == ord("i"):
                box = resize_box(box, 0, -step, capture_region)
            elif key == ord("k"):
                box = resize_box(box, 0, step, capture_region)
            elif key == ord("j"):
                box = resize_box(box, -step, 0, capture_region)
            elif key == ord("l"):
                box = resize_box(box, step, 0, capture_region)
            regions[selected_name] = pixel_region_to_normalized(list(box), capture_region)
            persist()

    cv2.destroyAllWindows()
    write_regions(loader, args.game, regions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
