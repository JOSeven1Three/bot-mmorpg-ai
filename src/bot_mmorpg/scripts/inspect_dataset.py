"""
Inspect recorded gameplay datasets.

Lists chunk files, frame counts, action-vector shapes, and can export a few
sample frames so captures are easy to verify visually.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np


def find_training_files(root: Path) -> List[Path]:
    """Find training_data-*.npy files under a root folder."""
    return sorted(root.rglob("training_data-*.npy"))


def summarize_chunk(path: Path) -> dict:
    """Load one chunk and return lightweight summary information."""
    data = np.load(path, allow_pickle=True)
    frame_count = len(data)
    frame_shape = None
    action_shape = None

    if frame_count > 0:
        frame_shape = tuple(data[0][0].shape)
        action_shape = tuple(np.array(data[0][1]).shape)

    return {
        "path": path,
        "frames": frame_count,
        "frame_shape": frame_shape,
        "action_shape": action_shape,
    }


def export_sample_frames(
    training_files: Iterable[Path],
    output_dir: Path,
    samples_per_file: int = 2,
) -> int:
    """Export sample frames from dataset chunks as PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for chunk in training_files:
        data = np.load(chunk, allow_pickle=True)
        if len(data) == 0:
            continue

        indices = np.linspace(0, len(data) - 1, num=min(samples_per_file, len(data)), dtype=int)
        for idx in indices:
            frame = data[idx][0]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out_path = output_dir / f"{chunk.stem}-sample-{idx}.png"
            cv2.imwrite(str(out_path), frame_bgr)
            written += 1

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect recorded BOT-MMORPG datasets")
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="datasets",
        help="Dataset root folder to inspect",
    )
    parser.add_argument(
        "--export-samples",
        action="store_true",
        help="Export a few sample frames as PNG files",
    )
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=2,
        help="Number of sample frames to export per chunk file",
    )
    args = parser.parse_args(argv)

    root = Path(args.dataset_root)
    if not root.exists():
        print(f"[Error] Dataset root not found: {root.resolve()}")
        return 1

    training_files = find_training_files(root)
    if not training_files:
        print(f"No training_data-*.npy files found under: {root.resolve()}")
        print("If you just recorded, make sure the capture was stopped cleanly with Q or Ctrl+C.")
        return 0

    print(f"Dataset root: {root.resolve()}")
    print(f"Chunk files: {len(training_files)}")
    print("-" * 60)

    total_frames = 0
    summaries = [summarize_chunk(path) for path in training_files]
    for summary in summaries:
        total_frames += summary["frames"]
        print(
            f"{summary['path']}: frames={summary['frames']} "
            f"frame_shape={summary['frame_shape']} action_shape={summary['action_shape']}"
        )

    print("-" * 60)
    print(f"Total frames: {total_frames}")

    if args.export_samples:
        output_dir = root / "inspector_samples"
        written = export_sample_frames(
            training_files,
            output_dir,
            samples_per_file=max(1, args.samples_per_file),
        )
        print(f"Exported {written} sample frame(s) to: {output_dir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
