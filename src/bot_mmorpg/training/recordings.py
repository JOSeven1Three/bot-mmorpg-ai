"""Immutable, pickle-free sessions and sequence-safe legacy dataset loading."""

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from .actions import ActionSchema


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_session(
    root,
    frames,
    actions,
    schema,
    *,
    role="demonstration",
    split="train",
    game="unknown",
    task="default",
    parent_model=None,
    timestamps=None,
):
    if role not in ("demonstration", "correction") or split not in (
        "train",
        "validation",
        "test",
    ):
        raise ValueError("Invalid session role/split")
    if role == "correction" and (split != "train" or not parent_model):
        raise ValueError("Corrections require parent model provenance and train split")
    frames, actions = np.asarray(frames), np.asarray(actions, dtype=np.float32)
    schema.validate(actions)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("Frames must be uint8 RGB [N,H,W,3]")
    if len(frames) != len(actions) or len(frames) == 0:
        raise ValueError("Empty or mismatched frames/actions")
    if timestamps is None:
        timestamps = np.arange(len(frames), dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if (
        timestamps.shape != (len(frames),)
        or not np.isfinite(timestamps).all()
        or (np.diff(timestamps) <= 0).any()
    ):
        raise ValueError("Timestamps must be finite and strictly increasing")
    session_id = uuid.uuid4().hex
    directory = Path(root) / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=False)
    data_path = directory / "samples.npz"
    np.savez_compressed(
        data_path, frames=frames, actions=actions, timestamps=timestamps
    )
    # Manifest is published last; interrupted writes are not eligible training data.
    metadata = {
        "version": 1,
        "session_id": session_id,
        "schema": schema.to_dict(),
        "role": role,
        "split": split,
        "game": game,
        "task": task,
        "parent_model": parent_model,
        "samples": len(frames),
        "sha256": fingerprint(data_path),
    }
    atomic_json(directory / "session.json", metadata)
    return directory


def read_sessions(root):
    result = []
    for path in sorted((Path(root) / "sessions").glob("*/session.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        data_path = path.parent / "samples.npz"
        if meta.get("version") != 1 or meta.get("session_id") != path.parent.name:
            raise ValueError(f"Invalid session manifest: {path}")
        if fingerprint(data_path) != meta["sha256"]:
            raise ValueError(f"Session modified: {path}")
        meta["manifest_hash"] = fingerprint(path)
        meta["path"] = str(data_path)
        result.append(meta)
    return result


class SessionDataset(Dataset):
    def __init__(self, sessions, seq_len=1):
        if seq_len < 1 or not sessions:
            raise ValueError("Need sessions and positive sequence length")
        self.seq_len, self.sessions = seq_len, list(sessions)
        self.schema = ActionSchema.from_dict(sessions[0]["schema"])
        self.num_actions = len(self.schema.names)
        self.arrays, self.windows = [], []
        shape = None
        for j, meta in enumerate(sessions):
            if ActionSchema.from_dict(meta["schema"]) != self.schema:
                raise ValueError("Mixed action schemas")
            with np.load(meta["path"], allow_pickle=False) as data:
                frames, actions = data["frames"], data["actions"]
                stamps = data["timestamps"]
            self.schema.validate(actions)
            if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError("Invalid RGB frame array")
            if len(frames) != len(actions) or len(frames) != meta["samples"]:
                raise ValueError("Session sample count mismatch")
            if (
                stamps.shape != (len(frames),)
                or not np.isfinite(stamps).all()
                or (np.diff(stamps) <= 0).any()
            ):
                raise ValueError("Invalid session timestamps")
            if shape is not None and shape != frames.shape[1:]:
                raise ValueError("Mixed frame sizes")
            shape = frames.shape[1:]
            self.arrays.append((frames, actions))
            self.windows.extend((j, i) for i in range(len(frames) - seq_len + 1))
        if not self.windows:
            raise ValueError("No complete temporal sequences")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        session, start = self.windows[index]
        frames, actions = self.arrays[session]
        x = (
            torch.from_numpy(frames[start : start + self.seq_len].copy())
            .permute(0, 3, 1, 2)
            .float()
            / 255
        )
        return (
            x[0] if self.seq_len == 1 else x,
            torch.from_numpy(actions[start + self.seq_len - 1].copy()).float(),
        )


class LegacyGameplayDataset(Dataset):
    """Trusted local .npy compatibility; chunks never form cross-file sequences.

    Sidecars bind chunks from one recording to one split group. Unknown legacy
    chunks in a directory are conservatively treated as a single recording.
    """

    def __init__(
        self,
        data_dir,
        seq_len=1,
        transform=None,
        limit_files=None,
        file_pattern="training_data-*.npy",
    ):
        self.data_dir, self.seq_len, self.transform = Path(data_dir), seq_len, transform
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        self.files = sorted(
            set(self.data_dir.rglob(file_pattern))
            | set(self.data_dir.rglob("preprocessed_training_data-*.npy"))
        )
        if limit_files:
            self.files = self.files[:limit_files]
        self.arrays, self.windows, self.groups = [], [], []
        width, shape = None, None
        for path in self.files:
            # Legacy object arrays are only for recordings made on a trusted machine.
            from bot_mmorpg.utils.secure_loader import load_training_data_secure

            data = load_training_data_secure(path, validate=True, allow_untrusted=True)
            frames = np.asarray([item[0] for item in data])
            actions = np.asarray([item[1] for item in data], dtype=np.float32)
            if not len(frames) or frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError(f"Invalid recording: {path}")
            if actions.ndim != 2 or not np.isfinite(actions).all():
                raise ValueError(f"Invalid actions: {path}")
            if width is not None and (
                width != actions.shape[1] or shape != frames.shape[1:]
            ):
                raise ValueError(
                    "Mixed action widths or frame sizes; split incompatible recordings"
                )
            width, shape = actions.shape[1], frames.shape[1:]
            sidecar = path.with_suffix(".session.json")
            group = str(path.parent.resolve())
            if sidecar.exists():
                group = json.loads(sidecar.read_text(encoding="utf-8"))["session_id"]
            j = len(self.arrays)
            self.arrays.append((frames, actions))
            self.groups.append(group)
            self.windows.extend((j, i) for i in range(len(frames) - seq_len + 1))
        if not self.windows:
            raise ValueError("No complete recording sequences found")
        self.num_actions = width
        self.schema = ActionSchema.legacy(width)

    __len__ = SessionDataset.__len__
    __getitem__ = SessionDataset.__getitem__

    def split(self, fraction=0.1, seed=42):
        if not 0 < fraction < 1:
            raise ValueError("Validation fraction must be between 0 and 1")
        groups = sorted(set(self.groups))
        if len(groups) < 2:
            raise ValueError(
                "Need at least two independent recording sessions for validation; record another session"
            )
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
        validation = set(
            groups[: max(1, min(len(groups) - 1, round(len(groups) * fraction)))]
        )
        train, val = [], []
        for i, (j, _) in enumerate(self.windows):
            (val if self.groups[j] in validation else train).append(i)
        if not train or not val:
            raise ValueError("Each split needs at least one complete sequence")
        return Subset(self, train), Subset(self, val)
