"""Local supervised improvement with correction replay and measured promotion.

No capture or input imports here. Every candidate is evaluated against the same
incumbent on frozen, independent sessions. Scores measure imitation, not wins.
"""

import copy
import json
import math
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .actions import ActionSchema, MixedActionLoss, action_metrics
from .recordings import SessionDataset, atomic_json, fingerprint, read_sessions


class GameplayPolicy(nn.Module):
    """Small temporal RGB policy with explicit binary/continuous output channels."""

    def __init__(self, outputs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
        )
        self.memory = nn.GRU(288, 64, batch_first=True)
        self.output = nn.Linear(64, outputs)

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        batch, length = x.shape[:2]
        features = self.encoder(x.flatten(0, 1)).reshape(batch, length, -1)
        temporal, _ = self.memory(features)
        return self.output(temporal[:, -1])


def load_policy(path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("format") != "gameplay-policy-v1":
        raise ValueError("Not a learning-loop checkpoint")
    schema = ActionSchema.from_dict(state["schema"])
    model = GameplayPolicy(len(schema.names)).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state, schema


def evaluate(model, dataset, device="cpu", batch_size=32):
    model.eval()
    predictions, targets = [], []
    loss_sum, count = 0.0, 0
    criterion = MixedActionLoss(dataset.schema)
    with torch.no_grad():
        for frames, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            logits = model(frames.to(device))
            decoded = dataset.schema.decode(logits)
            loss_sum += float(criterion(logits, labels.to(device))) * len(labels)
            count += len(labels)
            predictions.append(decoded.cpu().numpy())
            targets.append(labels.numpy())
    result = action_metrics(
        np.concatenate(predictions), np.concatenate(targets), dataset.schema
    )
    result["loss"] = loss_sum / count
    return result


def should_promote(candidate, baseline, margin=0.005):
    """Require score gain AND no regression in button or continuous metrics."""
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("Promotion margin must be finite and positive")
    numbers = [
        candidate["score"],
        baseline["score"],
        candidate["button_f1"],
        baseline["button_f1"],
        candidate["button_exact"],
        baseline["button_exact"],
    ]
    numbers += list(candidate["continuous_mae"].values()) + list(
        baseline["continuous_mae"].values()
    )
    if not all(math.isfinite(v) for v in numbers):
        return False
    if candidate["samples"] != baseline["samples"] or candidate["samples"] <= 0:
        return False
    if candidate["score"] < baseline["score"] + margin:
        return False
    if (
        candidate["button_f1"] < baseline["button_f1"]
        or candidate["button_exact"] < baseline["button_exact"]
    ):
        return False
    if candidate["continuous_mae"].keys() != baseline["continuous_mae"].keys():
        return False
    return all(
        v <= baseline["continuous_mae"][k]
        for k, v in candidate["continuous_mae"].items()
    )


@contextmanager
def corpus_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "training.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            "Training lock exists; another cycle may be active"
        ) from None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def validate_corpus(sessions):
    if not sessions:
        raise ValueError("Record gameplay first")
    identity = {
        (s["game"], s["task"], json.dumps(s["schema"], sort_keys=True))
        for s in sessions
    }
    if len(identity) != 1:
        raise ValueError("Use one game, task and action schema per learning root")
    by_split = {name: [] for name in ("train", "validation", "test")}
    for s in sessions:
        if s["split"] not in by_split or s["role"] not in (
            "demonstration",
            "correction",
        ):
            raise ValueError("Invalid split/role")
        if s["role"] == "correction" and (
            s["split"] != "train" or not s.get("parent_model")
        ):
            raise ValueError("Correction data cannot be evaluation data")
        by_split[s["split"]].append(s)
    if not all(by_split.values()):
        raise ValueError("Need separate train, validation and test recording sessions")
    # Catch copied recordings even if NPZ metadata, names, or labels were changed.
    seen = {}
    for split, items in by_split.items():
        for s in items:
            with np.load(s["path"], allow_pickle=False) as data:
                import hashlib

                frames = data["frames"]
                digest = hashlib.sha256(frames.tobytes()).hexdigest()
            if digest in seen and seen[digest] != split:
                raise ValueError(
                    "Duplicate recording frames across evaluation/training splits"
                )
            seen[digest] = split
    return by_split


def run_cycle(
    root,
    epochs=5,
    batch_size=16,
    seq_len=4,
    learning_rate=0.0005,
    correction_weight=2.0,
    seed=42,
    margin=0.005,
    device="cpu",
    stop=None,
    on_epoch=None,
):
    """One finite update. Reinvoke after new demonstrations/corrections are saved."""
    if epochs < 1 or batch_size < 1 or seq_len < 1:
        raise ValueError("epochs, batch_size and seq_len must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid learning rate")
    if not math.isfinite(correction_weight) or not 1 <= correction_weight <= 10:
        raise ValueError("correction_weight must be between 1 and 10")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("Invalid promotion margin")
    root = Path(root)
    with corpus_lock(root):
        sessions = read_sessions(root)
        parts = validate_corpus(sessions)
        registry_path = root / "registry.json"
        registry = (
            json.loads(registry_path.read_text())
            if registry_path.exists()
            else {"version": 1, "history": [], "active": None}
        )
        signatures = {s["session_id"]: s["manifest_hash"] for s in sessions}
        old = registry.get("sessions", {})
        if any(signatures.get(k) != v for k, v in old.items()):
            raise ValueError("Previously used session was changed or removed")
        heldout = {
            s["session_id"]: s["manifest_hash"]
            for s in sessions
            if s["split"] != "train"
        }
        if "heldout" in registry and registry["heldout"] != heldout:
            raise ValueError(
                "Evaluation sessions are frozen; use a new learning root for a new benchmark"
            )
        if signatures == old:
            return {"status": "no_new_data", "active": registry["active"]}
        datasets = {
            name: SessionDataset(items, seq_len) for name, items in parts.items()
        }
        shapes = {data.arrays[0][0].shape[1:] for data in datasets.values()}
        if len(shapes) != 1:
            raise ValueError("Train and evaluation frame sizes differ")
        schema = datasets["train"].schema
        torch.manual_seed(seed)
        baseline = GameplayPolicy(len(schema.names)).to(device)
        parent = None
        if registry["active"]:
            active = registry["active"]
            model_path = root / active["path"]
            if fingerprint(model_path) != active["sha256"]:
                raise ValueError("Active checkpoint changed")
            baseline, metadata, prior_schema = load_policy(model_path, device)
            if prior_schema != schema or metadata["seq_len"] != seq_len:
                raise ValueError("Active model schema/sequence length mismatch")
            parent = active["sha256"]
        known_models = {
            h["candidate_sha256"]
            for h in registry["history"]
            if "candidate_sha256" in h
        }
        for s in sessions:
            if s["role"] == "correction" and s["parent_model"] not in known_models:
                raise ValueError("Correction refers to an unknown model")
        candidate = copy.deepcopy(baseline)
        criterion = MixedActionLoss(schema)
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=learning_rate)
        training = datasets["train"]
        weights = [
            correction_weight if training.sessions[j]["role"] == "correction" else 1.0
            for j, _ in training.windows
        ]
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights, len(weights), replacement=True, generator=generator
        )
        loader = DataLoader(training, batch_size=batch_size, sampler=sampler)
        run_id = uuid.uuid4().hex
        run_dir = root / "runs" / run_id
        run_dir.mkdir(parents=True)
        atomic_json(
            run_dir / "request.json",
            {
                "sessions": signatures,
                "parent_model": parent,
                "epochs": epochs,
                "seq_len": seq_len,
                "seed": seed,
                "learning_rate": learning_rate,
                "correction_weight": correction_weight,
                "margin": margin,
            },
        )
        best_state, best_score, history = None, -float("inf"), []
        try:
            for epoch in range(epochs):
                candidate.train()
                total, count = 0.0, 0
                for frames, labels in loader:
                    if stop is not None and stop.is_set():
                        raise InterruptedError(
                            "Training stopped; active model unchanged"
                        )
                    optimizer.zero_grad()
                    loss = criterion(candidate(frames.to(device)), labels.to(device))
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
                    optimizer.step()
                    total += loss.item() * len(labels)
                    count += len(labels)
                metrics = evaluate(
                    candidate, datasets["validation"], device, batch_size
                )
                history.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": total / count,
                        "validation": metrics,
                    }
                )
                if on_epoch:
                    on_epoch(history[-1])
                if metrics["score"] > best_score:
                    best_score = metrics["score"]
                    best_state = copy.deepcopy(candidate.state_dict())
            if stop is not None and stop.is_set():
                raise InterruptedError("Training stopped; active model unchanged")
            if best_state is None:
                raise ValueError("No finite candidate")
            candidate.load_state_dict(best_state)
            baseline_metrics = evaluate(baseline, datasets["test"], device, batch_size)
            candidate_metrics = evaluate(
                candidate, datasets["test"], device, batch_size
            )
            promoted = should_promote(candidate_metrics, baseline_metrics, margin)
            checkpoint = run_dir / "candidate.pth"
            torch.save(
                {
                    "format": "gameplay-policy-v1",
                    "model_state_dict": candidate.state_dict(),
                    "schema": schema.to_dict(),
                    "seq_len": seq_len,
                    "game": sessions[0]["game"],
                    "task": sessions[0]["task"],
                    "frame_shape": list(training.arrays[0][0].shape[1:]),
                    "sessions": signatures,
                    "parent_model": parent,
                },
                checkpoint,
            )
            digest = fingerprint(checkpoint)
            report = {
                "status": "promoted" if promoted else "rejected",
                "run_id": run_id,
                "candidate_sha256": digest,
                "candidate_path": str(checkpoint.relative_to(root)),
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "epochs": history,
                "parent_model": parent,
                "evidence": "held-out imitation; gameplay outcomes not measured",
            }
            atomic_json(run_dir / "report.json", report)
            if stop is not None and stop.is_set():
                raise InterruptedError("Training stopped before promotion")
            registry.update(sessions=signatures, heldout=heldout)
            registry["history"].append(
                {k: v for k, v in report.items() if k != "epochs"}
            )
            if promoted:
                registry["active"] = {
                    "path": str(checkpoint.relative_to(root)),
                    "sha256": digest,
                }
            atomic_json(registry_path, registry)
            return report
        except BaseException as exc:
            atomic_json(
                run_dir / "failure.json", {"error": str(exc), "epochs": history}
            )
            raise
