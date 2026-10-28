"""Offline PyTorch evaluation using the same schemas/preprocessing as training."""

import argparse
import json
from pathlib import Path

import torch

from bot_mmorpg.training.actions import ActionSchema
from bot_mmorpg.training.learning_loop import evaluate, load_policy
from bot_mmorpg.training.recordings import (
    LegacyGameplayDataset,
    SessionDataset,
    read_sessions,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="PyTorch checkpoint (.pth)")
    p.add_argument(
        "--model-dir",
        help="Directory containing exactly one *_best.pth or candidate.pth",
    )
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--output")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--max-batches", type=int, help="Deprecated; evaluation always uses all samples"
    )
    args = p.parse_args(argv)
    try:
        if bool(args.model) == bool(args.model_dir):
            raise ValueError("Specify exactly one of --model or --model-dir")
        if args.model_dir:
            choices = list(Path(args.model_dir).glob("*_best.pth")) + list(
                Path(args.model_dir).glob("candidate.pth")
            )
            if len(choices) != 1:
                raise ValueError(
                    "Ambiguous/missing checkpoint; pass --model explicitly"
                )
            path = choices[0]
        else:
            path = Path(args.model)
        meta = torch.load(path, map_location="cpu", weights_only=True)
        if meta.get("format") == "gameplay-policy-v1":
            model, meta, schema = load_policy(path)
            sessions = [
                s for s in read_sessions(args.dataset_dir) if s["split"] == args.split
            ]
            if any(
                s["session_id"] in meta.get("sessions", {})
                and s["manifest_hash"] != meta["sessions"][s["session_id"]]
                for s in sessions
            ):
                raise ValueError("Evaluation session changed since training")
            data = SessionDataset(sessions, meta["seq_len"])
            if data.schema != schema:
                raise ValueError("Checkpoint/dataset action schema mismatch")
        else:
            from bot_mmorpg.scripts.models_pytorch import load_model

            model, meta = load_model(str(path), device=torch.device("cpu"))
            data = LegacyGameplayDataset(
                args.dataset_dir, seq_len=meta.get("temporal_frames", 1)
            )
            if meta.get("action_schema") and data.schema != ActionSchema.from_dict(
                meta["action_schema"]
            ):
                raise ValueError("Checkpoint/dataset action schema mismatch")
        result = evaluate(model, data, batch_size=args.batch_size)
        result["evidence"] = "Offline imitation metrics; not gameplay win rate"
        output = json.dumps(result, indent=2, allow_nan=False)
        print(output)
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Evaluation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
