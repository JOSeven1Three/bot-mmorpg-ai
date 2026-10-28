"""Explicit mixed action contracts shared by training, scoring and inference."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ActionSchema:
    names: tuple
    kinds: tuple  # binary, unit [0,1], signed [-1,1], real

    def __post_init__(self):
        if not self.names or len(self.names) != len(self.kinds):
            raise ValueError("Action names and kinds must have equal, nonzero lengths")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Duplicate action names")
        if set(self.kinds) - {"binary", "unit", "signed", "real"}:
            raise ValueError("Unknown action kind")

    def to_dict(self):
        return {"names": list(self.names), "kinds": list(self.kinds)}

    @classmethod
    def from_dict(cls, value):
        return cls(tuple(value["names"]), tuple(value["kinds"]))

    def validate(self, values):
        a = np.asarray(values)
        if a.ndim != 2 or a.shape[1] != len(self.names) or not np.isfinite(a).all():
            raise ValueError("Nonfinite or mismatched action data")
        for i, kind in enumerate(self.kinds):
            v = a[:, i]
            if kind == "binary" and not np.isin(v, [0, 1]).all():
                raise ValueError(f"{self.names[i]} must contain binary labels")
            if kind in ("binary", "unit") and ((v < 0) | (v > 1)).any():
                raise ValueError(f"{self.names[i]} outside [0,1]")
            if kind == "signed" and (np.abs(v) > 1).any():
                raise ValueError(f"{self.names[i]} outside [-1,1]")

    @classmethod
    def legacy(cls, width):
        # Legacy recorder has 9 movement classes, 20 gamepad channels,
        # and optionally six old or ten current mouse channels.
        base = 29 if width in (29, 35, 39) else 9
        if width not in (9, 15, 19, 29, 35, 39):
            raise ValueError(
                f"Unknown legacy action width {width}; explicit schema required"
            )
        names = [f"action_{i}" for i in range(base)]
        kinds = ["binary"] * base
        if base == 29:
            kinds[9:15] = ["unit", "unit"] + ["signed"] * 4
        if width > base:
            extended = width - base == 10
            names += ["mouse_x", "mouse_y"]
            kinds += ["unit", "unit"]
            if extended:
                names += ["mouse_dx", "mouse_dy", "mouse_vx", "mouse_vy"]
                kinds += ["signed"] * 4
            names += ["mouse_left", "mouse_right", "mouse_middle", "scroll"]
            kinds += ["binary"] * 3 + ["real"]
        return cls(tuple(names), tuple(kinds))

    def decode(self, logits):
        if logits.shape[-1] != len(self.names):
            raise ValueError("Model output does not match action schema")
        return torch.stack(
            [
                torch.sigmoid(logits[..., i])
                if k in ("binary", "unit")
                else torch.tanh(logits[..., i])
                if k == "signed"
                else logits[..., i]
                for i, k in enumerate(self.kinds)
            ],
            dim=-1,
        )


class MixedActionLoss(nn.Module):
    def __init__(self, schema):
        super().__init__()
        self.schema = schema

    def forward(self, logits, targets):
        decoded = self.schema.decode(logits)
        binary = [i for i, k in enumerate(self.schema.kinds) if k == "binary"]
        continuous = [i for i, k in enumerate(self.schema.kinds) if k != "binary"]
        loss = logits.sum() * 0
        if binary:
            loss = loss + nn.functional.binary_cross_entropy_with_logits(
                logits[:, binary], targets[:, binary]
            )
        if continuous:
            loss = loss + nn.functional.smooth_l1_loss(
                decoded[:, continuous], targets[:, continuous]
            )
        return loss


def action_metrics(predictions, targets, schema):
    """Scores decoded outputs; never compares continuous coordinates as bits."""
    p, y = np.asarray(predictions), np.asarray(targets)
    schema.validate(y)
    if p.shape != y.shape or not len(y) or not np.isfinite(p).all():
        raise ValueError("Empty, nonfinite or mismatched predictions")
    bits = [i for i, k in enumerate(schema.kinds) if k == "binary"]
    continuous = [i for i, k in enumerate(schema.kinds) if k != "binary"]
    result = {"samples": len(y)}
    if bits:
        pb, yb = p[:, bits] >= 0.5, y[:, bits].astype(bool)
        tp = (pb & yb).sum(axis=0)
        fp = (pb & ~yb).sum(axis=0)
        fn = (~pb & yb).sum(axis=0)
        denom = 2 * tp + fp + fn
        active = denom > 0
        result["button_f1"] = (
            float(np.mean(2 * tp[active] / denom[active])) if active.any() else 1.0
        )
        result["button_exact"] = float((pb == yb).all(axis=1).mean())
    else:
        result.update(button_f1=1.0, button_exact=1.0)
    errors = {
        schema.names[i]: float(np.abs(p[:, i] - y[:, i]).mean()) for i in continuous
    }
    result["continuous_mae"] = errors
    # Each bounded channel has comparable unit-range error; scroll stays separately reported.
    result["score"] = result["button_f1"] - (
        float(np.mean(list(errors.values()))) if errors else 0.0
    )
    return result
