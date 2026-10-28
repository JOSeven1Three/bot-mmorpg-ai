"""Record, correct, retrain, evaluate and inspect a local gameplay policy.

Run ``python -m bot_mmorpg.scripts.learning --help``. Recording and shadow
prediction send no game input. Live execution requires a separate --live flag.
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np


def profile_schema(game):
    from bot_mmorpg.config.profile_loader import GameProfileLoader
    from bot_mmorpg.training.actions import ActionSchema

    profile = GameProfileLoader().load(game)
    bindings = profile.default_bindings
    names = list(dict.fromkeys(b["key"].upper() for b in bindings))
    if not names:
        raise ValueError("Profile needs explicit default_bindings")
    # Recording/control hotkeys are reserved, never model outputs.
    if any(k in names for k in ("F8", "F9", "F12")):
        raise ValueError("F8/F9/F12 are reserved")
    names += ["MOUSE_X", "MOUSE_Y"]
    return profile, ActionSchema(
        tuple(names), tuple(["binary"] * (len(names) - 2) + ["unit", "unit"])
    )


class WindowIO:
    """One exact foreground client window. Lazy platform imports; no import-time input."""

    def __init__(self, title, schema, live=False):
        import ctypes

        import mss
        import win32api
        import win32gui

        # Obtain physical client coordinates before storing the geometry guard.
        ctypes.windll.user32.SetProcessDPIAware()
        self.api, self.gui = win32api, win32gui
        matches = []
        win32gui.EnumWindows(
            lambda h, _: (
                matches.append(h)
                if win32gui.IsWindowVisible(h) and win32gui.GetWindowText(h) == title
                else None
            ),
            None,
        )
        if len(matches) != 1:
            raise ValueError(
                "Need exactly one visible window with the exact supplied title"
            )
        self.hwnd, self.title, self.schema = matches[0], title, schema
        self.rect = self.geometry()
        self.live, self.held = live, set()
        self.key_codes = {
            "SPACE": 32,
            "SHIFT": 16,
            "CTRL": 17,
            "ALT": 18,
            "TAB": 9,
            "ESC": 27,
            "ENTER": 13,
            "MOUSE_LEFT": 1,
            "MOUSE_RIGHT": 2,
            "MOUSE_MIDDLE": 4,
        }
        for name in schema.names[:-2]:
            if name not in self.key_codes and len(name) == 1 and name.isascii():
                self.key_codes[name] = ord(name)
            if name not in self.key_codes:
                raise ValueError(f"Unsupported key binding: {name}")
        self.keyboard = self.mouse = None
        if live:
            from pynput.keyboard import Controller, Key
            from pynput.mouse import Button
            from pynput.mouse import Controller as MouseController

            self.keyboard, self.mouse = Controller(), MouseController()
            self.special = {
                "SPACE": Key.space,
                "SHIFT": Key.shift,
                "CTRL": Key.ctrl,
                "ALT": Key.alt,
                "TAB": Key.tab,
                "ESC": Key.esc,
                "ENTER": Key.enter,
            }
            self.buttons = {
                "MOUSE_LEFT": Button.left,
                "MOUSE_RIGHT": Button.right,
                "MOUSE_MIDDLE": Button.middle,
            }
        self.capture = mss.mss()

    def geometry(self):
        left, top = self.gui.ClientToScreen(self.hwnd, (0, 0))
        _, _, w, h = self.gui.GetClientRect(self.hwnd)
        if w <= 0 or h <= 0:
            raise RuntimeError("Invalid window geometry")
        return left, top, w, h

    def healthy(self):
        return (
            self.gui.IsWindow(self.hwnd)
            and not self.gui.IsIconic(self.hwnd)
            and self.gui.GetWindowText(self.hwnd) == self.title
            and self.gui.GetForegroundWindow() == self.hwnd
            and self.geometry() == self.rect
        )

    def down(self, key):
        return bool(self.api.GetAsyncKeyState(key) & 0x8000)

    def observe(self):
        import cv2

        if not self.healthy():
            raise RuntimeError("Game focus or geometry changed")
        left, top, w, h = self.rect
        raw = np.asarray(
            self.capture.grab({"left": left, "top": top, "width": w, "height": h})
        )
        frame = cv2.cvtColor(cv2.resize(raw[:, :, :3], (160, 90)), cv2.COLOR_BGR2RGB)
        x, y = self.api.GetCursorPos()
        values = [float(self.down(self.key_codes[k])) for k in self.schema.names[:-2]]
        values += [
            float(np.clip((x - left) / w, 0, 1)),
            float(np.clip((y - top) / h, 0, 1)),
        ]
        if not self.healthy():
            raise RuntimeError("Game focus or geometry changed during capture")
        return frame, np.asarray(values, dtype=np.float32)

    def _button(self, name, press):
        if name in self.buttons:
            (self.mouse.press if press else self.mouse.release)(self.buttons[name])
        else:
            key = self.special.get(name, name.lower())
            (self.keyboard.press if press else self.keyboard.release)(key)

    def release(self):
        errors = []
        for name in list(self.held):
            try:
                self._button(name, False)
                self.held.remove(name)
            except Exception as exc:  # noqa: BLE001 - attempt every release before reporting failure
                errors.append(exc)
        if errors:
            raise RuntimeError("Could not release all model-held keys") from errors[0]

    def execute(self, prediction):
        if not self.live:
            raise RuntimeError("Input disabled")
        if self.down(0x7B) or self.down(0x78) or not self.healthy():
            self.release()
            raise RuntimeError("Stop key, focus or geometry guard")
        prediction = np.asarray(prediction)
        if (
            prediction.shape != (len(self.schema.names),)
            or not np.isfinite(prediction).all()
        ):
            self.release()
            raise ValueError("Invalid model prediction")
        desired = {
            name for name, v in zip(self.schema.names[:-2], prediction[:-2]) if v >= 0.7
        }
        # Never hold opposite movement directions simultaneously.
        for a, b in (("W", "S"), ("A", "D")):
            if a in desired and b in desired:
                desired -= {a, b}
        for name in self.held - desired:
            self._button(name, False)
        self.held &= desired
        left, top, w, h = self.rect
        if not self.healthy() or self.down(0x7B):
            self.release()
            raise RuntimeError("Dispatch guard failed")
        self.mouse.position = (
            left + round(np.clip(prediction[-2], 0, 1) * (w - 1)),
            top + round(np.clip(prediction[-1], 0, 1) * (h - 1)),
        )
        for name in desired - self.held:
            if not self.healthy() or self.down(0x7B):
                self.release()
                raise RuntimeError("Dispatch guard failed")
            # Track before dispatch, so uncertain input is released on failure.
            self.held.add(name)
            self._button(name, True)

    def close(self):
        try:
            self.release()
        finally:
            self.capture.close()


def record(args):
    import torch

    from bot_mmorpg.training.learning_loop import load_policy
    from bot_mmorpg.training.recordings import fingerprint, write_session

    _, schema = profile_schema(args.game)
    root = Path(args.root)
    model = metadata = parent = None
    if args.mode in ("correct", "shadow", "play"):
        registry = json.loads((root / "registry.json").read_text())
        active = registry["active"]
        if not active:
            raise ValueError(
                "No promoted model; record demonstrations and run a cycle first"
            )
        checkpoint = root / active["path"]
        parent = fingerprint(checkpoint)
        if parent != active["sha256"]:
            raise ValueError("Active model hash mismatch")
        model, metadata, model_schema = load_policy(checkpoint)
        if (
            model_schema != schema
            or metadata["game"] != args.game
            or metadata["task"] != args.task
        ):
            raise ValueError("Model profile/task mismatch")
    if args.live and args.mode != "play":
        raise ValueError("--live is only valid for play")
    if args.mode != "record" and args.split != "train":
        raise ValueError("Corrections and model-assisted sessions are train only")
    if not 0 < args.seconds <= 3600 or not 1 <= args.fps <= 20:
        raise ValueError("Use 1..3600 seconds and 1..20 FPS")
    io = WindowIO(args.window_title, schema, live=args.live)
    frames, actions, stamps = [], [], []
    history = deque(maxlen=metadata["seq_len"] if metadata else 1)
    human = args.mode != "play" or not args.live
    saved = []
    emergency_stopped = False

    def flush():
        if frames:
            saved.append(
                str(
                    write_session(
                        root,
                        frames,
                        actions,
                        schema,
                        role="correction" if parent else "demonstration",
                        split=args.split,
                        game=args.game,
                        task=args.task,
                        parent_model=parent,
                        timestamps=stamps,
                    )
                )
            )
            frames.clear()
            actions.clear()
            stamps.clear()

    print(
        "Focus the selected game. F8 toggles human takeover in live play; F9/F12 stop.",
        flush=True,
    )
    try:
        time.sleep(3)
        end = time.monotonic() + args.seconds
        prev_toggle = False
        while time.monotonic() < end:
            started = time.monotonic()
            if io.down(0x7B):
                emergency_stopped = True
                break
            if io.down(0x78):
                break
            toggle = io.down(0x77)
            if toggle and not prev_toggle and args.live:
                io.release()
                flush()
                history.clear()
                human = not human
                print("Human correction" if human else "Model control", flush=True)
                # Discard this transition frame; do not label lingering synthetic state.
                prev_toggle = toggle
                time.sleep(0.15)
                continue
            prev_toggle = toggle
            frame, labels = io.observe()
            history.append(
                torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255
            )
            if human:
                frames.append(frame)
                actions.append(labels)
                stamps.append(started)
                if len(frames) >= 300:
                    flush()
            if model and len(history) == history.maxlen:
                with torch.no_grad():
                    prediction = schema.decode(model(torch.stack(list(history))[None]))[
                        0
                    ].numpy()
                if args.live and not human:
                    if time.monotonic() >= end:
                        break
                    io.execute(prediction)
                elif len(frames) % max(1, int(args.fps)) == 0:
                    print(
                        "Suggested:",
                        {
                            k: round(float(v), 3)
                            for k, v in zip(schema.names, prediction)
                            if v > 0.5
                        },
                        flush=True,
                    )
            time.sleep(max(0, 1 / args.fps - (time.monotonic() - started)))
    finally:
        try:
            io.close()
        finally:
            flush()
    print(json.dumps({"saved_sessions": saved}, indent=2))
    return not emergency_stopped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", required=True, help="One game/task learning directory"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for mode in ("record", "correct", "shadow", "play"):
        p = sub.add_parser(mode)
        p.set_defaults(mode=mode)
        p.add_argument("--game", required=True)
        p.add_argument("--task", default="combat")
        p.add_argument(
            "--window-title", required=True, help="Exact visible game window title"
        )
        p.add_argument("--seconds", type=float, required=True)
        p.add_argument("--fps", type=float, default=10)
        p.add_argument(
            "--split", choices=["train", "validation", "test"], default="train"
        )
        p.add_argument(
            "--live",
            action="store_true",
            help="Opt into input for this finite play session",
        )
        p.add_argument(
            "--update",
            action="store_true",
            help="Run one training cycle after this session ends normally",
        )
    p = sub.add_parser("cycle")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--margin", type=float, default=0.005)
    p.add_argument("--learning-rate", type=float, default=0.0005)
    sub.add_parser("status")
    args = parser.parse_args(argv)
    try:
        if args.command == "cycle":
            from bot_mmorpg.training.learning_loop import run_cycle

            result = run_cycle(
                args.root,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                device=args.device,
                seed=args.seed,
                margin=args.margin,
                learning_rate=args.learning_rate,
                on_epoch=lambda row: print(json.dumps(row), flush=True),
            )
            print(json.dumps(result, indent=2))
        elif args.command == "status":
            path = Path(args.root) / "registry.json"
            print(
                path.read_text() if path.exists() else '{"active": null, "history": []}'
            )
        else:
            completed = record(args)
            if args.update and completed:
                from bot_mmorpg.training.learning_loop import run_cycle

                print(json.dumps(run_cycle(args.root), indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
