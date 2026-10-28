"""Offline learning acceptance. No Windows capture or input is used."""

import copy
import json
import threading

import numpy as np
import pytest
import torch

from bot_mmorpg.training.actions import ActionSchema, MixedActionLoss, action_metrics
from bot_mmorpg.training.learning_loop import (
    corpus_lock,
    load_policy,
    run_cycle,
    should_promote,
    validate_corpus,
)
from bot_mmorpg.training.recordings import (
    LegacyGameplayDataset,
    SessionDataset,
    read_sessions,
    write_session,
)


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


SCHEMA = ActionSchema(("attack", "mouse_x", "mouse_y"), ("binary", "unit", "unit"))


def session(root, split="train", seed=1, role="demonstration", parent=None, count=24):
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 40, (count, 16, 16, 3), dtype=np.uint8)
    # A deliberately easy task, with independently sampled recordings.
    actions = np.tile([1.0, 0.8, 0.2], (count, 1)).astype(np.float32)
    return write_session(
        root,
        frames,
        actions,
        SCHEMA,
        split=split,
        role=role,
        parent_model=parent,
        game="fixture",
        task="fixture",
    )


def corpus(root):
    for i, split in enumerate(("train", "validation", "test")):
        session(root, split, i)


def test_mixed_loss_signed_and_real_targets_have_finite_gradients():
    schema = ActionSchema(
        ("button", "x", "dx", "scroll"), ("binary", "unit", "signed", "real")
    )
    logits = torch.zeros((2, 4), requires_grad=True)
    target = torch.tensor([[1.0, 0.25, -0.8, -3.0], [0.0, 0.75, 0.5, 2.0]])
    loss = MixedActionLoss(schema)(logits, target)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2] > 0  # gradient descent moves signed output negative
    metrics = action_metrics(target.numpy(), target.numpy(), schema)
    assert metrics["button_exact"] == 1 and metrics["continuous_mae"]["dx"] == 0


@pytest.mark.parametrize("width", [9, 15, 19, 29, 35, 39])
def test_legacy_schema_dimensions(width):
    assert len(ActionSchema.legacy(width).names) == width


def test_invalid_width_rejected():
    with pytest.raises(ValueError, match="schema"):
        ActionSchema.legacy(30)


def test_sessions_never_cross_boundaries(tmp_path):
    session(tmp_path, count=5)
    session(tmp_path, seed=2, count=5)
    data = SessionDataset(read_sessions(tmp_path), seq_len=4)
    assert len(data) == 4
    assert {i for _, i in data.windows} == {0, 1}
    assert data[0][0].shape == (4, 3, 16, 16)


def legacy_file(folder, session_id=None, seed=1, width=29):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"training_data-{seed}.npy"
    frames = np.random.default_rng(seed).integers(0, 255, (8, 8, 8, 3), dtype=np.uint8)
    pairs = np.empty((8, 2), dtype=object)
    for i, f in enumerate(frames):
        pairs[i] = [f, np.zeros(width, dtype=np.float32)]
    np.save(path, pairs)
    if session_id:
        path.with_suffix(".session.json").write_text(
            json.dumps({"session_id": session_id})
        )


def test_legacy_split_keeps_all_chunks_of_session_together(tmp_path):
    legacy_file(tmp_path, "a", 1)
    legacy_file(tmp_path, "a", 2)
    legacy_file(tmp_path, "b", 3)
    data = LegacyGameplayDataset(tmp_path, seq_len=4)
    train, val = data.split(0.5)
    a = {data.groups[data.windows[i][0]] for i in train.indices}
    b = {data.groups[data.windows[i][0]] for i in val.indices}
    assert a.isdisjoint(b)
    assert len(train) + len(val) == 15  # no windows bridge chunks


def test_unknown_legacy_chunks_cannot_be_random_split(tmp_path):
    legacy_file(tmp_path, seed=1)
    legacy_file(tmp_path, seed=2)
    with pytest.raises(ValueError, match="independent"):
        LegacyGameplayDataset(tmp_path).split()


def test_mixed_legacy_action_width_rejected(tmp_path):
    legacy_file(tmp_path, seed=1, width=29)
    legacy_file(tmp_path, seed=2, width=39)
    with pytest.raises(ValueError, match="Mixed"):
        LegacyGameplayDataset(tmp_path)


def test_mutated_session_fails_hash_verification(tmp_path):
    path = session(tmp_path)
    with (path / "samples.npz").open("ab") as stream:
        stream.write(b"mutation")
    with pytest.raises(ValueError, match="modified"):
        read_sessions(tmp_path)


def test_duplicate_copied_frames_cannot_be_heldout(tmp_path):
    corpus(tmp_path)
    session(tmp_path, "test", seed=0)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_corpus(read_sessions(tmp_path))


def test_correction_requires_provenance_and_train_split(tmp_path):
    with pytest.raises(ValueError, match="Corrections"):
        session(tmp_path, role="correction")
    with pytest.raises(ValueError, match="Corrections"):
        session(tmp_path, split="test", role="correction", parent="fake")


def test_all_three_splits_required(tmp_path):
    session(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        run_cycle(tmp_path)


def test_lock_prevents_concurrent_training(tmp_path):
    with corpus_lock(tmp_path), pytest.raises(RuntimeError, match="lock"):
        run_cycle(tmp_path)
    assert not (tmp_path / "training.lock").exists()


def score(value=0.5, mae=0.1):
    return {
        "score": value - mae,
        "button_f1": value,
        "button_exact": value,
        "continuous_mae": {"x": mae},
        "samples": 10,
    }


@pytest.mark.parametrize("kind", ["regression", "same", "nonfinite", "missing_channel"])
def test_bad_candidate_not_promoted(kind):
    base, new = score(), score(0.8, 0.05)
    if kind == "regression":
        new["continuous_mae"]["x"] = 0.2
    if kind == "same":
        new = copy.deepcopy(base)
    if kind == "nonfinite":
        new["score"] = float("nan")
    if kind == "missing_channel":
        new["continuous_mae"] = {}
    assert not should_promote(new, base)


def test_real_training_promotion_correction_replay_and_evaluation(tmp_path):
    corpus(tmp_path)
    report = run_cycle(tmp_path, epochs=4, seq_len=2, batch_size=8, learning_rate=0.01)
    assert report["status"] == "promoted"
    registry = json.loads((tmp_path / "registry.json").read_text())
    active = registry["active"]
    _model, checkpoint, schema = load_policy(tmp_path / active["path"])
    assert schema == SCHEMA and checkpoint["seq_len"] == 2
    assert run_cycle(tmp_path, seq_len=2)["status"] == "no_new_data"
    session(tmp_path, seed=10, role="correction", parent=active["sha256"])
    # An impossible promotion margin deliberately exercises rejection/preservation.
    second = run_cycle(tmp_path, epochs=1, seq_len=2, batch_size=8, margin=100)
    assert second["status"] == "rejected"
    updated = json.loads((tmp_path / "registry.json").read_text())
    assert updated["active"] == active
    assert len(updated["sessions"]) == 4 and len(updated["history"]) == 2
    assert (tmp_path / second["candidate_path"]).exists()
    from bot_mmorpg.scripts.evaluate_model import main

    assert (
        main(
            ["--model", str(tmp_path / active["path"]), "--dataset-dir", str(tmp_path)]
        )
        == 0
    )


def test_cancel_never_promotes_and_preserves_failure(tmp_path):
    corpus(tmp_path)
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        run_cycle(tmp_path, epochs=1, seq_len=2, stop=stop)
    assert not (tmp_path / "registry.json").exists()
    assert list((tmp_path / "runs").glob("*/failure.json"))


def test_evaluation_benchmark_cannot_change_after_cycle(tmp_path):
    corpus(tmp_path)
    run_cycle(tmp_path, epochs=1, seq_len=2)
    session(tmp_path, "test", seed=10)
    with pytest.raises(ValueError, match="frozen"):
        run_cycle(tmp_path, seq_len=2)


def test_profile_includes_skill_and_mouse_bindings():
    from bot_mmorpg.scripts.learning import profile_schema

    _, schema = profile_schema("diablo_4")
    assert "Q" in schema.names and "1" in schema.names and "MOUSE_LEFT" in schema.names
    assert schema.kinds[-2:] == ("unit", "unit")


def test_input_guard_prevents_dispatch_and_releases():
    from bot_mmorpg.scripts.learning import WindowIO

    io = WindowIO.__new__(WindowIO)
    io.live = True
    io.down = lambda _: False
    io.healthy = lambda: False
    released = []
    io.release = lambda: released.append(True)
    with pytest.raises(RuntimeError, match="guard"):
        io.execute(np.zeros(3))
    assert released == [True]


def test_shadow_input_always_refused():
    from bot_mmorpg.scripts.learning import WindowIO

    io = WindowIO.__new__(WindowIO)
    io.live = False
    with pytest.raises(RuntimeError, match="disabled"):
        io.execute(np.zeros(3))


def test_bridge_does_not_report_success_without_data(tmp_path):
    from bot_mmorpg.bridge.handlers import CommandHandler

    handler = CommandHandler()
    events = []
    handler.emit = lambda name, value: events.append((name, value))
    handler.training_state.is_training = True
    handler._training_loop(str(tmp_path), {"epochs": 1})
    assert [name for name, _ in events] == ["training:error"]
    assert not handler.training_state.is_training


def test_stop_does_not_allow_a_second_training_worker():
    from bot_mmorpg.bridge.handlers import CommandHandler

    handler = CommandHandler()
    handler.training_state.is_training = True
    assert handler.handle_training_stop()["status"] == "stopping"
    assert handler.training_state.is_training
    with pytest.raises(RuntimeError, match="already"):
        handler.handle_training_start("ignored", {})


def test_record_saves_only_human_frames_without_input(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from bot_mmorpg.scripts import learning

    now = [0.0]
    monkeypatch.setattr(
        learning,
        "time",
        SimpleNamespace(
            monotonic=lambda: now[0], sleep=lambda n: now.__setitem__(0, now[0] + n)
        ),
    )
    monkeypatch.setattr(learning, "profile_schema", lambda _: (None, SCHEMA))
    observed, closed = [], []

    class FakeWindow:
        def __init__(self, title, schema, live):
            assert not live

        def down(self, _):
            return False

        def observe(self):
            observed.append(True)
            return np.zeros((16, 16, 3), dtype=np.uint8), np.array([1.0, 0.2, 0.3])

        def close(self):
            closed.append(True)

        def execute(self, _):
            pytest.fail("Recorder must not send game input")

    monkeypatch.setattr(learning, "WindowIO", FakeWindow)
    args = SimpleNamespace(
        root=tmp_path,
        game="fixture",
        task="fixture",
        mode="record",
        live=False,
        split="train",
        seconds=0.5,
        fps=10,
        window_title="fixture",
    )
    assert learning.record(args)
    sessions = read_sessions(tmp_path)
    assert len(sessions) == 1 and sessions[0]["samples"] == len(observed)
    assert closed == [True]
    data = SessionDataset(sessions)
    np.testing.assert_allclose(data[0][1].numpy(), [1.0, 0.2, 0.3])


def test_auto_update_does_not_run_after_emergency_stop(tmp_path, monkeypatch):
    from bot_mmorpg.scripts import learning
    from bot_mmorpg.training import learning_loop

    monkeypatch.setattr(learning, "record", lambda _: False)
    monkeypatch.setattr(
        learning_loop, "run_cycle", lambda *a, **k: pytest.fail("emergency stop")
    )
    assert (
        learning.main(
            [
                "--root",
                str(tmp_path),
                "record",
                "--game",
                "diablo_4",
                "--window-title",
                "fixture",
                "--seconds",
                "1",
                "--update",
            ]
        )
        == 0
    )


def test_checkpoint_retains_mixed_action_width_and_schema(tmp_path):
    from bot_mmorpg.scripts.models_pytorch import get_model, load_model, save_model

    schema = ActionSchema.legacy(39)
    model = get_model("mobilenet_v3", num_actions=39, pretrained=False)
    path = tmp_path / "model.pth"
    save_model(
        model, str(path), model_name="mobilenet_v3", action_schema=schema.to_dict()
    )
    restored, metadata = load_model(str(path), device=torch.device("cpu"))
    assert metadata["num_actions"] == 39
    assert metadata["action_schema"] == schema.to_dict()
    with torch.no_grad():
        assert restored(torch.zeros(2, 3, 32, 32)).shape == (2, 39)
