from pathlib import Path

import numpy as np


def test_find_training_files(tmp_path: Path):
    from bot_mmorpg.scripts.inspect_dataset import find_training_files

    (tmp_path / "nested").mkdir()
    np.save(tmp_path / "training_data-1.npy", np.array([], dtype=object), allow_pickle=True)
    np.save(tmp_path / "nested" / "training_data-2.npy", np.array([], dtype=object), allow_pickle=True)

    files = find_training_files(tmp_path)

    assert len(files) == 2
    assert {path.name for path in files} == {"training_data-1.npy", "training_data-2.npy"}


def test_export_sample_frames(tmp_path: Path):
    from bot_mmorpg.scripts.inspect_dataset import export_sample_frames

    chunk = tmp_path / "training_data-1.npy"
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    action = np.array([1, 0, 0], dtype=np.float32)
    payload = np.array([[frame, action], [frame, action]], dtype=object)
    np.save(chunk, payload, allow_pickle=True)

    out_dir = tmp_path / "samples"
    written = export_sample_frames([chunk], out_dir, samples_per_file=2)

    assert written == 2
    assert len(list(out_dir.glob("*.png"))) == 2
