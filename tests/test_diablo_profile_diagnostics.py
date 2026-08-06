from pathlib import Path

import cv2
import numpy as np


def test_normalized_region_to_pixels():
    from bot_mmorpg.scripts.collect_data import normalized_region_to_pixels

    region = normalized_region_to_pixels([0.25, 0.50, 0.10, 0.20], (100, 40, 2020, 1120))

    assert region == (580, 580, 772, 796)


def test_pixel_region_to_normalized():
    from bot_mmorpg.scripts.collect_data import pixel_region_to_normalized

    normalized = pixel_region_to_normalized([580, 580, 772, 796], (100, 40, 2020, 1120))

    assert normalized == [0.25, 0.5, 0.1, 0.2]


def test_diablo_profile_annotation_metadata():
    from bot_mmorpg.config import GameProfileLoader
    from bot_mmorpg.scripts.collect_data import annotate_profile_regions

    profile = GameProfileLoader().load("diablo_4")
    assert profile.typical_resolution == [2560, 1009]
    screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    annotated, metadata = annotate_profile_regions(
        screen, profile.important_regions, (0, 0, 1920, 1080)
    )

    assert annotated.shape == screen.shape
    assert set(metadata) >= {"health_orb", "resource_orb", "skill_bar", "center_focus"}
    assert metadata["health_orb"]["width"] > 0
    assert metadata["health_orb"]["height"] > 0
    assert metadata["resource_orb"]["width"] > 0
    assert metadata["resource_orb"]["height"] > 0
    assert np.any(annotated != screen)


def test_save_profile_diagnostics(tmp_path: Path):
    from bot_mmorpg.config import GameProfileLoader
    from bot_mmorpg.scripts.collect_data import save_profile_diagnostics

    profile = GameProfileLoader().load("diablo_4")
    screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    diag_dir = save_profile_diagnostics(screen, profile, (0, 0, 1920, 1080), tmp_path)

    raw = diag_dir / "raw.png"
    annotated = diag_dir / "annotated_regions.png"
    metadata = diag_dir / "metadata.json"
    template = diag_dir / "roi_overrides.template.json"

    assert raw.exists()
    assert annotated.exists()
    assert metadata.exists()
    assert template.exists()
    assert cv2.imread(str(annotated)) is not None
    assert '"game_id": "diablo_4"' in metadata.read_text(encoding="utf-8")
    assert '"pixels"' in template.read_text(encoding="utf-8")


def test_action_name_for_index_falls_back():
    from bot_mmorpg.scripts.test_model import action_name_for_index

    assert action_name_for_index(0) == "straight"
    assert action_name_for_index(999) == "action_999"


def test_build_game_output_dir_uses_timestamped_game_folder():
    from bot_mmorpg.scripts.collect_data import build_game_output_dir

    path = build_game_output_dir("datasets", "diablo_4", "loot_collection")

    assert "datasets" in path
    assert "diablo_4" in path
    assert "loot_collection" in path
