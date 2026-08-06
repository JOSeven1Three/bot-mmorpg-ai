from pathlib import Path


def test_move_box_clamps_to_capture_region():
    from bot_mmorpg.scripts.region_catcher import move_box

    moved = move_box((10, 10, 30, 30), -50, 500, (0, 0, 100, 100))

    assert moved == (0, 80, 20, 100)


def test_resize_box_respects_minimum_size():
    from bot_mmorpg.scripts.region_catcher import resize_box

    resized = resize_box((20, 20, 60, 60), -100, -100, (0, 0, 200, 200))

    assert resized == (20, 20, 32, 32)


def test_save_display_regions_updates_profile(tmp_path: Path):
    from bot_mmorpg.config.profile_loader import GameProfileLoader

    profiles_dir = tmp_path / "profiles"
    game_dir = profiles_dir / "demo_game"
    game_dir.mkdir(parents=True)
    profile_path = game_dir / "profile.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "game:",
                '  id: "demo_game"',
                '  name: "Demo Game"',
                '  publisher: "Demo Studio"',
                "display:",
                "  typical_resolution: [1920, 1080]",
                "  ui_scale_range: [0.8, 1.2]",
                "  important_regions:",
                "    test_box: [0.1, 0.2, 0.3, 0.4]",
                "input:",
                '  action_space: "hybrid"',
                "  num_actions: 1",
                "  requires_mouse: true",
                '  mouse_precision: "medium"',
                "  default_bindings: []",
                "training:",
                '  recommended_architecture: "efficientnet_simple"',
                "  recommended_input_size: [224, 224]",
                "  temporal_frames: 1",
                "  minimum_samples: 1",
                "  class_balance_tolerance: 0.3",
                "  learning_rate: {initial: 0.001, decay_factor: 0.1, decay_epochs: [20, 40]}",
                "hardware_tiers:",
                "  medium:",
                '    architecture: "efficientnet_simple"',
                "    batch_size: 1",
                "    input_size: [224, 224]",
                "    temporal_frames: 1",
                "    workers: 1",
                "tasks:",
                "  combat:",
                '    description: "demo"',
                '    priority: "accuracy"',
                "    temporal: false",
                '    recommended_architecture: "efficientnet_simple"',
                "    augmentation: []",
                "    fps_target: 30",
            ]
        ),
        encoding="utf-8",
    )

    loader = GameProfileLoader(profiles_dir=profiles_dir)
    loader.save_display_regions("demo_game", {"test_box": [0.2, 0.3, 0.2, 0.2]})

    updated = loader.load_raw_data("demo_game")
    assert updated["display"]["important_regions"]["test_box"] == [0.2, 0.3, 0.2, 0.2]
    assert profile_path.with_suffix(".yaml.bak").exists()
