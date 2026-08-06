"""
Tests for Configuration System

Tests hardware detection, profile loading, settings management, and model selection.
"""

import pytest


class TestHardwareDetector:
    """Test hardware detection functionality."""

    def test_detect_returns_info(self):
        """Test hardware detection returns valid info."""
        from bot_mmorpg.config import HardwareDetector

        detector = HardwareDetector()
        info = detector.detect()

        assert info is not None
        assert info.platform.lower() in ["linux", "windows", "darwin"]
        assert info.cpu_cores > 0
        assert info.ram_mb > 0

    def test_hardware_tier_assignment(self):
        """Test hardware tier is assigned."""
        from bot_mmorpg.config import HardwareDetector, HardwareTier

        detector = HardwareDetector()
        info = detector.detect()

        assert info.tier in [HardwareTier.LOW, HardwareTier.MEDIUM, HardwareTier.HIGH]

    def test_hardware_summary(self):
        """Test hardware summary generation."""
        from bot_mmorpg.config import HardwareDetector

        detector = HardwareDetector()
        info = detector.detect()
        summary = info.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "CPU" in summary or "RAM" in summary


class TestGameProfileLoader:
    """Test game profile loading."""

    def test_list_games(self):
        """Test listing available games."""
        from bot_mmorpg.config import GameProfileLoader

        loader = GameProfileLoader()
        games = loader.list_games()

        assert isinstance(games, list)
        assert len(games) > 0
        for game in games:
            assert "id" in game
            assert "name" in game

    def test_load_wow_profile(self):
        """Test loading World of Warcraft profile."""
        from bot_mmorpg.config import GameProfileLoader

        loader = GameProfileLoader()
        profile = loader.load("world_of_warcraft")

        assert profile.id == "world_of_warcraft"
        assert profile.name == "World of Warcraft"
        assert len(profile.tasks) > 0

    def test_load_gw2_profile(self):
        """Test loading Guild Wars 2 profile."""
        from bot_mmorpg.config import GameProfileLoader

        loader = GameProfileLoader()
        profile = loader.load("guild_wars_2")

        assert profile.id == "guild_wars_2"
        assert profile.name == "Guild Wars 2"

    def test_load_diablo_4_profile(self):
        """Test loading Diablo IV profile."""
        from bot_mmorpg.config import GameProfileLoader

        loader = GameProfileLoader()
        profile = loader.load("diablo_4")

        assert profile.id == "diablo_4"
        assert profile.name == "Diablo IV"
        assert profile.requires_mouse is True
        assert "combat" in profile.list_tasks()
        assert "loot_collection" in profile.list_tasks()
        assert "helltides" in profile.external_overlays
        assert (
            profile.external_overlays["helltides"]["project_path"]
            == "C:/Users/Jose/HermesOS/Projects/Diablo4 Helltides overlay"
        )

    def test_load_nonexistent_profile(self):
        """Test loading nonexistent profile raises error."""
        from bot_mmorpg.config import GameProfileLoader

        loader = GameProfileLoader()

        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_game")

    def test_profile_has_tasks(self):
        """Test profile contains tasks."""
        from bot_mmorpg.config import GameProfileLoader

        loader = GameProfileLoader()
        profile = loader.load("world_of_warcraft")

        assert len(profile.tasks) > 0
        task_names = profile.list_tasks()
        assert "combat" in task_names or len(task_names) > 0

    def test_profile_hardware_tiers(self):
        """Test profile has hardware tier configs."""
        from bot_mmorpg.config import GameProfileLoader

        loader = GameProfileLoader()
        profile = loader.load("world_of_warcraft")

        assert "low" in profile.hardware_tiers or len(profile.hardware_tiers) > 0


class TestModelSelector:
    """Test model selection and recommendation."""

    def test_recommend_for_combat(self):
        """Test model recommendation for combat task."""
        from bot_mmorpg.config import ModelSelector

        selector = ModelSelector()
        rec = selector.recommend(game_id="world_of_warcraft", task="combat")

        assert rec is not None
        assert rec.architecture is not None
        assert rec.confidence > 0
        assert rec.confidence <= 1.0

    def test_recommend_returns_reasons(self):
        """Test recommendation includes reasons."""
        from bot_mmorpg.config import ModelSelector

        selector = ModelSelector()
        rec = selector.recommend(game_id="world_of_warcraft", task="combat")

        assert isinstance(rec.reasons, list)
        assert len(rec.reasons) > 0

    def test_list_architectures(self):
        """Test listing available architectures."""
        from bot_mmorpg.config import ModelSelector

        selector = ModelSelector()
        archs = selector.list_architectures()

        assert isinstance(archs, list)
        assert len(archs) > 0
        for arch in archs:
            assert "id" in arch
            assert "name" in arch

    def test_recommend_includes_estimates(self):
        """Test recommendation includes speed/accuracy estimates."""
        from bot_mmorpg.config import ModelSelector

        selector = ModelSelector()
        rec = selector.recommend(game_id="world_of_warcraft", task="combat")

        assert rec.estimated_speed is not None
        assert rec.estimated_accuracy is not None
        assert rec.recommended_batch_size > 0
        assert rec.recommended_input_size is not None


class TestSettingsManager:
    """Test settings management."""

    def test_get_quick_config(self):
        """Test getting quick configuration."""
        from bot_mmorpg.config import SettingsManager

        manager = SettingsManager()
        config = manager.get_quick_config(game_id="world_of_warcraft", task="combat")

        assert config is not None
        assert isinstance(config, dict)
        # Quick config returns training parameters directly
        assert "architecture" in config
        assert "batch_size" in config
        assert "epochs" in config

    def test_create_session_config(self):
        """Test creating session configuration."""
        from bot_mmorpg.config import SettingsManager

        manager = SettingsManager()

        config = manager.create_session_config(
            game_id="world_of_warcraft",
            task="combat",
            overrides={"training": {"model": {"architecture": "mobilenetv3"}}},
        )

        assert config is not None
        assert config.game_id == "world_of_warcraft"
        assert config.task == "combat"

    def test_save_session_config(self, tmp_path):
        """Test saving session configuration."""
        from bot_mmorpg.config import SettingsManager

        manager = SettingsManager()
        manager.settings_dir = tmp_path

        config = manager.create_session_config(
            game_id="world_of_warcraft", task="combat"
        )

        path = manager.save_session_config(config)

        assert path.exists()
        assert path.suffix == ".yaml"


class TestHardwareTierEnum:
    """Test hardware tier enum."""

    def test_tier_values(self):
        """Test tier enum values."""
        from bot_mmorpg.config import HardwareTier

        assert HardwareTier.LOW.value == "low"
        assert HardwareTier.MEDIUM.value == "medium"
        assert HardwareTier.HIGH.value == "high"


class TestGameProfile:
    """Test game profile data class."""

    def test_get_task_config(self):
        """Test getting task configuration."""
        from bot_mmorpg.config import GameProfileLoader

        loader = GameProfileLoader()
        profile = loader.load("world_of_warcraft")

        tasks = profile.list_tasks()
        if tasks:
            task_config = profile.get_task_config(tasks[0])
            assert task_config is not None

    def test_get_nonexistent_task(self):
        """Test getting nonexistent task returns None."""
        from bot_mmorpg.config import GameProfileLoader

        loader = GameProfileLoader()
        profile = loader.load("world_of_warcraft")

        task_config = profile.get_task_config("nonexistent_task")
        assert task_config is None
