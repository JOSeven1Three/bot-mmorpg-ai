"""
Production-Readiness Tests for BOT-MMORPG-AI

Comprehensive tests with mocks to verify:
1. All game profiles load and validate correctly
2. All neural network architectures instantiate and run forward passes
3. Mouse recording is additive and non-destructive
4. Data collection pipeline works end-to-end (mocked I/O)
5. Training pipeline works end-to-end (mocked data)
6. Inference pipeline works end-to-end (mocked model)
7. Stuck detection and evasive maneuver logic
8. Action mapping and encoding/decoding
9. Tauri/launcher production configuration
"""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ============================================================================
# Fixtures
# ============================================================================

torch = pytest.importorskip("torch", reason="PyTorch required")


@pytest.fixture
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def game_profiles_dir(project_root):
    """Return game profiles directory."""
    return project_root / "game_profiles"


@pytest.fixture
def mock_screen():
    """Return a mock screen capture (480x270 RGB)."""
    return np.random.randint(0, 255, (270, 480, 3), dtype=np.uint8)


@pytest.fixture
def mock_screen_bgr():
    """Return a mock screen capture in BGR format."""
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


# ============================================================================
# 1. Game Profile Tests — validate ALL profiles
# ============================================================================


class TestAllGameProfiles:
    """Validate every game profile loads, parses, and has required fields."""

    EXPECTED_GAMES = [
        "genshin_impact",
        "world_of_warcraft",
        "guild_wars_2",
        "final_fantasy_xiv",
        "lost_ark",
        "diablo_4",
        "new_world",
        "dragon_ball_online",
    ]

    def test_profile_index_exists(self, game_profiles_dir):
        """Test that index.yaml exists and lists all games."""
        index_path = game_profiles_dir / "index.yaml"
        assert index_path.exists(), "game_profiles/index.yaml is missing"

    def test_all_expected_profiles_exist(self, game_profiles_dir):
        """Test that every expected game has a profile.yaml."""
        for game_id in self.EXPECTED_GAMES:
            profile_path = game_profiles_dir / game_id / "profile.yaml"
            assert profile_path.exists(), f"Missing profile for {game_id}"

    def test_template_profile_exists(self, game_profiles_dir):
        """Test that the template profile exists for new games."""
        template = game_profiles_dir / "_template" / "profile.yaml"
        assert template.exists(), "Template profile is missing"

    def test_load_all_profiles_via_loader(self):
        """Test that GameProfileLoader can load every game."""
        from bot_mmorpg.config import GameProfileLoader

        loader = GameProfileLoader()
        games = loader.list_games()
        game_ids = [g["id"] for g in games]

        for game_id in self.EXPECTED_GAMES:
            assert game_id in game_ids, f"{game_id} not listed by loader"

        for game_id in self.EXPECTED_GAMES:
            profile = loader.load(game_id)
            assert profile.id == game_id
            assert profile.name is not None and len(profile.name) > 0

    def test_all_profiles_have_required_sections(self):
        """Test that every profile has input, training, hardware_tiers, and tasks."""
        from bot_mmorpg.config import GameProfileLoader

        loader = GameProfileLoader()
        for game_id in self.EXPECTED_GAMES:
            profile = loader.load(game_id)
            assert profile.action_space is not None, f"{game_id}: missing action_space"
            assert profile.num_actions > 0, f"{game_id}: num_actions must be > 0"
            assert profile.recommended_architecture is not None, (
                f"{game_id}: missing recommended_architecture"
            )
            assert len(profile.hardware_tiers) > 0, f"{game_id}: no hardware tiers"
            assert len(profile.tasks) > 0, f"{game_id}: no tasks defined"

    def test_all_profiles_have_valid_architectures(self):
        """Test that recommended architectures in profiles are real models."""
        from bot_mmorpg.config import GameProfileLoader
        from bot_mmorpg.scripts.models_pytorch import list_models

        valid_models = list_models()
        # Profile architectures use slightly different names, map them
        arch_aliases = {
            "mobilenetv3": "mobilenet_v3",
            "efficientnet_simple": "efficientnet_simple",
            "efficientnet_lstm": "efficientnet_lstm",
            "resnet18_lstm": "resnet18_lstm",
        }

        loader = GameProfileLoader()
        for game_id in self.EXPECTED_GAMES:
            profile = loader.load(game_id)
            for tier_name, tier_config in profile.hardware_tiers.items():
                arch = tier_config.architecture
                resolved = arch_aliases.get(arch, arch)
                assert resolved in valid_models, (
                    f"{game_id} tier '{tier_name}' has unknown arch '{arch}'"
                )

    def test_all_profiles_have_combat_task(self):
        """Test that every game profile defines a combat task."""
        from bot_mmorpg.config import GameProfileLoader

        loader = GameProfileLoader()
        for game_id in self.EXPECTED_GAMES:
            profile = loader.load(game_id)
            tasks = profile.list_tasks()
            assert "combat" in tasks, f"{game_id}: missing 'combat' task"

    def test_profile_hardware_tier_configs_valid(self):
        """Test hardware tiers have valid batch_size and input_size."""
        from bot_mmorpg.config import GameProfileLoader

        loader = GameProfileLoader()
        for game_id in self.EXPECTED_GAMES:
            profile = loader.load(game_id)
            for tier_name, config in profile.hardware_tiers.items():
                assert config.batch_size > 0, (
                    f"{game_id}/{tier_name}: batch_size must be > 0"
                )
                assert len(config.input_size) == 2, (
                    f"{game_id}/{tier_name}: input_size must be [h, w]"
                )


# ============================================================================
# 2. Neural Network Tests — ALL architectures
# ============================================================================


class TestAllNeuralNetworks:
    """Test every registered neural network architecture."""

    def test_list_models_returns_all(self):
        """Test list_models returns expected count."""
        from bot_mmorpg.scripts.models_pytorch import list_models

        models = list_models()
        assert len(models) >= 8, f"Expected >=8 models, got {len(models)}"

    @pytest.mark.parametrize(
        "model_name",
        [
            "efficientnet_lstm",
            "efficientnet_simple",
            "mobilenet_v3",
            "resnet18_lstm",
            "inception_v3",
            "alexnet",
            "sentnet",
            "sentnet_2d",
        ],
    )
    def test_model_instantiates(self, model_name):
        """Test that each model can be instantiated."""
        from bot_mmorpg.scripts.models_pytorch import get_model

        model = get_model(model_name, num_actions=29, pretrained=False)
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.parametrize(
        "model_name",
        [
            "efficientnet_lstm",
            "efficientnet_simple",
            "mobilenet_v3",
            "resnet18_lstm",
            "inception_v3",
            "alexnet",
            "sentnet_2d",
        ],
    )
    def test_model_forward_pass(self, model_name):
        """Test forward pass produces correct output shape."""
        from bot_mmorpg.scripts.models_pytorch import get_model

        model = get_model(model_name, num_actions=29, pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 270, 480)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 29), f"{model_name}: expected (1,29), got {out.shape}"

    @pytest.mark.parametrize("model_name", ["efficientnet_lstm", "resnet18_lstm"])
    def test_temporal_model_sequence_forward(self, model_name):
        """Test temporal models with frame sequences."""
        from bot_mmorpg.scripts.models_pytorch import get_model

        model = get_model(model_name, num_actions=29, pretrained=False)
        model.eval()

        x = torch.randn(1, 4, 3, 270, 480)  # batch, seq, C, H, W
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 29)

    @pytest.mark.parametrize(
        "model_name",
        [
            "efficientnet_lstm",
            "efficientnet_simple",
            "mobilenet_v3",
            "resnet18_lstm",
        ],
    )
    def test_model_batch_processing(self, model_name):
        """Test models work with batch > 1."""
        from bot_mmorpg.scripts.models_pytorch import get_model

        model = get_model(model_name, num_actions=29, pretrained=False)
        model.eval()

        x = torch.randn(4, 3, 270, 480)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (4, 29)

    def test_model_info_complete(self):
        """Test that every model has complete metadata."""
        from bot_mmorpg.scripts.models_pytorch import get_model_info, list_models

        required_keys = {"name", "description", "params", "temporal"}
        for name in list_models():
            info = get_model_info(name)
            for key in required_keys:
                assert key in info, f"{name}: missing '{key}' in model info"

    def test_model_save_and_load_roundtrip(self, tmp_path):
        """Test model can be saved and loaded back."""
        from bot_mmorpg.scripts.models_pytorch import get_model, save_model

        model = get_model("mobilenet_v3", num_actions=29, pretrained=False)
        save_path = str(tmp_path / "test_model.pth")
        save_model(model, save_path, model_name="mobilenet_v3")

        assert Path(save_path).exists()

        # Verify it loads
        checkpoint = torch.load(save_path, weights_only=False)
        assert "model_state_dict" in checkpoint

    @pytest.mark.parametrize("num_actions", [9, 29, 48, 73])
    def test_model_different_action_sizes(self, num_actions):
        """Test models work with different action space sizes."""
        from bot_mmorpg.scripts.models_pytorch import get_model

        model = get_model("mobilenet_v3", num_actions=num_actions, pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 270, 480)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, num_actions)


# ============================================================================
# 3. Mouse Recording Tests — additive, non-destructive
# ============================================================================


class TestMouseRecordingAdditive:
    """Verify mouse recording is additive and never breaks existing pipeline."""

    def test_capture_input_without_mouse(self):
        """Test capture_input returns 2-part tuple when mouse is None."""
        from bot_mmorpg.scripts.collect_data import capture_input

        with patch("bot_mmorpg.scripts.collect_data.key_check", return_value=[]):
            with patch(
                "bot_mmorpg.scripts.collect_data.gamepad_check",
                return_value=[0] * 20,
            ):
                kb, gp, mouse = capture_input(mouse_capturer=None)

        assert len(kb) == 9
        assert len(gp) == 20
        assert mouse is None

    def test_capture_input_with_mock_mouse(self):
        """Test capture_input appends mouse data when capturer provided."""
        from bot_mmorpg.scripts.collect_data import capture_input
        from bot_mmorpg.scripts.mouse_capture import MouseCapture, MouseState

        mock_mc = MagicMock(spec=MouseCapture)
        mock_mc.snapshot.return_value = MouseState(
            x=0.5,
            y=0.3,
            dx=0.01,
            dy=-0.02,
            vx=0.1,
            vy=-0.1,
            lmb=1,
            rmb=0,
            mmb=0,
            scroll=0.0,
            timestamp=time.time(),
        )

        with patch("bot_mmorpg.scripts.collect_data.key_check", return_value=["W"]):
            with patch(
                "bot_mmorpg.scripts.collect_data.gamepad_check",
                return_value=[0] * 20,
            ):
                kb, gp, mouse = capture_input(mouse_capturer=mock_mc)

        assert len(kb) == 9
        assert kb[0] == 1  # W key
        assert len(gp) == 20
        assert mouse is not None
        assert mouse.shape == (10,)  # x, y, dx, dy, vx, vy, lmb, rmb, mmb, scroll
        assert abs(mouse[0] - 0.5) < 0.01  # x
        assert mouse[6] == 1.0  # lmb (index 6 in 10-value vector)

    def test_mouse_failure_does_not_break_recording(self):
        """Test that mouse capture failure returns None, not an error."""
        from bot_mmorpg.scripts.collect_data import capture_input

        mock_mc = MagicMock()
        mock_mc.snapshot.side_effect = RuntimeError("Mouse device lost")

        with patch("bot_mmorpg.scripts.collect_data.key_check", return_value=[]):
            with patch(
                "bot_mmorpg.scripts.collect_data.gamepad_check",
                return_value=[0] * 20,
            ):
                kb, gp, mouse = capture_input(mouse_capturer=mock_mc)

        assert len(kb) == 9
        assert len(gp) == 20
        assert mouse is None  # Gracefully degraded

    def test_output_vector_sizes(self):
        """Test combined vector sizes: kb(9) + gp(20) = 29 without mouse, +10 with."""
        from bot_mmorpg.scripts.collect_data import capture_input
        from bot_mmorpg.scripts.mouse_capture import MouseCapture, MouseState

        mock_mc = MagicMock(spec=MouseCapture)
        mock_mc.snapshot.return_value = MouseState(x=0.5, y=0.5)

        with patch("bot_mmorpg.scripts.collect_data.key_check", return_value=[]):
            with patch(
                "bot_mmorpg.scripts.collect_data.gamepad_check",
                return_value=[0] * 20,
            ):
                # Without mouse
                kb, gp, mouse_off = capture_input(mouse_capturer=None)
                parts_off = [np.array(kb)]
                if gp:
                    parts_off.append(np.array(gp))
                vec_off = np.concatenate(parts_off)

                # With mouse
                kb2, gp2, mouse_on = capture_input(mouse_capturer=mock_mc)
                parts_on = [np.array(kb2)]
                if gp2:
                    parts_on.append(np.array(gp2))
                if mouse_on is not None:
                    parts_on.append(mouse_on)
                vec_on = np.concatenate(parts_on)

        assert vec_off.shape == (29,)
        assert vec_on.shape == (
            39,
        )  # 29 + 10 mouse (x,y,dx,dy,vx,vy,lmb,rmb,mmb,scroll)

    def test_mouse_state_serialization_roundtrip(self, tmp_path):
        """Test mouse data can be saved and loaded in numpy format."""
        from bot_mmorpg.scripts.mouse_capture import MouseState

        state = MouseState(
            x=0.75,
            y=0.25,
            dx=0.01,
            dy=-0.05,
            vx=0.2,
            vy=-0.3,
            lmb=1,
            rmb=0,
            mmb=1,
            scroll=-1.5,
        )
        arr = state.to_array()
        assert arr.shape == (10,)

        path = tmp_path / "mouse_data.npy"
        np.save(str(path), arr)
        loaded = np.load(str(path))

        np.testing.assert_array_almost_equal(arr, loaded)

    def test_mouse_state_legacy_format(self):
        """Test backward-compatible 6-value legacy format."""
        from bot_mmorpg.scripts.mouse_capture import MouseState

        state = MouseState(x=0.5, y=0.5, lmb=1, rmb=1, mmb=0, scroll=2.0)
        arr = state.to_array_legacy()
        assert arr.shape == (6,)
        assert arr[0] == 0.5  # x
        assert arr[2] == 1.0  # lmb
        assert arr[3] == 1.0  # rmb

    def test_mouse_state_delta_and_velocity(self):
        """Test delta movement and velocity fields are in the 10-value vector."""
        from bot_mmorpg.scripts.mouse_capture import MouseState

        state = MouseState(
            x=0.5,
            y=0.5,
            dx=0.1,
            dy=-0.2,
            vx=0.3,
            vy=-0.4,
            lmb=0,
            rmb=0,
            mmb=0,
            scroll=0.0,
        )
        arr = state.to_array()
        assert abs(arr[2] - 0.1) < 0.001  # dx
        assert abs(arr[3] - (-0.2)) < 0.001  # dy
        assert abs(arr[4] - 0.3) < 0.001  # vx
        assert abs(arr[5] - (-0.4)) < 0.001  # vy

    def test_mouse_buttons_recorded_correctly(self):
        """Test all 3 mouse buttons are recorded and retrievable."""
        from bot_mmorpg.scripts.mouse_capture import MouseState

        # All buttons pressed
        state = MouseState(lmb=1, rmb=1, mmb=1)
        arr = state.to_array()
        assert arr[6] == 1.0  # lmb
        assert arr[7] == 1.0  # rmb
        assert arr[8] == 1.0  # mmb

        # No buttons pressed
        state2 = MouseState(lmb=0, rmb=0, mmb=0)
        arr2 = state2.to_array()
        assert arr2[6] == 0.0
        assert arr2[7] == 0.0
        assert arr2[8] == 0.0

    def test_mouse_capture_thread_safe(self):
        """Test MouseCapture snapshot is thread-safe."""
        from bot_mmorpg.scripts.mouse_capture import MouseCapture

        mc = MouseCapture(capture_region=(0, 0, 1920, 1080))
        errors = []

        def worker():
            try:
                for i in range(50):
                    with mc._state.lock:
                        mc._state.abs_x = i * 20
                        mc._state.abs_y = i * 10
                    s = mc.snapshot()
                    assert 0.0 <= s.x <= 1.0
                    assert 0.0 <= s.y <= 1.0
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"


# ============================================================================
# 4. Data Collection Pipeline (Mocked I/O)
# ============================================================================


class TestDataCollectionPipeline:
    """Test the data collection pipeline with mocked screen/input."""

    def test_keys_to_output_w(self):
        """Test W key encoding."""
        from bot_mmorpg.scripts.collect_data import keys_to_output

        out = keys_to_output(["W"])
        assert out == [1, 0, 0, 0, 0, 0, 0, 0, 0]

    def test_keys_to_output_combo_wa(self):
        """Test W+A combo encoding."""
        from bot_mmorpg.scripts.collect_data import keys_to_output

        out = keys_to_output(["W", "A"])
        assert out == [0, 0, 0, 0, 1, 0, 0, 0, 0]  # WA combo

    def test_keys_to_output_no_keys(self):
        """Test no-key encoding."""
        from bot_mmorpg.scripts.collect_data import keys_to_output

        out = keys_to_output([])
        assert out == [0, 0, 0, 0, 0, 0, 0, 0, 1]  # NOKEY

    def test_keys_to_output_all_combos(self):
        """Test all 8 direction combos + no-key = exactly one hot."""
        from bot_mmorpg.scripts.collect_data import keys_to_output

        combos = [
            (["W"], 0),
            (["S"], 1),
            (["A"], 2),
            (["D"], 3),
            (["W", "A"], 4),
            (["W", "D"], 5),
            (["S", "A"], 6),
            (["S", "D"], 7),
            ([], 8),
        ]
        for keys, expected_idx in combos:
            out = keys_to_output(keys)
            assert sum(out) == 1, f"keys={keys}: not one-hot"
            assert out[expected_idx] == 1, f"keys={keys}: wrong index"

    def test_gamepad_keys_to_output(self):
        """Test gamepad output conversion."""
        from bot_mmorpg.scripts.collect_data import gamepad_keys_to_output

        raw = [0.5, 1.0, -0.3, 0.7, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        out = gamepad_keys_to_output(raw)
        assert len(out) == 20
        assert all(isinstance(v, int) for v in out)

    def test_save_training_data_roundtrip(self, tmp_path):
        """Test training data save and load."""
        from bot_mmorpg.scripts.collect_data import save_training_data

        # Create mock data: list of [screen, action] pairs
        data = []
        for _ in range(10):
            screen = np.random.randint(0, 255, (270, 480, 3), dtype=np.uint8)
            action = np.zeros(29, dtype=np.float32)
            action[0] = 1.0
            data.append([screen, action])

        path = tmp_path / "test_training.npy"
        success = save_training_data(data, path)
        assert success is True
        assert path.exists()

        # Load and verify
        loaded = np.load(str(path), allow_pickle=True)
        assert len(loaded) == 10
        assert loaded[0][0].shape == (270, 480, 3)
        assert loaded[0][1].shape == (29,)

    def test_capture_screen_mock(self):
        """Test screen capture function with mocked grab_screen."""
        from bot_mmorpg.scripts.collect_data import capture_screen

        fake_screen = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        with patch(
            "bot_mmorpg.scripts.collect_data.grab_screen", return_value=fake_screen
        ):
            result = capture_screen(region=(0, 0, 1920, 1080))

        assert result.shape == (270, 480, 3)

    def test_validate_dependencies_passes(self):
        """Test dependency validation passes when modules are available."""
        from bot_mmorpg.scripts.collect_data import validate_dependencies

        with patch("bot_mmorpg.scripts.collect_data.grab_screen", return_value=True):
            with patch("bot_mmorpg.scripts.collect_data.key_check", return_value=True):
                with patch(
                    "bot_mmorpg.scripts.collect_data.gamepad_check",
                    return_value=True,
                ):
                    assert validate_dependencies() is True


# ============================================================================
# 5. Training Pipeline (Mocked Data)
# ============================================================================


class TestTrainingPipeline:
    """Test training pipeline with synthetic data."""

    def test_gameplay_dataset_loads(self, tmp_path):
        """Test GameplayDataset loads .npy files."""
        from bot_mmorpg.scripts.train_model import GameplayDataset

        # Create synthetic training data
        data = []
        for _ in range(50):
            screen = np.random.randint(0, 255, (270, 480, 3), dtype=np.uint8)
            action = np.zeros(29, dtype=np.float32)
            action[np.random.randint(0, 9)] = 1.0
            data.append([screen, action])

        np.save(
            str(tmp_path / "training_data-1.npy"),
            np.array(data, dtype=object),
            allow_pickle=True,
        )

        dataset = GameplayDataset(tmp_path, seq_len=1)
        assert len(dataset) == 50

        frame, action = dataset[0]
        assert frame.shape == (3, 270, 480)  # CHW
        assert action.shape == (29,)

    def test_training_loop_runs(self, tmp_path):
        """Test full training loop with tiny model and synthetic data."""
        from bot_mmorpg.scripts.train_model import train_epoch, validate

        # Tiny model
        model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(3, 29),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Synthetic batch
        frames = torch.randn(8, 3, 32, 32)
        actions = torch.zeros(8, 29)
        for i in range(8):
            actions[i, i % 9] = 1.0

        dataset = torch.utils.data.TensorDataset(frames, actions)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        # Train one epoch
        loss = train_epoch(model, loader, optimizer, criterion, torch.device("cpu"), 0)
        assert loss > 0
        assert np.isfinite(loss)

        # Validate
        val_loss, val_acc = validate(model, loader, criterion, torch.device("cpu"))
        assert val_loss > 0
        assert 0.0 <= val_acc <= 1.0


# ============================================================================
# 6. Inference Pipeline (Mocked Model)
# ============================================================================


class TestInferencePipeline:
    """Test inference pipeline with mocked components."""

    def test_inference_engine_predict(self):
        """Test InferenceEngine.predict with mocked model."""
        from bot_mmorpg.scripts.test_model import InferenceEngine, build_action_weights

        # Mock model that returns fixed predictions
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_output = torch.randn(1, 29)
        mock_model.return_value = mock_output

        # Create engine with mocked internals
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model = mock_model
        engine.device = torch.device("cpu")
        engine.is_temporal = False
        engine.temporal_frames = 1
        engine.frame_buffer = []
        engine.enable_gamepad = False
        engine.motion_log = []
        engine.metadata = {"temporal_frames": 1}
        engine._last_evasion_time = 0.0
        engine._consecutive_stuck = 0
        engine.num_actions = 29
        engine.has_mouse_output = False
        engine.mouse_output_size = 0
        engine._action_weights = build_action_weights(29)

        # Run predict
        screen = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        action_idx, action_val, preds = engine.predict(screen)

        assert 0 <= action_idx < 29
        assert isinstance(action_val, (float, np.floating))
        assert preds.shape == (29,)


# ============================================================================
# 7. Stuck Detection and Evasive Maneuver
# ============================================================================


class TestStuckDetection:
    """Test stuck detection and recovery logic."""

    def _make_engine(self):
        """Create a minimal InferenceEngine for stuck detection testing."""
        from collections import deque

        from bot_mmorpg.scripts.test_model import (
            LOG_LEN,
            InferenceEngine,
        )

        engine = InferenceEngine.__new__(InferenceEngine)
        engine.motion_log = deque(maxlen=LOG_LEN)
        engine._last_evasion_time = 0.0
        engine._consecutive_stuck = 0
        return engine

    def test_not_stuck_with_high_motion(self):
        """Test that high motion does not trigger stuck."""
        engine = self._make_engine()

        for _ in range(30):
            result = engine.check_stuck(2000)  # High motion
        assert result is False

    def test_stuck_with_low_motion(self):
        """Test that low motion triggers stuck detection."""
        engine = self._make_engine()

        for _ in range(30):
            result = engine.check_stuck(100)  # Very low motion
        assert result is True

    def test_stuck_cooldown_prevents_spam(self):
        """Test that cooldown prevents rapid evasive maneuvers."""
        engine = self._make_engine()

        # Fill motion log with low values
        for _ in range(30):
            engine.check_stuck(100)

        # First detection should pass
        assert engine.check_stuck(100) is True

        # Execute evasive (sets cooldown)
        engine._last_evasion_time = time.time()
        engine.motion_log.clear()

        # Fill again
        for _ in range(30):
            engine.motion_log.append(100)

        # Should NOT trigger due to cooldown
        assert engine.check_stuck(100) is False

    def test_evasive_maneuver_escalation(self):
        """Test evasive maneuver escalates on consecutive stuck."""
        engine = self._make_engine()

        # Patch directkeys to avoid actual key presses
        with patch("bot_mmorpg.scripts.test_model.DIRECTKEYS_AVAILABLE", False):
            engine.evasive_maneuver()
            assert engine._consecutive_stuck == 1

            engine.evasive_maneuver()
            assert engine._consecutive_stuck == 2

            engine.evasive_maneuver()
            assert engine._consecutive_stuck == 3

    def test_reset_stuck_counter(self):
        """Test stuck counter resets with good motion."""
        from bot_mmorpg.scripts.test_model import LOG_LEN, MOTION_REQ

        engine = self._make_engine()
        engine._consecutive_stuck = 3

        # Fill with high motion
        for _ in range(LOG_LEN):
            engine.motion_log.append(MOTION_REQ * 2)

        engine.reset_stuck_counter()
        assert engine._consecutive_stuck == 0


# ============================================================================
# 8. Action Mapping System
# ============================================================================


class TestActionMapping:
    """Test the action mapping and encoding system."""

    def test_action_categories_defined(self):
        """Test action categories enum exists."""
        from bot_mmorpg.config.action_mapping import ActionCategory

        assert hasattr(ActionCategory, "MOVEMENT")
        assert hasattr(ActionCategory, "SKILLS")
        assert hasattr(ActionCategory, "COMBAT")
        assert hasattr(ActionCategory, "CAMERA")

    def test_action_definition_dataclass(self):
        """Test ActionDefinition creation."""
        from bot_mmorpg.config.action_mapping import ActionCategory, ActionDefinition

        action = ActionDefinition(
            id=0,
            name="move_forward",
            category=ActionCategory.MOVEMENT,
            key_binding="W",
            is_continuous=True,
        )
        assert action.id == 0
        assert action.name == "move_forward"
        assert action.category == ActionCategory.MOVEMENT

    def test_movement_actions_defined(self):
        """Test movement actions are defined."""
        from bot_mmorpg.config.action_mapping import MOVEMENT_ACTIONS

        assert len(MOVEMENT_ACTIONS) > 0
        names = [a.name for a in MOVEMENT_ACTIONS]
        assert "move_forward" in names
        assert "move_backward" in names

    def test_action_weights_match_action_count(self):
        """Test default ACTION_WEIGHTS array matches expected 29 actions."""
        from bot_mmorpg.scripts.test_model import ACTION_WEIGHTS

        assert ACTION_WEIGHTS.shape == (29,)
        assert all(w > 0 for w in ACTION_WEIGHTS)

    def test_action_names_match_weights(self):
        """Test ACTION_NAMES list matches ACTION_WEIGHTS length."""
        from bot_mmorpg.scripts.test_model import ACTION_NAMES, ACTION_WEIGHTS

        assert len(ACTION_NAMES) == len(ACTION_WEIGHTS)

    def test_build_action_weights_standard(self):
        """Test build_action_weights for 29 standard actions."""
        from bot_mmorpg.scripts.test_model import build_action_weights

        w = build_action_weights(29)
        assert w.shape == (29,)

    def test_build_action_weights_with_mouse_10(self):
        """Test build_action_weights for 29+10 mouse actions."""
        from bot_mmorpg.scripts.test_model import build_action_weights

        w = build_action_weights(39)
        assert w.shape == (39,)
        # First 29 are keyboard+gamepad weights
        assert w[0] == 2.5  # W key
        # Mouse position weights are low (not discrete actions)
        assert w[29] < 0.5  # mouse_x
        assert w[30] < 0.5  # mouse_y
        # Mouse button weights are higher
        assert w[35] >= 0.8  # lmb
        assert w[36] >= 0.8  # rmb

    def test_build_action_weights_with_mouse_6(self):
        """Test build_action_weights for 29+6 legacy mouse actions."""
        from bot_mmorpg.scripts.test_model import build_action_weights

        w = build_action_weights(35)
        assert w.shape == (35,)

    def test_inference_engine_with_mouse_output(self):
        """Test InferenceEngine predict with 39-action model (includes mouse)."""
        from bot_mmorpg.scripts.test_model import InferenceEngine, build_action_weights

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_output = torch.randn(1, 39)
        mock_model.return_value = mock_output

        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model = mock_model
        engine.device = torch.device("cpu")
        engine.is_temporal = False
        engine.temporal_frames = 1
        engine.frame_buffer = []
        engine.enable_gamepad = False
        engine.motion_log = []
        engine.metadata = {"temporal_frames": 1}
        engine._last_evasion_time = 0.0
        engine._consecutive_stuck = 0
        engine.num_actions = 39
        engine.has_mouse_output = True
        engine.mouse_output_size = 10
        engine._action_weights = build_action_weights(39)

        screen = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        action_idx, action_val, preds = engine.predict(screen)

        # Discrete action should be from first 29 slots
        assert 0 <= action_idx < 29
        # Full predictions include mouse
        assert preds.shape == (39,)


# ============================================================================
# 9. Tauri/Launcher Production Config
# ============================================================================


class TestProductionConfig:
    """Test production configuration for Tauri and launcher."""

    def test_tauri_conf_exists(self, project_root):
        """Test tauri.conf.json exists."""
        conf = project_root / "src-tauri" / "tauri.conf.json"
        assert conf.exists()

    def test_tauri_main_html_exists(self, project_root):
        """Test main HTML file exists."""
        html = project_root / "tauri-ui" / "index.html"
        assert html.exists()

    def test_tauri_main_js_exists(self, project_root):
        """Test main JS file exists."""
        js = project_root / "tauri-ui" / "main.js"
        assert js.exists()

    def test_launcher_exists(self, project_root):
        """Test Python/Eel launcher exists."""
        launcher = project_root / "launcher" / "launcher.py"
        assert launcher.exists()

    def test_makefile_has_required_targets(self, project_root):
        """Test Makefile has all required targets."""
        makefile = (project_root / "Makefile").read_text()
        required = ["install:", "test:", "collect-data:", "train-model:", "test-model:"]
        for target in required:
            assert target in makefile, f"Missing Makefile target: {target}"

    def test_pyproject_entry_points(self, project_root):
        """Test pyproject.toml has CLI entry points."""
        toml = (project_root / "pyproject.toml").read_text()
        assert "bot-mmorpg-collect" in toml
        assert "bot-mmorpg-train" in toml
        assert "bot-mmorpg-play" in toml

    def test_settings_defaults_exist(self, project_root):
        """Test default settings file exists."""
        settings = project_root / "settings" / "defaults.yaml"
        assert settings.exists()


# ============================================================================
# 10. Full Pipeline Mock E2E
# ============================================================================


class TestMockEndToEnd:
    """Mock end-to-end test: record → train → infer (no real I/O)."""

    def test_record_train_infer_pipeline(self, tmp_path):
        """Simulate full pipeline with mocks: record → train → infer.

        Uses the 10-value mouse vector (x, y, dx, dy, vx, vy, lmb, rmb, mmb, scroll)
        to verify the entire pipeline handles variable action vector sizes.
        """
        # === RECORD ===
        from bot_mmorpg.scripts.collect_data import (
            keys_to_output,
            save_training_data,
        )
        from bot_mmorpg.scripts.mouse_capture import MouseCapture, MouseState

        training_data = []
        mock_mc = MagicMock(spec=MouseCapture)

        for i in range(20):
            screen = np.random.randint(0, 255, (270, 480, 3), dtype=np.uint8)

            # Simulate different key combos
            keys = [["W"], ["S"], ["A"], ["D"], ["W", "A"], []][i % 5]
            kb_out = keys_to_output(keys)
            gp_out = [0] * 20

            # With mouse (10-value: x, y, dx, dy, vx, vy, lmb, rmb, mmb, scroll)
            mock_mc.snapshot.return_value = MouseState(
                x=float(i) / 20,
                y=0.5,
                dx=0.01 * i,
                dy=-0.005 * i,
                vx=0.1 * i,
                vy=-0.05 * i,
                lmb=int(i % 2 == 0),
                rmb=int(i % 3 == 0),
                mmb=0,
                scroll=0.0,
            )

            parts = [np.array(kb_out), np.array(gp_out)]
            mouse_arr = mock_mc.snapshot().to_array()
            assert mouse_arr.shape == (10,)
            parts.append(mouse_arr)
            action = np.concatenate(parts)
            assert action.shape == (39,)  # 9 + 20 + 10

            training_data.append([screen, action])

        # Save
        save_path = tmp_path / "training_data-1.npy"
        assert save_training_data(training_data, save_path) is True

        # === TRAIN ===
        # Model outputs 39 values: 29 keyboard+gamepad + 10 mouse
        model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(3, 39),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Load data
        loaded = np.load(str(save_path), allow_pickle=True)
        frames = []
        actions = []
        for item in loaded:
            f = torch.tensor(item[0], dtype=torch.float32).permute(2, 0, 1) / 255.0
            a = torch.tensor(item[1], dtype=torch.float32)
            frames.append(f)
            actions.append(a)

        X = torch.stack(frames)
        Y = torch.stack(actions)

        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()

        assert loss.item() < 100

        # === INFER ===
        model.eval()
        test_frame = torch.randn(1, 3, 270, 480)
        with torch.no_grad():
            output = model(test_frame)
            probs = torch.sigmoid(output)

        assert probs.shape == (1, 39)  # 29 + 10 mouse
        assert (probs >= 0).all() and (probs <= 1).all()

        # Verify mouse outputs are at indices 29-38
        mouse_probs = probs[0, 29:]
        assert mouse_probs.shape == (10,)
        # lmb/rmb/mmb should be valid probabilities
        assert (mouse_probs[6:9] >= 0).all() and (mouse_probs[6:9] <= 1).all()
