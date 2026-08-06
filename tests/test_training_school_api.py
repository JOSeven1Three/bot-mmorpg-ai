"""
Tests for Training School API (Bridge Server and Handlers)

Tests the JSON-RPC bridge that connects Tauri frontend to Python backend.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_torch():
    """Mock torch module for tests."""
    with patch.dict("sys.modules", {"torch": MagicMock()}):
        yield


class TestRPCParsing:
    """Test JSON-RPC request/response parsing."""

    def test_parse_valid_request(self):
        """Test parsing a valid JSON-RPC request."""
        from bot_mmorpg.bridge.server import BridgeServer

        server = BridgeServer()
        line = '{"jsonrpc": "2.0", "method": "system.ping", "params": {}, "id": 1}'

        request = server.parse_request(line)

        assert request is not None
        assert request.jsonrpc == "2.0"
        assert request.method == "system.ping"
        assert request.params == {}
        assert request.id == 1

    def test_parse_request_with_params(self):
        """Test parsing request with parameters."""
        from bot_mmorpg.bridge.server import BridgeServer

        server = BridgeServer()
        line = '{"jsonrpc": "2.0", "method": "config.get_profile", "params": {"game_id": "world_of_warcraft"}, "id": 2}'

        request = server.parse_request(line)

        assert request is not None
        assert request.method == "config.get_profile"
        assert request.params == {"game_id": "world_of_warcraft"}

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns None."""
        from bot_mmorpg.bridge.server import BridgeServer

        server = BridgeServer()
        line = "not valid json"

        request = server.parse_request(line)

        assert request is None

    def test_parse_missing_method(self):
        """Test parsing request without method returns None."""
        from bot_mmorpg.bridge.server import BridgeServer

        server = BridgeServer()
        line = '{"jsonrpc": "2.0", "params": {}}'

        request = server.parse_request(line)

        assert request is None


class TestRPCResponse:
    """Test RPC response formatting."""

    def test_success_response_json(self):
        """Test successful response JSON formatting."""
        from bot_mmorpg.bridge.server import RPCResponse

        response = RPCResponse(result={"status": "ok"}, id=1)
        json_str = response.to_json()
        data = json.loads(json_str)

        assert data["jsonrpc"] == "2.0"
        assert data["result"] == {"status": "ok"}
        assert data["id"] == 1
        assert "error" not in data

    def test_error_response_json(self):
        """Test error response JSON formatting."""
        from bot_mmorpg.bridge.server import RPCResponse

        response = RPCResponse(
            error={"code": -32601, "message": "Method not found"}, id=1
        )
        json_str = response.to_json()
        data = json.loads(json_str)

        assert data["jsonrpc"] == "2.0"
        assert data["error"]["code"] == -32601
        assert data["id"] == 1
        assert "result" not in data


class TestEventEmitter:
    """Test event emission for real-time updates."""

    def test_emit_event_format(self, capsys):
        """Test event emission format."""
        from bot_mmorpg.bridge.server import EventEmitter

        emitter = EventEmitter()
        emitter.emit("training:metrics", {"epoch": 1, "loss": 0.5})

        captured = capsys.readouterr()
        assert "@@EVENT@@" in captured.out
        assert "training:metrics" in captured.out
        assert "epoch" in captured.out


class TestCommandHandler:
    """Test command handler functionality."""

    def test_system_ping(self):
        """Test system ping returns status ok."""
        from bot_mmorpg.bridge.handlers import CommandHandler

        handler = CommandHandler()
        result = handler.handle_system_ping()

        assert result["status"] == "ok"
        assert "timestamp" in result

    def test_system_get_version(self):
        """Test version information retrieval."""
        pytest.importorskip("torch", reason="PyTorch not installed")

        from bot_mmorpg.bridge.handlers import CommandHandler

        handler = CommandHandler()
        result = handler.handle_system_get_version()

        assert "app_version" in result
        assert "python_version" in result
        assert "pytorch_version" in result

    def test_training_state_initial(self):
        """Test initial training state."""
        from bot_mmorpg.bridge.handlers import CommandHandler

        handler = CommandHandler()
        result = handler.handle_training_get_state()

        assert result["is_training"] is False
        assert result["epoch"] == 0

    def test_inference_state_initial(self):
        """Test initial inference state."""
        from bot_mmorpg.bridge.handlers import CommandHandler

        handler = CommandHandler()
        result = handler.handle_inference_get_state()

        assert result["is_running"] is False
        assert result["model_loaded"] is False

    def test_training_stop(self):
        """Test training stop command."""
        from bot_mmorpg.bridge.handlers import CommandHandler

        handler = CommandHandler()
        result = handler.handle_training_stop()

        assert result["status"] == "stopped"

    def test_inference_stop(self):
        """Test inference stop command."""
        from bot_mmorpg.bridge.handlers import CommandHandler

        handler = CommandHandler()
        result = handler.handle_inference_stop()

        assert result["status"] == "stopped"

    def test_inference_emergency_stop(self):
        """Test emergency stop command."""
        from bot_mmorpg.bridge.handlers import CommandHandler

        handler = CommandHandler()
        result = handler.handle_inference_emergency_stop()

        assert result["status"] == "emergency_stopped"

    def test_get_diablo_profile_exposes_helltides_overlay(self):
        """Test Diablo IV profile exposes the companion Helltides overlay."""
        from bot_mmorpg.bridge.handlers import CommandHandler

        handler = CommandHandler()
        profile = handler.handle_config_get_profile("diablo_4")

        overlay = profile["external_overlays"]["helltides"]
        assert overlay["name"] == "Diablo 4 Helltides Overlay"
        assert overlay["safety_model"] == "screen_only"
        assert overlay["run_command"][-1] == "tools/run-overlay.ps1"

    def test_launch_region_catcher_is_screen_only(self):
        """Training School can launch Region Catcher with screen-only defaults."""
        from bot_mmorpg.bridge.handlers import CommandHandler

        handler = CommandHandler()
        fake_process = MagicMock(pid=4321)
        fake_process.poll.return_value = None

        with patch("subprocess.Popen", return_value=fake_process) as popen:
            result = handler.handle_training_launch_region_catcher(
                game_id="diablo_4",
                capture_region=[0, 0, 1920, 1080],
            )

        assert result["status"] == "started"
        assert result["screen_only"] is True
        assert result["tool"] == "region_catcher"
        args = popen.call_args.kwargs["args"]
        assert "--game" in args
        assert "diablo_4" in args
        assert "--region" in args

    def test_capture_profile_diagnostics_returns_bundle_paths(self, tmp_path):
        """One-click diagnostics returns the annotated bundle paths."""
        from bot_mmorpg.bridge.handlers import CommandHandler

        handler = CommandHandler()
        fake_screen = MagicMock()
        fake_screen.size = 1
        fake_dir = tmp_path / "diagnostics" / "diablo_4" / "20260806_120000"

        with (
            patch("bot_mmorpg.scripts.grabscreen.grab_screen", return_value=fake_screen),
            patch(
                "bot_mmorpg.scripts.collect_data.save_profile_diagnostics",
                return_value=fake_dir,
            ) as save_diag,
        ):
            result = handler.handle_training_capture_profile_diagnostics(
                game_id="diablo_4",
                output_dir=str(tmp_path),
                capture_region=[0, 0, 1920, 1080],
            )

        assert result["status"] == "saved"
        assert result["screen_only"] is True
        assert result["diagnostics_dir"].endswith("20260806_120000")
        assert result["annotated_image"].endswith("annotated_regions.png")
        save_diag.assert_called_once()

    def test_dry_run_preview_enforces_cap_and_no_input(self):
        """Dry-run preview always launches with no-input safety flags and a cap."""
        from bot_mmorpg.bridge.handlers import CommandHandler

        handler = CommandHandler()
        fake_process = MagicMock(pid=9876)
        fake_process.poll.return_value = None

        with patch("subprocess.Popen", return_value=fake_process) as popen:
            result = handler.handle_inference_launch_dry_run_preview(
                model_path="C:/models/diablo4_best.pth",
                max_frames=999,
            )

        assert result["dry_run"] is True
        assert result["sends_input"] is False
        assert result["screen_only"] is True
        assert result["max_frames"] == 300
        args = popen.call_args.kwargs["args"]
        assert "--dry-run" in args
        assert "--no-gamepad" in args
        assert "--max-frames" in args
        assert "300" in args


class TestBridgeServerHandling:
    """Test bridge server request handling."""

    def test_handle_unknown_method(self):
        """Test handling of unknown method."""
        from bot_mmorpg.bridge.server import BridgeServer, RPCRequest

        server = BridgeServer()
        request = RPCRequest(jsonrpc="2.0", method="unknown.method", params={}, id=1)

        response = server.handle_request(request)

        assert response.error is not None
        assert response.error["code"] == -32601
        assert "Method not found" in response.error["message"]

    def test_handle_system_ping(self):
        """Test handling system.ping method."""
        from bot_mmorpg.bridge.server import BridgeServer, RPCRequest

        server = BridgeServer()
        request = RPCRequest(jsonrpc="2.0", method="system.ping", params={}, id=1)

        response = server.handle_request(request)

        assert response.error is None
        assert response.result["status"] == "ok"


class TestChatHandler:
    """Test chat command handling."""

    def test_chat_local_fallback(self):
        """Test local chat fallback responses."""
        from bot_mmorpg.bridge.handlers import CommandHandler

        handler = CommandHandler()

        # Test accuracy question
        result = handler._chat_local("My accuracy is low, what should I do?", "")
        assert "accuracy" in result.lower() or "data" in result.lower()

        # Test imbalance question
        result = handler._chat_local("How to fix class imbalance?", "")
        assert "imbalance" in result.lower() or "class" in result.lower()

        # Test generic question
        result = handler._chat_local("Hello", "")
        assert len(result) > 0
