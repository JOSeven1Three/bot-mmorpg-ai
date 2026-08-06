"""
Command Handlers

Handles all commands from Tauri frontend.
"""

import base64
import io
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bridge.handlers")

# Lazy imports to speed up startup
_torch = None
_PIL = None
_version_utils = None


def _get_version_utils():
    global _version_utils
    if _version_utils is None:
        try:
            from bot_mmorpg.utils import version as version_module

            _version_utils = version_module
        except ImportError:
            _version_utils = False  # Mark as unavailable
    return _version_utils if _version_utils else None


def _get_torch():
    global _torch
    if _torch is None:
        import torch

        _torch = torch
    return _torch


def _get_pil():
    global _PIL
    if _PIL is None:
        from PIL import Image

        _PIL = Image
    return _PIL


@dataclass
class TrainingState:
    """Current training state."""

    is_training: bool = False
    phase: str = ""
    epoch: int = 0
    total_epochs: int = 0
    batch: int = 0
    total_batches: int = 0
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    eta_seconds: int = 0


@dataclass
class InferenceState:
    """Current inference state."""

    is_running: bool = False
    model_loaded: bool = False
    model_name: str = ""
    fps: float = 0.0
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0
    actions_count: int = 0


class CommandHandler:
    """
    Handles all bridge commands from Tauri.

    Commands are organized by category:
    - config.*  : Configuration and settings
    - training.*: Training operations
    - inference.*: Inference operations
    - chat.*    : AI chat operations
    - visual.*  : Visualization operations
    """

    def __init__(self):
        self.event_emitter = None

        # State
        self.training_state = TrainingState()
        self.inference_state = InferenceState()

        # Resources
        self._model = None
        self._trainer = None
        self._inference_engine = None
        self._training_thread = None
        self._inference_thread = None
        self._stop_training = threading.Event()
        self._stop_inference = threading.Event()

        # Caches
        self._hardware_info = None
        self._game_profiles = None
        self._workflow_processes: Dict[str, subprocess.Popen] = {}

    def set_event_emitter(self, emitter):
        """Set the event emitter for real-time updates."""
        self.event_emitter = emitter

    def emit(self, event_type: str, data: Any):
        """Emit an event if emitter is available."""
        if self.event_emitter:
            self.event_emitter.emit(event_type, data)

    def cleanup(self):
        """Clean up resources on shutdown."""
        self._stop_training.set()
        self._stop_inference.set()

        if self._training_thread and self._training_thread.is_alive():
            self._training_thread.join(timeout=5)

        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=5)

        for process in self._workflow_processes.values():
            try:
                if process.poll() is None:
                    process.terminate()
            except Exception:
                logger.debug("Workflow process cleanup failed", exc_info=True)

    def _script_path(self, name: str) -> Path:
        return Path(__file__).resolve().parents[1] / "scripts" / name

    def _parse_capture_region(
        self, capture_region: Optional[List[int]], fallback_resolution: List[int]
    ) -> tuple[int, int, int, int]:
        if capture_region and len(capture_region) == 4:
            return tuple(int(v) for v in capture_region)
        return (0, 0, int(fallback_resolution[0]), int(fallback_resolution[1]))

    def _launch_workflow_process(
        self, process_key: str, args: List[str], cwd: Optional[Path] = None
    ) -> Dict[str, Any]:
        previous = self._workflow_processes.get(process_key)
        if previous and previous.poll() is None:
            return {
                "status": "already_running",
                "pid": previous.pid,
                "command": args,
            }

        kwargs: Dict[str, Any] = {
            "args": args,
            "cwd": str(cwd or Path(args[1]).resolve().parents[2]),
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        process = subprocess.Popen(**kwargs)
        self._workflow_processes[process_key] = process
        return {
            "status": "started",
            "pid": process.pid,
            "command": args,
        }

    # =========================================================================
    # CONFIG COMMANDS
    # =========================================================================

    def handle_config_get_hardware(self) -> Dict[str, Any]:
        """Get hardware information."""
        if self._hardware_info is None:
            from bot_mmorpg.config import HardwareDetector

            detector = HardwareDetector()
            info = detector.detect()

            self._hardware_info = {
                "platform": info.platform,
                "cpu_cores": info.cpu_cores,
                "ram_mb": info.ram_mb,
                "gpu": (
                    {
                        "name": info.gpu.name if info.gpu else None,
                        "vram_mb": info.gpu.vram_mb if info.gpu else 0,
                        "cuda_available": info.gpu.cuda_available
                        if info.gpu
                        else False,
                        "cuda_version": info.gpu.cuda_version if info.gpu else None,
                    }
                    if info.gpu
                    else None
                ),
                "tier": info.tier.value,
                "summary": info.summary(),
            }

        return self._hardware_info

    def handle_config_list_games(self) -> List[Dict[str, str]]:
        """List available game profiles."""
        from bot_mmorpg.config import GameProfileLoader

        loader = GameProfileLoader()
        return loader.list_games()

    def handle_config_get_profile(self, game_id: str) -> Dict[str, Any]:
        """Get a game profile."""
        from bot_mmorpg.config import GameProfileLoader

        loader = GameProfileLoader()
        profile = loader.load(game_id)

        return {
            "id": profile.id,
            "name": profile.name,
            "publisher": profile.publisher,
            "tasks": [
                {
                    "id": task_name,
                    "name": task_name.replace("_", " ").title(),
                    "description": task.description,
                    "temporal": task.temporal,
                    "recommended_architecture": task.recommended_architecture,
                }
                for task_name, task in profile.tasks.items()
            ],
            "recommended_architecture": profile.recommended_architecture,
            "minimum_samples": profile.minimum_samples,
            "external_overlays": profile.external_overlays,
            "hardware_tiers": {
                tier: {
                    "architecture": cfg.architecture,
                    "batch_size": cfg.batch_size,
                    "input_size": cfg.input_size,
                    "temporal_frames": cfg.temporal_frames,
                }
                for tier, cfg in profile.hardware_tiers.items()
            },
        }

    def handle_config_get_recommendation(
        self, game_id: str, task: str
    ) -> Dict[str, Any]:
        """Get model recommendation for game/task."""
        from bot_mmorpg.config import ModelSelector

        selector = ModelSelector()
        rec = selector.recommend(game_id=game_id, task=task)

        return {
            "architecture": rec.architecture.value,
            "architecture_name": rec.architecture.display_name,
            "confidence": rec.confidence,
            "reasons": rec.reasons,
            "warnings": rec.warnings,
            "estimated_speed": rec.estimated_speed,
            "estimated_accuracy": rec.estimated_accuracy,
            "estimated_vram_mb": rec.estimated_vram_mb,
            "recommended_input_size": list(rec.recommended_input_size),
            "recommended_temporal_frames": rec.recommended_temporal_frames,
            "recommended_batch_size": rec.recommended_batch_size,
        }

    def handle_config_save_session(self, config: Dict[str, Any]) -> str:
        """Save session configuration."""
        from bot_mmorpg.config import SettingsManager

        manager = SettingsManager()
        session = manager.create_session_config(
            game_id=config["game_id"],
            task=config["task"],
            overrides=config.get("overrides", {}),
        )
        path = manager.save_session_config(session)
        return str(path)

    # =========================================================================
    # TRAINING COMMANDS
    # =========================================================================

    def handle_training_start(
        self,
        data_path: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Start training with given configuration."""
        if self.training_state.is_training:
            raise RuntimeError("Training already in progress")

        self._stop_training.clear()

        # Start training in background thread
        self._training_thread = threading.Thread(
            target=self._training_loop,
            args=(data_path, config),
            daemon=True,
        )
        self._training_thread.start()

        self.training_state.is_training = True

        return {"status": "started"}

    def _training_loop(self, data_path: str, config: Dict[str, Any]):
        """Background training loop with event emission."""
        try:
            _get_torch()
            from bot_mmorpg.scripts.models_pytorch import create_model
            from bot_mmorpg.training import CurriculumConfig, CurriculumTrainer

            # Create model
            model = create_model(
                architecture=config.get("architecture", "efficientnet_lstm"),
                num_classes=config.get("num_classes", 12),
                input_size=tuple(config.get("input_size", [224, 224])),
                temporal_frames=config.get("temporal_frames", 4),
                pretrained=True,
            )

            # Create curriculum config
            curriculum = CurriculumConfig.default(total_epochs=config.get("epochs", 50))

            # Create trainer
            trainer = CurriculumTrainer(model, curriculum)

            # Set callbacks for event emission
            def on_epoch_end(metrics):
                self.training_state.epoch = metrics.epoch
                self.training_state.train_loss = metrics.train_loss
                self.training_state.train_accuracy = metrics.train_accuracy
                self.training_state.val_loss = metrics.val_loss
                self.training_state.val_accuracy = metrics.val_accuracy
                self.training_state.learning_rate = metrics.learning_rate

                self.emit(
                    "training:metrics",
                    {
                        "epoch": metrics.epoch,
                        "phase": metrics.phase.value,
                        "train_loss": metrics.train_loss,
                        "train_accuracy": metrics.train_accuracy,
                        "val_loss": metrics.val_loss,
                        "val_accuracy": metrics.val_accuracy,
                        "learning_rate": metrics.learning_rate,
                    },
                )

            trainer._on_epoch_end = on_epoch_end

            # TODO: Load actual data loaders
            # For now, emit completion
            self.emit("training:complete", {"status": "success"})

        except Exception as e:
            logger.exception("Training error")
            self.emit("training:error", {"error": str(e)})
        finally:
            self.training_state.is_training = False

    def handle_training_stop(self) -> Dict[str, Any]:
        """Stop current training."""
        self._stop_training.set()
        self.training_state.is_training = False
        return {"status": "stopped"}

    def handle_training_get_state(self) -> Dict[str, Any]:
        """Get current training state."""
        return asdict(self.training_state)

    def handle_training_launch_region_catcher(
        self,
        game_id: str = "diablo_4",
        capture_region: Optional[List[int]] = None,
        screenshot_path: Optional[str] = None,
        roi_overrides_path: Optional[str] = None,
        no_autosave: bool = False,
    ) -> Dict[str, Any]:
        """Launch the interactive Region Catcher for HUD calibration."""
        from bot_mmorpg.config import GameProfileLoader

        profile = GameProfileLoader().load(game_id)
        resolved_region = self._parse_capture_region(
            capture_region, profile.typical_resolution
        )
        script = self._script_path("region_catcher.py")
        args = [
            sys.executable,
            str(script),
            "--game",
            game_id,
            "--region",
            ",".join(str(v) for v in resolved_region),
        ]
        if screenshot_path:
            args.extend(["--screenshot", screenshot_path])
        if roi_overrides_path:
            args.extend(["--roi-overrides", roi_overrides_path])
        if no_autosave:
            args.append("--no-autosave")

        result = self._launch_workflow_process(f"region_catcher:{game_id}", args)
        result.update(
            {
                "game_id": game_id,
                "capture_region": list(resolved_region),
                "screen_only": True,
                "writes_profile": True,
                "tool": "region_catcher",
            }
        )
        return result

    def handle_training_capture_profile_diagnostics(
        self,
        game_id: str = "diablo_4",
        capture_region: Optional[List[int]] = None,
        output_dir: Optional[str] = None,
        roi_overrides_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Capture one annotated profile-diagnostics bundle."""
        from bot_mmorpg.config import GameProfileLoader
        from bot_mmorpg.scripts.collect_data import (
            load_roi_overrides,
            save_profile_diagnostics,
        )
        from bot_mmorpg.scripts.grabscreen import grab_screen

        loader = GameProfileLoader()
        profile = loader.load(game_id)
        resolved_region = self._parse_capture_region(
            capture_region, profile.typical_resolution
        )
        important_regions = profile.important_regions
        if roi_overrides_path:
            important_regions = load_roi_overrides(
                Path(roi_overrides_path), resolved_region, important_regions
            )

        screen = grab_screen(region=resolved_region)
        if screen is None or getattr(screen, "size", 0) == 0:
            raise RuntimeError("Screen capture returned empty image")

        root = Path(output_dir) if output_dir else Path("artifacts")
        diag_dir = save_profile_diagnostics(
            screen,
            profile,
            resolved_region,
            root,
            important_regions=important_regions,
        )
        return {
            "status": "saved",
            "game_id": game_id,
            "capture_region": list(resolved_region),
            "screen_only": True,
            "diagnostics_dir": str(diag_dir),
            "raw_image": str(diag_dir / "raw.png"),
            "annotated_image": str(diag_dir / "annotated_regions.png"),
            "metadata_path": str(diag_dir / "metadata.json"),
            "roi_template_path": str(diag_dir / "roi_overrides.template.json"),
        }

    # =========================================================================
    # INFERENCE COMMANDS
    # =========================================================================

    def handle_inference_load_model(self, model_path: str) -> Dict[str, Any]:
        """Load a model for inference."""
        from bot_mmorpg.inference import InferenceEngine

        self._inference_engine = InferenceEngine.from_checkpoint(
            model_path,
            confidence_threshold=0.7,
            cooldown_ms=100,
        )

        self.inference_state.model_loaded = True
        self.inference_state.model_name = Path(model_path).stem

        return {
            "status": "loaded",
            "model_name": self.inference_state.model_name,
            "architecture": self._inference_engine.metadata.architecture,
            "num_classes": self._inference_engine.metadata.num_classes,
            "class_names": self._inference_engine.metadata.class_names,
        }

    def handle_inference_start(self) -> Dict[str, Any]:
        """Start inference loop."""
        if not self.inference_state.model_loaded:
            raise RuntimeError("No model loaded")

        if self.inference_state.is_running:
            raise RuntimeError("Inference already running")

        self._stop_inference.clear()
        self.inference_state.is_running = True

        # Start inference in background
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
        )
        self._inference_thread.start()

        return {"status": "started"}

    def _inference_loop(self):
        """Background inference loop."""
        try:
            frame_times = []
            while not self._stop_inference.is_set():
                start = time.perf_counter()

                # TODO: Capture actual screen
                # For now, simulate
                time.sleep(0.033)  # ~30 FPS

                elapsed = time.perf_counter() - start
                frame_times.append(elapsed)

                if len(frame_times) > 30:
                    frame_times.pop(0)

                avg_time = sum(frame_times) / len(frame_times)
                self.inference_state.fps = 1.0 / avg_time if avg_time > 0 else 0
                self.inference_state.avg_latency_ms = avg_time * 1000

                # Emit stats periodically
                self.emit(
                    "inference:stats",
                    {
                        "fps": self.inference_state.fps,
                        "latency_ms": self.inference_state.avg_latency_ms,
                        "actions_count": self.inference_state.actions_count,
                    },
                )

        except Exception as e:
            logger.exception("Inference error")
            self.emit("inference:error", {"error": str(e)})
        finally:
            self.inference_state.is_running = False

    def handle_inference_stop(self) -> Dict[str, Any]:
        """Stop inference loop."""
        self._stop_inference.set()
        self.inference_state.is_running = False
        return {"status": "stopped"}

    def handle_inference_emergency_stop(self) -> Dict[str, Any]:
        """Emergency stop - immediately halt all inference."""
        self._stop_inference.set()
        if self._inference_engine:
            self._inference_engine.set_emergency_stop(True)
        self.inference_state.is_running = False
        return {"status": "emergency_stopped"}

    def handle_inference_get_state(self) -> Dict[str, Any]:
        """Get current inference state."""
        return asdict(self.inference_state)

    def handle_inference_launch_dry_run_preview(
        self,
        model_path: str,
        max_frames: int = 120,
        width: int = 1920,
        height: int = 1080,
        cpu: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Launch capped dry-run inference preview with no game input."""
        capped_frames = max(1, min(int(max_frames), 300))
        script = self._script_path("test_model.py")
        args = [
            sys.executable,
            str(script),
            "--model",
            model_path,
            "--dry-run",
            "--no-gamepad",
            "--max-frames",
            str(capped_frames),
            "--width",
            str(int(width)),
            "--height",
            str(int(height)),
        ]
        if cpu:
            args.append("--cpu")
        if verbose:
            args.append("--verbose")

        result = self._launch_workflow_process(
            f"dry_run_preview:{Path(model_path).resolve()}",
            args,
        )
        result.update(
            {
                "model_path": model_path,
                "dry_run": True,
                "screen_only": True,
                "sends_input": False,
                "max_frames": capped_frames,
            }
        )
        return result

    # =========================================================================
    # VISUALIZATION COMMANDS
    # =========================================================================

    def handle_visual_get_attention_map(self, frame_base64: str) -> Dict[str, Any]:
        """Generate attention map for a frame."""
        from bot_mmorpg.visualization import generate_attention_map

        # Decode frame
        Image = _get_pil()
        frame_bytes = base64.b64decode(frame_base64)
        frame = Image.open(io.BytesIO(frame_bytes))

        # Generate attention map
        attention = generate_attention_map(
            self._inference_engine.model if self._inference_engine else None,
            frame,
        )

        # Encode result
        buffer = io.BytesIO()
        attention.save(buffer, format="PNG")
        attention_base64 = base64.b64encode(buffer.getvalue()).decode()

        return {"attention_map": attention_base64}

    def handle_visual_get_prediction_overlay(self, frame_base64: str) -> Dict[str, Any]:
        """Generate prediction overlay for a frame."""
        if not self._inference_engine:
            return {"error": "No model loaded"}

        from bot_mmorpg.visualization import generate_prediction_overlay

        # Decode frame
        Image = _get_pil()
        frame_bytes = base64.b64decode(frame_base64)
        frame = Image.open(io.BytesIO(frame_bytes))

        # Get prediction and overlay
        result = self._inference_engine.predict(frame)
        overlay = generate_prediction_overlay(frame, result)

        # Encode result
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        overlay_base64 = base64.b64encode(buffer.getvalue()).decode()

        return {
            "overlay": overlay_base64,
            "prediction": {
                "action": result.action_name if result else None,
                "confidence": result.confidence if result else 0,
                "all_probs": result.all_probabilities if result else {},
            },
        }

    # =========================================================================
    # CHAT COMMANDS
    # =========================================================================

    def handle_chat_send_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        provider: str = "claude",
    ) -> Dict[str, Any]:
        """Send a message to the AI assistant."""
        # Build context
        full_context = self._build_chat_context(context or {})

        # Route to provider
        if provider == "claude":
            response = self._chat_claude(message, full_context)
        elif provider == "ollama":
            response = self._chat_ollama(message, full_context)
        else:
            response = self._chat_local(message, full_context)

        return {"response": response}

    def _build_chat_context(self, user_context: Dict) -> str:
        """Build context string for AI chat."""
        parts = ["You are an AI assistant for the BOT-MMORPG-AI training system."]

        # Add hardware context
        if self._hardware_info:
            parts.append(
                f"User hardware: {self._hardware_info['tier']} tier, "
                f"GPU: {self._hardware_info['gpu']['name'] if self._hardware_info['gpu'] else 'None'}"
            )

        # Add training context
        if self.training_state.is_training:
            parts.append(
                f"Training in progress: Phase {self.training_state.phase}, "
                f"Epoch {self.training_state.epoch}/{self.training_state.total_epochs}, "
                f"Accuracy: {self.training_state.train_accuracy:.2%}"
            )

        # Add user context
        if user_context.get("game_id"):
            parts.append(f"Selected game: {user_context['game_id']}")
        if user_context.get("task"):
            parts.append(f"Selected task: {user_context['task']}")

        return "\n".join(parts)

    def _chat_claude(self, message: str, context: str) -> str:
        """Chat using Claude API."""
        try:
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=context,
                messages=[{"role": "user", "content": message}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"Error: {e}. Please check your API key."

    def _chat_ollama(self, message: str, context: str) -> str:
        """Chat using local Ollama."""
        try:
            import requests

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2",
                    "prompt": f"{context}\n\nUser: {message}\n\nAssistant:",
                    "stream": False,
                },
                timeout=60,
            )
            return response.json().get("response", "No response")
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"Error: {e}. Make sure Ollama is running."

    def _chat_local(self, message: str, context: str) -> str:
        """Fallback local response."""
        # Simple pattern matching for common questions
        message_lower = message.lower()

        if "accuracy" in message_lower and "low" in message_lower:
            return (
                "Low accuracy can be caused by:\n"
                "1. Insufficient training data\n"
                "2. Class imbalance\n"
                "3. Wrong model architecture for your task\n"
                "4. Learning rate too high or low\n\n"
                "Try collecting more diverse training data first."
            )

        if "dodge" in message_lower or "imbalance" in message_lower:
            return (
                "For class imbalance, you can:\n"
                "1. Record more samples of the minority class\n"
                "2. Enable class weighting (Settings > Training)\n"
                "3. Use data augmentation\n"
                "4. Try oversampling the minority class"
            )

        return (
            "I can help with training, models, and game strategies. "
            "For more detailed assistance, configure the Claude API in Settings."
        )

    # =========================================================================
    # SYSTEM COMMANDS
    # =========================================================================

    def handle_system_ping(self) -> Dict[str, Any]:
        """Health check."""
        return {"status": "ok", "timestamp": time.time()}

    def handle_system_get_version(self) -> Dict[str, Any]:
        """Get version information."""
        torch = _get_torch()
        version_utils = _get_version_utils()

        app_version = "1.0.0"
        if version_utils:
            app_version = version_utils.get_current_version()

        return {
            "app_version": app_version,
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "pytorch_version": torch.__version__ if torch else "not loaded",
            "cuda_available": torch.cuda.is_available() if torch else False,
        }

    def handle_system_check_updates(self) -> Dict[str, Any]:
        """Check for available updates."""
        version_utils = _get_version_utils()

        if not version_utils:
            return {
                "available": False,
                "current_version": "1.0.0",
                "error": "Version utilities not available",
            }

        try:
            info = version_utils.check_for_updates(timeout=10.0)
            return {
                "available": info.available,
                "current_version": info.current_version,
                "latest_version": info.latest_version,
                "release_url": info.release_url,
                "release_notes": info.release_notes,
                "error": info.error,
            }
        except Exception as e:
            logger.error(f"Update check failed: {e}")
            return {
                "available": False,
                "current_version": version_utils.get_current_version(),
                "error": str(e),
            }
