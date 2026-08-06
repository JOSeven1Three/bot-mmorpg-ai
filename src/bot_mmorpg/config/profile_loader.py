"""
Game Profile Loader

Loads and manages game-specific configuration profiles.
"""

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class TaskConfig:
    """Configuration for a specific task type."""

    name: str
    description: str
    priority: str
    temporal: bool
    recommended_architecture: str
    augmentation: List[str]
    fps_target: int


@dataclass
class HardwareTierConfig:
    """Hardware-specific configuration."""

    architecture: str
    batch_size: int
    input_size: List[int]
    temporal_frames: int
    workers: int


@dataclass
class GameProfile:
    """Complete game profile with all configurations."""

    # Game identification
    id: str
    name: str
    publisher: str

    # Display settings
    typical_resolution: List[int]
    ui_scale_range: List[float]
    important_regions: Dict[str, List[float]]

    # Input configuration
    action_space: str
    num_actions: int
    requires_mouse: bool
    mouse_precision: str
    default_bindings: List[Dict[str, str]]

    # Training defaults
    recommended_architecture: str
    recommended_input_size: List[int]
    temporal_frames: int
    minimum_samples: int
    class_balance_tolerance: float
    learning_rate: Dict[str, Any]

    # Hardware tier configs
    hardware_tiers: Dict[str, HardwareTierConfig]

    # Task configs
    tasks: Dict[str, TaskConfig]

    # Optional companion tools such as external overlays.
    external_overlays: Dict[str, Dict[str, Any]]

    # Source path
    profile_path: Optional[Path] = None

    def get_tier_config(self, tier: str) -> HardwareTierConfig:
        """Get configuration for a specific hardware tier."""
        if tier not in self.hardware_tiers:
            # Fallback to medium
            tier = "medium"
        return self.hardware_tiers[tier]

    def get_task_config(self, task: str) -> Optional[TaskConfig]:
        """Get configuration for a specific task."""
        return self.tasks.get(task)

    def list_tasks(self) -> List[str]:
        """List available tasks for this game."""
        return list(self.tasks.keys())


class GameProfileLoader:
    """
    Loads and manages game profiles from YAML files.

    Usage:
        loader = GameProfileLoader()
        profile = loader.load("world_of_warcraft")
        print(profile.recommended_architecture)
    """

    def __init__(self, profiles_dir: Optional[Path] = None):
        """
        Initialize the profile loader.

        Args:
            profiles_dir: Directory containing game profiles.
                          Defaults to project's game_profiles directory.
        """
        if profiles_dir is None:
            # Find project root and game_profiles
            self.profiles_dir = self._find_profiles_dir()
        else:
            self.profiles_dir = Path(profiles_dir)

        self._profiles_cache: Dict[str, GameProfile] = {}
        self._index: Optional[Dict] = None

    def _find_profiles_dir(self) -> Path:
        """Find the game_profiles directory."""
        # Try relative to this file
        current = Path(__file__).parent
        for _ in range(5):  # Go up max 5 levels
            candidate = current / "game_profiles"
            if candidate.exists():
                return candidate
            # Also check parent directories
            candidate = current.parent / "game_profiles"
            if candidate.exists():
                return candidate
            current = current.parent

        # Fallback to current working directory
        cwd_candidate = Path.cwd() / "game_profiles"
        if cwd_candidate.exists():
            return cwd_candidate

        raise FileNotFoundError(
            "Could not find game_profiles directory. "
            "Please specify profiles_dir explicitly."
        )

    def _load_index(self) -> Dict:
        """Load the profiles index."""
        if self._index is not None:
            return self._index

        index_path = self.profiles_dir / "index.yaml"
        if not index_path.exists():
            raise FileNotFoundError(f"Profiles index not found: {index_path}")

        with open(index_path) as f:
            self._index = yaml.safe_load(f)

        return self._index

    def list_games(self) -> List[Dict[str, str]]:
        """
        List all available game profiles.

        Returns:
            List of dicts with 'id', 'name', and 'status' keys
        """
        index = self._load_index()
        games = []

        for game_id, info in index.get("profiles", {}).items():
            games.append(
                {
                    "id": game_id,
                    "name": info.get("name", game_id),
                    "status": info.get("status", "unknown"),
                }
            )

        return games

    def load(self, game_id: str) -> GameProfile:
        """
        Load a game profile by ID.

        Args:
            game_id: The game identifier (e.g., "world_of_warcraft")

        Returns:
            GameProfile object

        Raises:
            FileNotFoundError: If profile doesn't exist
            ValueError: If profile is invalid
        """
        # Check cache
        if game_id in self._profiles_cache:
            return self._profiles_cache[game_id]

        # Find profile path
        profile_path = self.profiles_dir / game_id / "profile.yaml"

        if not profile_path.exists():
            raise FileNotFoundError(f"Game profile not found: {profile_path}")

        # Load and parse
        with open(profile_path) as f:
            data = yaml.safe_load(f)

        profile = self._parse_profile(data, profile_path)
        self._profiles_cache[game_id] = profile

        return profile

    def load_raw_data(self, game_id: str) -> Dict[str, Any]:
        """Load raw YAML data for a game profile."""
        profile_path = self.profiles_dir / game_id / "profile.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"Game profile not found: {profile_path}")

        with open(profile_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def save_display_regions(
        self,
        game_id: str,
        important_regions: Dict[str, List[float]],
        backup: bool = True,
    ) -> Path:
        """
        Persist updated display regions back into the profile YAML.

        Args:
            game_id: Profile identifier
            important_regions: Normalized [x, y, w, h] regions
            backup: If True, create a one-time .bak file before first save

        Returns:
            Path to the written profile
        """
        profile_path = self.profiles_dir / game_id / "profile.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"Game profile not found: {profile_path}")

        data = self.load_raw_data(game_id)
        data.setdefault("display", {})
        data["display"]["important_regions"] = important_regions

        if backup:
            backup_path = profile_path.with_suffix(".yaml.bak")
            if not backup_path.exists():
                shutil.copyfile(profile_path, backup_path)

        with open(profile_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)

        self._profiles_cache.pop(game_id, None)
        return profile_path

    def _parse_profile(self, data: Dict, profile_path: Path) -> GameProfile:
        """Parse YAML data into a GameProfile object."""
        try:
            # Parse hardware tiers
            hardware_tiers = {}
            for tier_name, tier_data in data.get("hardware_tiers", {}).items():
                hardware_tiers[tier_name] = HardwareTierConfig(
                    architecture=tier_data.get("architecture", "efficientnet_lstm"),
                    batch_size=tier_data.get("batch_size", 16),
                    input_size=tier_data.get("input_size", [224, 224]),
                    temporal_frames=tier_data.get("temporal_frames", 4),
                    workers=tier_data.get("workers", 4),
                )

            # Parse tasks
            tasks = {}
            for task_name, task_data in data.get("tasks", {}).items():
                tasks[task_name] = TaskConfig(
                    name=task_name,
                    description=task_data.get("description", ""),
                    priority=task_data.get("priority", "accuracy"),
                    temporal=task_data.get("temporal", False),
                    recommended_architecture=task_data.get(
                        "recommended_architecture", "efficientnet_lstm"
                    ),
                    augmentation=task_data.get("augmentation", []),
                    fps_target=task_data.get("fps_target", 30),
                )

            game_data = data.get("game", {})
            display_data = data.get("display", {})
            input_data = data.get("input", {})
            training_data = data.get("training", {})
            integrations_data = data.get("integrations", {})

            return GameProfile(
                id=game_data.get("id", "unknown"),
                name=game_data.get("name", "Unknown Game"),
                publisher=game_data.get("publisher", "Unknown"),
                typical_resolution=display_data.get("typical_resolution", [1920, 1080]),
                ui_scale_range=display_data.get("ui_scale_range", [0.8, 1.2]),
                important_regions=display_data.get("important_regions", {}),
                action_space=input_data.get("action_space", "discrete"),
                num_actions=input_data.get("num_actions", 12),
                requires_mouse=input_data.get("requires_mouse", True),
                mouse_precision=input_data.get("mouse_precision", "medium"),
                default_bindings=input_data.get("default_bindings", []),
                recommended_architecture=training_data.get(
                    "recommended_architecture", "efficientnet_lstm"
                ),
                recommended_input_size=training_data.get(
                    "recommended_input_size", [224, 224]
                ),
                temporal_frames=training_data.get("temporal_frames", 4),
                minimum_samples=training_data.get("minimum_samples", 5000),
                class_balance_tolerance=training_data.get(
                    "class_balance_tolerance", 0.3
                ),
                learning_rate=training_data.get(
                    "learning_rate",
                    {"initial": 0.001, "decay_factor": 0.1, "decay_epochs": [20, 40]},
                ),
                hardware_tiers=hardware_tiers,
                tasks=tasks,
                external_overlays=integrations_data.get("external_overlays", {}),
                profile_path=profile_path,
            )

        except KeyError as e:
            raise ValueError(f"Invalid profile format, missing key: {e}")

    def get_template_path(self) -> Path:
        """Get path to the template profile for creating custom games."""
        return self.profiles_dir / "_template" / "profile.yaml"

    def create_custom_profile(self, game_id: str, display_name: str) -> Path:
        """
        Create a new custom game profile from template.

        Args:
            game_id: Unique identifier (lowercase, underscores)
            display_name: Human-readable game name

        Returns:
            Path to the new profile directory
        """
        import shutil

        # Validate game_id
        if not game_id.replace("_", "").isalnum():
            raise ValueError("game_id must be alphanumeric with underscores only")

        new_profile_dir = self.profiles_dir / game_id
        if new_profile_dir.exists():
            raise ValueError(f"Profile already exists: {game_id}")

        # Copy template
        template_dir = self.profiles_dir / "_template"
        shutil.copytree(template_dir, new_profile_dir)

        # Update the profile.yaml with new game info
        profile_path = new_profile_dir / "profile.yaml"
        with open(profile_path) as f:
            content = f.read()

        content = content.replace("my_custom_game", game_id)
        content = content.replace("My Custom Game", display_name)

        with open(profile_path, "w") as f:
            f.write(content)

        # Update index
        self._add_to_index(game_id, display_name)

        return new_profile_dir

    def _add_to_index(self, game_id: str, display_name: str):
        """Add a new game to the index."""
        index = self._load_index()

        if "custom_profiles" not in index:
            index["custom_profiles"] = []

        index["custom_profiles"].append(
            {"id": game_id, "name": display_name, "path": f"{game_id}/profile.yaml"}
        )

        index_path = self.profiles_dir / "index.yaml"
        with open(index_path, "w") as f:
            yaml.dump(index, f, default_flow_style=False)

        # Clear cache
        self._index = None
