"""Compatibility entry point for the shared PyTorch offline evaluator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bot_mmorpg.scripts.evaluate_model import main
if __name__ == "__main__":
    raise SystemExit(main())
