"""
Run grid search using Hydra multirun.
"""

import subprocess
import sys
from pathlib import Path


def run_grid_search():
    """Run grid search using Hydra."""
    # Change to the scripts directory
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "configs" / "sweep" / "grid_search.yaml"

    # Build command
    cmd = [
        sys.executable, "train_with_config.py",
        "--config-path", "../configs/sweep",
        "--config-name", "grid_search",
        "--multirun"
    ]

    print("Running grid search with command:")
    print(" ".join(cmd))
    print("\nThis will run multiple experiments in parallel...")

    # Run the command
    try:
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        print("\nGrid search completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Grid search failed with return code {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    run_grid_search()