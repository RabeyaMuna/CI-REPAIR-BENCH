import os
from pathlib import Path

from omegaconf import OmegaConf

# Project root = three levels up from this file:
# baselines/utilities/load_config.py -> baselines/utilities -> baselines -> CI-REPAIR-BENCH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: str | Path | None = None):
    """
    Load the Hydra/OmegaConf config for CI-REPAIR-BENCH.

    - By default, loads `<PROJECT_ROOT>/config.yaml`.
    - Optionally accepts an explicit config_path override.
    """
    if config_path is None:
        path = DEFAULT_CONFIG_PATH
    else:
        path = Path(config_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")

    return OmegaConf.load(str(path))
