import os
import json
from omegaconf import OmegaConf

def load_config(config_path: str = None):
    """
    Load OmegaConf config.
    If config_path is None, default to project root /config.yaml.
    """
    if config_path is None:
        # project root assumed to be the parent of this file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
        cfg_path = os.path.join(project_root, "config.yaml")
    else:
        cfg_path = config_path

    cfg = OmegaConf.load(cfg_path)
    return cfg, os.path.abspath(os.path.dirname(cfg_path))
