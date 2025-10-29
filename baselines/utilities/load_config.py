from omegaconf import OmegaConf
import os

def load_config():
    # Get the project root (one directory up from this file)
    # PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Correct filename
    # config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    return OmegaConf.load("/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/config.yaml")
