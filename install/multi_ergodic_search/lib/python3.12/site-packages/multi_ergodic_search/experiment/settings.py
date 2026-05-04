from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "datas" / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    loaded = yaml.safe_load(f)

config = loaded["opt_args"]
