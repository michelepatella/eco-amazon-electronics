import glob
import json
import os
from pathlib import Path

import yaml

from src.const import CONFIG_DIR


def load_config() -> dict:
    cfg = {}
    for f in sorted(Path(CONFIG_DIR).glob("*.yaml")):
        with open(f, encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
            stack = [(cfg, data)]
            while stack:
                base, update = stack.pop()
                for k, v in update.items():
                    if (
                        k in base
                        and isinstance(base[k], dict)
                        and isinstance(v, dict)
                    ):
                        stack.append((base[k], v))
                    else:
                        base[k] = v
    return cfg


def get_latest_checkpoint(model_name):
    """Get the latest model checkpoint from its folder."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    saved_model_dir = os.path.join(
        base_path,
        "2_recbole",
        "models",
        model_name,
    )
    checkpoint_files = glob.glob(os.path.join(saved_model_dir, "*.pth"))
    latest_file = sorted(checkpoint_files)[-1]
    return latest_file


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
