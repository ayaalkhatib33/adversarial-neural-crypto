"""
utils.py

Small helper functions:
- seeding
- device selection
- directory creation
- saving checkpoints and JSON logs
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set all relevant random seeds for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Return 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    """
    Create a directory if it does not already exist.
    """
    if path == "":
        return
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """
    Save a PyTorch checkpoint.

    Args:
        state: dict with arbitrary contents (e.g. model + optimizer state).
        path:  file path for the checkpoint.
    """
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def save_json(data: Dict[str, Any], path: str) -> None:
    """
    Save a Python dict as a nicely formatted JSON file.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
