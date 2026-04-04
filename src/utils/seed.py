"""Global seed setter for reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for numpy, stdlib random, and PYTHONHASHSEED."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
