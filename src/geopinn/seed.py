"""Seed Python, NumPy, and PyTorch from one integer."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed the RNGs used by sampling fallbacks and framework code.

    Network weights and quasi-random collocation use their own generators,
    also derived from this seed, so a run does not depend on call order.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
