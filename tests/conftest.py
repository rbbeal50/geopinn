"""Keep autograd checks in float64 so finite-difference comparisons are meaningful."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch


@pytest.fixture(autouse=True)
def _float64_default() -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)
