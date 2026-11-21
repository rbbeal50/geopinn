"""Reverse-mode derivatives for strong-form residuals."""

from __future__ import annotations

import torch
from torch import Tensor


def partial(output: Tensor, variable: Tensor) -> Tensor:
    """First partial of ``output`` with respect to ``variable``.

    The graph is retained so a later backward pass can differentiate the
    residual with respect to the network weights. Independent collocation
    rows are handled by the vector-Jacobian product with ``grad_outputs = 1``,
    which yields ``∂u_i / ∂variable_i`` when each sample depends only on its
    own coordinates.
    """
    if not variable.requires_grad:
        raise RuntimeError("partial() requires a tensor with requires_grad=True")
    grads = torch.autograd.grad(
        output,
        variable,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )
    gradient = grads[0]
    if not isinstance(gradient, Tensor):
        raise RuntimeError("autograd did not return a derivative")
    return gradient


def second_partial(output: Tensor, variable: Tensor) -> Tensor:
    """Pure second partial ``∂² output / ∂variable²``.

    ``variable`` is a single column of shape ``(N, 1)`` (time, or one Cartesian
    coordinate). For the sum over spatial coordinates use :func:`laplacian`.
    """
    return partial(partial(output, variable), variable)


def laplacian(output: Tensor, x: Tensor) -> Tensor:
    """Cartesian Laplacian ``∇² output`` for coordinates ``x`` of shape ``(N, d)``.

    Each pure second derivative ``∂²u/∂x_i²`` is obtained by reverse-mode
    autodiff and the Laplacian is their sum. This is the spatial operator in
    the acoustic equation ``u_tt - c² ∇²u = s``, and it is valid for any
    spatial dimension. The returned tensor has shape ``(N, 1)``.
    """
    if x.ndim != 2:
        raise ValueError(f"x must have shape (N, d), got {tuple(x.shape)}")
    first = partial(output, x)
    accumulated = torch.zeros(
        output.shape[0],
        1,
        dtype=output.dtype,
        device=output.device,
    )
    for axis in range(x.shape[1]):
        component = first[:, axis : axis + 1]
        second = partial(component, x)
        accumulated = accumulated + second[:, axis : axis + 1]
    return accumulated
