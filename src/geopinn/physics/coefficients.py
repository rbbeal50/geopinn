"""PDE coefficients as constants, tensors, or callables.

A later inversion should be able to replace the acoustic speed with a
trainable field ``c(x)`` or with a rock-physics map (porosity and saturation
in, wavespeed out) without editing the residual. Both are callables or
tensors. This module only evaluates them. It does not build those models.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import torch
from torch import Tensor, nn

from geopinn.config import WaveSpeedSpec
from geopinn.physics.speeds import layered_speed_hat

# Float, a (possibly trainable) tensor, or a map of the collocation points.
# One-argument callables receive x. Two-argument callables receive (x, t).
CoeffLike = float | Tensor | Callable[..., Tensor]


class CoefficientNotBuilt(RuntimeError):
    """The YAML asks for a coefficient this trainer does not instantiate.

    ``form: field`` is the schema slot for a trainable ``c(x)`` or a
    rock-physics map. Construct :class:`geopinn.physics.acoustic.AcousticWave`
    with a tensor or a callable directly until that builder exists.
    """


def resolve_wave_speed(
    spec: WaveSpeedSpec,
    c_hat: float,
    *,
    length: float = 1.0,
    time: float = 1.0,
) -> CoeffLike:
    """Build the speed the residual evaluates, or refuse a field spec.

    ``constant`` returns the nondimensional float ``c_hat = c T / L``. The
    returned value stays a Python float so the constant-coefficient residual
    is a plain multiply, which is the contraction the 1D benchmark was
    verified with.

    ``layered`` returns a callable ``c_hat(x, z)`` of the nondimensional
    coordinates. ``length`` and ``time`` are the reference scales ``Lx`` and
    ``T``. ``field`` is the slot for a trainable ``c(x, z)`` or a rock-physics
    map and is not built here.
    """
    if spec.form == "constant":
        if c_hat <= 0.0:
            raise ValueError("nondimensional wave speed must be positive")
        return c_hat
    if spec.form == "layered":
        if spec.c_top is None or spec.c_bottom is None or spec.interface is None:
            raise CoefficientNotBuilt("layered wave speed is missing its layer parameters")
        if spec.transition is None:
            raise CoefficientNotBuilt("layered wave speed is missing its transition width")
        return layered_speed_hat(
            c_top=spec.c_top,
            c_bottom=spec.c_bottom,
            interface=spec.interface,
            transition=spec.transition,
            length=length,
            time=time,
        )
    source = spec.source or "<unnamed>"
    raise CoefficientNotBuilt(
        "coefficients.wave_speed.form is 'field' "
        f"(source {source!r}). A trainable c(x, z) or a rock-physics map is not "
        "built by this trainer yet. Pass a tensor or a callable as "
        "AcousticWave(wave_speed=...) ; the residual already evaluates it."
    )


def evaluate_coefficient(coeff: CoeffLike, x: Tensor, t: Tensor) -> Tensor:
    """Evaluate ``coeff`` on a batch. Result shape is ``(N, 1)``.

    A float is broadcast as a constant. A tensor keeps its graph, so a
    parameter or a network output can be trained. A callable is called with
    ``(x, t)`` when it takes two arguments and with ``x`` when it takes one,
    which is the shape of a ``c(x)`` network or a Gassmann-type map.
    """
    if isinstance(coeff, bool) or isinstance(coeff, int):
        # bool is a subclass of int; reject it. Plain ints are constants.
        if isinstance(coeff, bool):
            raise TypeError("a coefficient cannot be a bool")
        return _constant(float(coeff), x)
    if isinstance(coeff, float):
        return _constant(coeff, x)
    if isinstance(coeff, Tensor):
        return _as_column(coeff, x)
    if not callable(coeff):
        raise TypeError(f"coefficient must be a float, Tensor, or callable, got {type(coeff)!r}")
    return _as_column(_call(coeff, x, t), x)


def _constant(value: float, x: Tensor) -> Tensor:
    return torch.full((x.shape[0], 1), value, dtype=x.dtype, device=x.device)


def _call(coeff: Callable[..., Tensor], x: Tensor, t: Tensor) -> Tensor:
    arity = _arity(coeff)
    if arity == 1:
        return coeff(x)
    if arity == 2:
        return coeff(x, t)
    raise TypeError(
        "a coefficient callable must take the spatial coordinates or (x, t); "
        f"this one takes {arity} positional arguments"
    )


def _arity(coeff: Callable[..., Tensor]) -> int:
    target: Callable[..., object]
    if isinstance(coeff, nn.Module):
        target = coeff.forward
    else:
        target = coeff
    signature = inspect.signature(target)
    count = 0
    for param in signature.parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            count += 1
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            return 2
    return count


def _as_column(value: Tensor, x: Tensor) -> Tensor:
    """Broadcast ``value`` to ``(N, 1)`` without detaching it."""
    rows = x.shape[0]
    if value.ndim == 0:
        scalar = value.to(dtype=x.dtype, device=x.device).reshape(1, 1)
        return scalar.expand(rows, 1)
    if value.ndim == 1:
        if value.shape[0] not in (1, rows):
            raise ValueError(
                f"coefficient length {value.shape[0]} does not match the batch of {rows}"
            )
        column = value.reshape(-1, 1).to(dtype=x.dtype, device=x.device)
        if column.shape[0] == 1:
            return column.expand(rows, 1)
        return column
    if value.ndim == 2 and value.shape[1] == 1 and value.shape[0] in (1, rows):
        column = value.to(dtype=x.dtype, device=x.device)
        if column.shape[0] == 1:
            return column.expand(rows, 1)
        return column
    raise ValueError(f"coefficient shape {tuple(value.shape)} is not broadcastable to {(rows, 1)}")
