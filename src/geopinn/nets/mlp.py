"""Fully connected networks used as PINN trial functions."""

from __future__ import annotations

import itertools
import math
from typing import Literal, cast

import torch
from torch import Tensor, nn

ActivationName = Literal["tanh", "sin"]


class Sine(nn.Module):
    """Sine nonlinearity phi(z) = sin(omega z).

    With ``omega = 1`` this is a plain sine unit. Larger ``omega`` is the
    SIREN frequency scale (Sitzmann et al., 2020).
    """

    def __init__(self, omega: float = 1.0) -> None:
        super().__init__()
        if omega <= 0.0:
            raise ValueError("omega must be positive")
        self.omega = float(omega)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega * x)


class FourierFeatures(nn.Module):
    """Random Fourier feature map gamma(v) = [sin(2 pi B v), cos(2 pi B v)].

    ``B`` has i.i.d. entries drawn from N(0, sigma^2) and is stored as a
    buffer so it is fixed for a run and restored from checkpoints.
    """

    def __init__(
        self,
        in_features: int,
        n_features: int,
        sigma: float,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        raw = torch.randn(in_features, n_features, generator=generator) * sigma
        self.register_buffer("B", raw)

    def forward(self, coords: Tensor) -> Tensor:
        basis = self.B
        if not isinstance(basis, Tensor):
            raise TypeError("Fourier basis buffer is missing")
        projected = (2.0 * math.pi) * (coords @ basis)
        return torch.cat((torch.sin(projected), torch.cos(projected)), dim=1)


def _fill_uniform(
    tensor: Tensor,
    low: float,
    high: float,
    generator: torch.Generator,
) -> None:
    sample = torch.rand(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        generator=generator,
    )
    with torch.no_grad():
        tensor.copy_(low + (high - low) * sample)


def _init_linear(
    layer: nn.Linear,
    *,
    activation: ActivationName,
    sine_omega: float,
    first: bool,
    generator: torch.Generator,
) -> None:
    weight = layer.weight
    fan_in = int(weight.shape[1])
    fan_out = int(weight.shape[0])
    if activation == "tanh":
        # Xavier uniform with the tanh gain, so hidden pre-activations start O(1).
        gain = float(nn.init.calculate_gain("tanh"))
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        bound = math.sqrt(3.0) * std
        _fill_uniform(weight, -bound, bound, generator)
    else:
        # SIREN initialisation. The first layer sees the coordinates; later
        # layers are scaled by 1/omega so sin(omega W x) is not saturated.
        if first:
            bound = 1.0 / fan_in
        else:
            bound = math.sqrt(6.0 / fan_in) / sine_omega
        _fill_uniform(weight, -bound, bound, generator)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """Coordinate MLP with tanh or sine activations.

    Inputs are the raw collocation coordinates. When Fourier features are
    off and ``map_unit_interval`` is set, each coordinate in ``[0, 1]`` is
    affinely mapped to ``[-1, 1]`` before the first linear layer, which keeps
    tanh units in their sensitive range. The last layer is linear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: int,
        hidden_features: int,
        *,
        activation: ActivationName = "tanh",
        sine_omega: float = 1.0,
        fourier_features: int = 0,
        fourier_sigma: float = 1.0,
        map_unit_interval: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("feature sizes must be positive")
        if hidden_layers < 1 or hidden_features < 1:
            raise ValueError("hidden_layers and hidden_features must be positive")
        if activation not in ("tanh", "sin"):
            raise ValueError("activation must be 'tanh' or 'sin'")
        if fourier_features < 0:
            raise ValueError("fourier_features must be non-negative")

        self.in_features = in_features
        self.out_features = out_features
        self.activation_name: ActivationName = activation
        self.map_unit_interval = map_unit_interval and fourier_features == 0

        linear_generator = torch.Generator()
        linear_generator.manual_seed(seed)
        fourier_generator = torch.Generator()
        fourier_generator.manual_seed(seed + 1)

        self.fourier: FourierFeatures | None
        mlp_in = in_features
        if fourier_features > 0:
            self.fourier = FourierFeatures(
                in_features,
                fourier_features,
                fourier_sigma,
                fourier_generator,
            )
            mlp_in = 2 * fourier_features
        else:
            self.fourier = None

        blocks: list[nn.Module] = []
        widths = [mlp_in] + [hidden_features] * hidden_layers + [out_features]
        for index, (fan_in, fan_out) in enumerate(itertools.pairwise(widths)):
            linear = nn.Linear(fan_in, fan_out)
            is_last = index == len(widths) - 2
            # The readout is linear. Initialise it like the hidden maps so the
            # initial field is the same order as the hidden features.
            _init_linear(
                linear,
                activation=activation,
                sine_omega=sine_omega,
                first=index == 0,
                generator=linear_generator,
            )
            blocks.append(linear)
            if not is_last:
                blocks.append(nn.Tanh() if activation == "tanh" else Sine(sine_omega))
        self.net = nn.Sequential(*blocks)

    def forward(self, coords: Tensor) -> Tensor:
        if coords.ndim != 2 or coords.shape[1] != self.in_features:
            raise ValueError(
                f"expected coordinates of shape (N, {self.in_features}), got {tuple(coords.shape)}"
            )
        features = coords
        if self.fourier is not None:
            features = self.fourier(coords)
        elif self.map_unit_interval:
            features = 2.0 * coords - 1.0
        # nn.Sequential.__call__ is typed as returning Any in the installed torch stubs.
        return cast(Tensor, self.net(features))
