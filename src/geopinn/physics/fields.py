"""Named network outputs at one collocation set.

The acoustic problem reads a single scalar ``u``. A later inversion can add
fields (porosity, saturation, or a vector displacement) without changing the
loss, by giving each column a name. Vector-valued quantities are stored as
one tensor of shape ``(N, components)`` rather than being split into columns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import Tensor


class Fields:
    """A mapping from field name to a tensor with a shared batch axis."""

    def __init__(self, values: Mapping[str, Tensor]) -> None:
        if not values:
            raise ValueError("Fields needs at least one named output")
        rows: int | None = None
        stored: dict[str, Tensor] = {}
        for name, value in values.items():
            if value.ndim != 2:
                shape = tuple(value.shape)
                raise ValueError(f"field {name!r} must have shape (N, components), got {shape}")
            if rows is None:
                rows = value.shape[0]
            elif value.shape[0] != rows:
                raise ValueError(f"field {name!r} has {value.shape[0]} rows, expected {rows}")
            stored[name] = value
        self._values = stored

    def __getitem__(self, name: str) -> Tensor:
        try:
            return self._values[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._values))
            raise KeyError(f"field {name!r} is not in this batch (have {known})") from exc

    def __contains__(self, name: str) -> bool:
        return name in self._values

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._values)

    @classmethod
    def from_columns(cls, output: Tensor, names: Sequence[str]) -> Fields:
        """Split a network output ``(N, F)`` into one scalar field per column.

        ``F`` must equal ``len(names)``. A single column named ``u`` is the
        forward acoustic problem. Extra names are carried along and ignored by
        equations that do not read them.
        """
        if output.ndim != 2:
            raise ValueError(f"network output must have shape (N, F), got {tuple(output.shape)}")
        if output.shape[1] != len(names):
            raise ValueError(
                f"network output has {output.shape[1]} columns but field names are {tuple(names)}"
            )
        return cls({name: output[:, index : index + 1] for index, name in enumerate(names)})


def coerce_fields(output: Tensor | Fields, names: Sequence[str] = ("u",)) -> Fields:
    """Wrap a raw network tensor, or pass a :class:`Fields` batch through."""
    if isinstance(output, Fields):
        return output
    return Fields.from_columns(output, names)
