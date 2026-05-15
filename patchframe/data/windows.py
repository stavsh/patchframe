"""Window specifications for dimensional slice expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class AxisWindow:
    """Regular per-axis window used to expand bounded slices into tiles.

    ``size`` and ``stride`` are expressed in the natural unit of the matching
    dimension. ``stride`` defaults to ``size`` for non-overlapping windows.
    """

    size: Any
    stride: Any = None
    offset: Any = 0
    include_partial: bool = False

    def __post_init__(self) -> None:
        stride = self.size if self.stride is None else self.stride
        if self.size <= 0:
            raise ValueError("AxisWindow size must be positive.")
        if stride <= 0:
            raise ValueError("AxisWindow stride must be positive.")
        object.__setattr__(self, "stride", stride)

    def counts(self, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
        """Return the number of generated windows for each extent."""

        first = starts + self.offset
        full_counts = _positive_floor_counts(stops - first - self.size, self.stride)
        if not self.include_partial:
            return full_counts

        next_start = first + full_counts * self.stride
        partial = next_start < stops
        return full_counts + partial.astype(np.int64)

    def intervals(
        self,
        starts: np.ndarray,
        stops: np.ndarray,
        counts: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return tiled start/stop vectors for repeated input extents."""

        counts = self.counts(starts, stops) if counts is None else counts
        total = int(counts.sum())
        if total == 0:
            return (
                np.asarray([], dtype=starts.dtype),
                np.asarray([], dtype=stops.dtype),
            )

        row_positions = np.repeat(np.arange(len(starts)), counts)
        block_offsets = np.repeat(np.cumsum(counts) - counts, counts)
        offsets = np.arange(total) - block_offsets
        tiled_starts = starts[row_positions] + self.offset + offsets * self.stride
        tiled_stops = tiled_starts + self.size
        if self.include_partial:
            tiled_stops = np.minimum(tiled_stops, stops[row_positions])
        return tiled_starts, tiled_stops


def _positive_floor_counts(values: np.ndarray, stride: Any) -> np.ndarray:
    counts = np.floor_divide(values, stride) + 1
    counts = np.where(values >= 0, counts, 0)
    return np.asarray(counts, dtype=np.int64)
