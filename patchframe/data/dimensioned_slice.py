"""
patchframe.data.dimensioned_slice

Natural-unit slice representation for dimensioned arrays.

DimensionedSlice stores per-axis slice values in the natural units of each
dimension (seconds for TemporalDimension, raw indices for IndexDimension, etc.).
Conversion to array indices is the responsibility of each Dimension subclass
via Dimension.to_index(), called by Dimensions.resolve().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class DimensionedSlice:
    """A slice of a dimensioned array, expressed in each dimension's natural units.

    Parameters
    ----------
    dims:
        Map of dimension name to a slice value in that dimension's natural unit.
        Dimensions absent from this map default to full selection at resolve time.
    metadata:
        Optional slice-level metadata.
    """

    dims: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
