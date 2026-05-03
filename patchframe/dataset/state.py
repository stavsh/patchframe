"""
patchframe.dataset.state

Primary dataset state container for patchframe.

DatasetState groups the core dataset aspects:

- schema
- table
- couplings
- sources

It also carries shared side tables used by DataAccessor objects.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from patchframe.data.descriptor import SourceDescriptor
from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.provenance import DatasetSourceInfo
from patchframe.dataset.schema import Schema


@dataclass(frozen=True, slots=True)
class DatasetState:
    """Complete in-memory dataset state."""

    schema: Schema
    table: pd.DataFrame
    couplings: CouplingSet = field(default_factory=CouplingSet)
    sources: tuple[DatasetSourceInfo, ...] = field(default_factory=tuple)
    source_descriptors: Mapping[int, SourceDescriptor] = field(default_factory=dict)
    assets: Mapping[int, str] = field(default_factory=dict)
    views: Mapping[int, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", deepcopy(self.schema))
        object.__setattr__(self, "couplings", deepcopy(self.couplings))
