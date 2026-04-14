"""
patchframe.dataset.state

Primary dataset state container for patchframe.

``DatasetState`` groups the four top-level pieces of dataset state:

- schema
- table
- binding specs
- provenance

It may also carry shared side tables used by tiny ``DataAccessor`` objects,
such as source descriptors, asset dictionaries, and compiled view tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from patchframe.data.descriptor import SourceDescriptor
from patchframe.dataset.bindings import BindingSpecSet
from patchframe.dataset.provenance import DatasetProvenance
from patchframe.dataset.schema import Schema


@dataclass(frozen=True, slots=True)
class DatasetState:
    """Complete in-memory dataset state."""

    schema: Schema
    table: pd.DataFrame
    bindings: BindingSpecSet = field(default_factory=BindingSpecSet)
    provenance: DatasetProvenance = field(default_factory=DatasetProvenance)
    source_descriptors: Mapping[int, SourceDescriptor] = field(default_factory=dict)
    assets: Mapping[int, str] = field(default_factory=dict)
    views: Mapping[int, Any] = field(default_factory=dict)