"""
patchframe.storage.array_store

Abstract array store interface for patchframe.

ArrayStore is the persistent-side counterpart to DataSource. It owns the
durable layout of arrays for a logical collection of items and knows its
own dimension layout. All arrays in a store share the same Dimensions.

The write/append interface accepts a DimensionedSlice extent so each backend
can record where in the shared dimensional space a given array sits. How that
extent is stored internally is each concrete store's concern.

ArrayStore.describe() produces the SourceDescriptor (with dimensions in
capabilities) that DataSource.open() consumes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from patchframe.data.descriptor import SourceDescriptor
from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.data.dimensions import Dimensions


class ArrayStore(ABC):
    """Persistent storage for lazy multidimensional arrays.

    All arrays in a store share the same Dimensions. Each stored array is
    identified by (item_id, asset_name). The extent parameter on write/append
    records the array's full coverage in the shared dimensional space.
    """

    dimensions: Dimensions

    @abstractmethod
    def write(self, item_id: Any, asset_name: str, array: np.ndarray, extent: DimensionedSlice) -> None:
        """Write or overwrite an array and its extent for the given item and asset."""
        ...

    @abstractmethod
    def append(self, item_id: Any, asset_name: str, array: np.ndarray, extent: DimensionedSlice) -> None:
        """Add a new array. Raises if (item_id, asset_name) already exists."""
        ...

    @abstractmethod
    def describe(self) -> SourceDescriptor:
        """Return a SourceDescriptor that DataSource.open() can consume.

        Must populate capabilities with at least ``"dimensions"`` and
        ``"asset_names"``, and open_config with whatever is needed to
        reopen this store in the same process (or deserialise from disk).
        """
        ...
