"""
patchframe.storage.dataset_store

PROVISIONAL — the dataset-store interface is **subject to change** while the IO
taxonomy is unsettled (docs/design/roadmap.md "IO and storage"; the
source-vs-storage split of design-constraints.md §5).

``DatasetStore`` is the minimal write side that ``offload`` targets: register a
dataset's content as a ``DatasetSource`` and return its ``source_desc_id``, and
expose the ``SourceManager`` it registered into (so produced ``DatasetAccessor``s
can be stamped with a ``manager_hint``). The pandas-memory backend
(``MemoryDatasetStore``/``MemoryDatasetSource``) is the only implementation today;
a disk / ``MetadataStore`` backend slots in behind the same ``put``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from patchframe.data.manager import SourceManager
    from patchframe.dataset.dataset import Dataset


@runtime_checkable
class DatasetStore(Protocol):
    """The write side of a dataset store (PROVISIONAL, subject to change)."""

    manager: "SourceManager"

    def put(self, ds: "Dataset") -> int:
        """Persist ``ds`` into the store and return the served ``source_desc_id``."""
        ...
