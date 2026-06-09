"""Convenience public API for patchframe."""

import warnings

from patchframe.data import *  # noqa: F403
from patchframe.data import __all__ as _data_all
from patchframe.dataset import *  # noqa: F403
from patchframe.dataset import __all__ as _dataset_all
from patchframe.ops import *  # noqa: F403
from patchframe.ops import __all__ as _ops_all

__all__ = [*_data_all, *_dataset_all, *_ops_all]

#: Renamed operators kept as deprecated aliases (the ``bind_`` prefix was
#: redundant once the eager/lazy duality made these ordinary coupling-producers).
_DEPRECATED_OP_ALIASES = {
    "bind_materialize": "materialize",
    "bind_slice": "slice_data",
    "bind_dimensions": "compose_slice",
}


def __getattr__(name: str):
    """Resolve deprecated operator aliases, warning and forwarding to the new name."""

    new_name = _DEPRECATED_OP_ALIASES.get(name)
    if new_name is not None:
        warnings.warn(
            f"patchframe.{name} was renamed to patchframe.{new_name}; the old name "
            "is deprecated and will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
