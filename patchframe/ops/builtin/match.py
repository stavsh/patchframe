"""patchframe.ops.builtin.match"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from patchframe.data.predicates import MatchPredicate, Stage, equals
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import DimensionedSliceField
from patchframe.ops.builtin.dimension_join import dimension_join

#: Chain ordering by stage: cheap, selective partitioners first so the
#: candidate set starts small (the fixed heuristic, not a cost model).
_STAGE_ORDER = {Stage.PARTITIONER: 0, Stage.BLOCK_PAIRER: 1, Stage.PAIR_FILTER: 2}


def match(
    left: Dataset,
    right: Dataset,
    *,
    on: str | Iterable[Any] = (),
    predicates: Mapping[str, MatchPredicate] | None = None,
) -> Dataset:
    """Build a correspondence plan by matching dimensional terms.

    Sugar over a ``dimension_join`` chain (the user's compositional path made
    declarative; ``dimension-join-execution.md``): each ``on`` key becomes a
    categorical ``equals`` partitioner, each ``predicates[dimension]`` becomes
    its predicate. Terms run in **stage order** — partitioners first, so the
    candidate set starts small — threaded through ``candidates``. Returns the
    left/right correspondence plan (``pair_id`` index, ``left_index``/
    ``right_index`` ``ForeignIndexField``s), the shape ``explode``/``merge``/
    ``partition`` consume downstream.

    ``on`` names scalar key fields: ``"clip"`` (same name both sides),
    ``["clip", "lang"]`` (several), or ``("left_key", "right_key")`` pairs.
    Interval predicates resolve their operand on each side to the unique
    ``DimensionedSliceField`` whose slices carry that dimension (compose the
    interval with ``compose_slice`` first). Resolution is name-based for v1;
    deeper ``comparable_with`` validation waits on the dimensional-slice
    consolidation (slices carrying an axis key).

    v1 is eager and carries the correspondence as expanded pairs. ``match``
    coexists with the strategy-based ``join``; reconciling the two is a
    deliberate later step.
    """

    on_terms = _normalize_on(on)
    predicate_terms = sorted(
        (predicates or {}).items(), key=lambda item: _STAGE_ORDER[item[1].stage]
    )
    if not on_terms and not predicate_terms:
        raise ValueError("match requires at least one `on` key or predicate term.")

    correspondence: Dataset | None = None
    for left_field, right_field in on_terms:
        correspondence = dimension_join(
            left,
            right,
            correspondence,
            predicate=equals(),
            left_on=left_field,
            right_on=right_field,
        )
    for dimension, predicate in predicate_terms:
        if not isinstance(predicate, MatchPredicate):
            raise TypeError(
                f"match: predicates[{dimension!r}] must be a MatchPredicate, got "
                f"{type(predicate).__name__}."
            )
        left_field = _resolve_slice_field(left, dimension, "left")
        right_field = _resolve_slice_field(right, dimension, "right")
        correspondence = dimension_join(
            left,
            right,
            correspondence,
            predicate=predicate,
            left_on=left_field,
            right_on=right_field,
            dimension=dimension,
        )
    return correspondence


def _normalize_on(on: str | Iterable[Any]) -> list[tuple[str, str]]:
    if not on:
        return []
    if isinstance(on, str):
        return [(on, on)]
    terms: list[tuple[str, str]] = []
    for key in on:
        if isinstance(key, str):
            terms.append((key, key))
        elif isinstance(key, (tuple, list)) and len(key) == 2:
            terms.append((str(key[0]), str(key[1])))
        else:
            raise TypeError(f"match: invalid `on` key {key!r}.")
    return terms


def _resolve_slice_field(dataset: Dataset, dimension: str, side: str) -> str:
    """The unique DimensionedSliceField on this side whose slices carry ``dimension``."""

    matches: list[str] = []
    for field in dataset.schema:
        if not isinstance(field, DimensionedSliceField):
            continue
        for value in dataset.table[field.name]:
            if value is not None and hasattr(value, "dims") and dimension in value.dims:
                matches.append(field.name)
                break
    if len(matches) != 1:
        raise ValueError(
            f"match: cannot resolve dimension {dimension!r} to a unique slice field "
            f"on the {side} side (found {matches}). Compose the interval with "
            "compose_slice first, or use dimension_join with explicit fields."
        )
    return matches[0]
