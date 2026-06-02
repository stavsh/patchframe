"""
patchframe.ops.dispatch

Unified aspect dispatch for the operator framework.

``_apply`` (``DatasetOperator``) and ``_compose`` (``CompositionOperator``)
are thin family-specific wrappers around ``compute_output_state``. Per-aspect
work is dispatched through a registry of small handler functions keyed on
``(aspect, concrete_mode)``. Extensions register their own handlers via
``register_aspect_handler``.

Schema is the one aspect that cannot go through the registry as-is: it
must be built first so ``resolve_derived_transitions`` has an output
schema to walk. ``_build_schema`` handles the ``preserve`` short-circuit
and falls through to ``operator.apply_schema`` for the structural modes.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

from patchframe.dataset.couplings import Coupling, CouplingSet
from patchframe.dataset.field_composition import resolve_merged_fields
from patchframe.dataset.identity import (
    mint_primary_index_identity,
    primary_index_identity,
    with_primary_index_identity,
)
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.resolution import resolve_derived_transitions
from patchframe.ops.transitions import TransitionPlan

Handler = Callable[
    [
        tuple[DatasetState, ...],   # states
        TransitionPlan,             # resolved
        Schema,                     # output_schema so far
        Any,                        # operator
        tuple,                      # args
        dict,                       # kwargs
    ],
    Any,
]


# ---------------------------------------------------------------------------
# Schema build (special: runs before resolve_derived_transitions)
# ---------------------------------------------------------------------------


def _build_schema(operator, declared, states, args, kwargs) -> Schema:
    """Produce the output schema. Short-circuits on ``preserve``."""
    if declared.schema.mode == "preserve":
        return states[declared.schema.input].schema
    return operator.apply_schema(*states, *args, **kwargs)


# ---------------------------------------------------------------------------
# Index identity handlers
# ---------------------------------------------------------------------------


def _identity_inherit(states, resolved, output_schema, operator, args, kwargs):
    input_idx = resolved.index_identity.input
    if isinstance(input_idx, str):
        raise ValueError(
            f"{operator.name}: dispatch does not yet support named-input "
            f"index identity inherit (input={input_idx!r}). Operators with "
            f"named-input dispatch must override __call__."
        )
    source_state = states[input_idx]
    try:
        return with_primary_index_identity(
            output_schema, primary_index_identity(source_state)
        )
    except ValueError:
        return output_schema


def _identity_mint(states, resolved, output_schema, operator, args, kwargs):
    return mint_primary_index_identity(output_schema)


def _identity_coalesce(states, resolved, output_schema, operator, args, kwargs):
    identities = set()
    for state in states:
        try:
            identities.add(primary_index_identity(state))
        except ValueError:
            continue
    if len(identities) == 1:
        try:
            return with_primary_index_identity(output_schema, next(iter(identities)))
        except ValueError:
            return output_schema
    return mint_primary_index_identity(output_schema)


def _identity_custom(states, resolved, output_schema, operator, args, kwargs):
    return operator.apply_index_identity(
        states[0], output_schema, resolved.index_identity, *args, **kwargs
    )


# ---------------------------------------------------------------------------
# Table handlers
# ---------------------------------------------------------------------------


def _table_preserve(states, resolved, output_schema, operator, args, kwargs):
    return states[resolved.table.input].table


def _table_call_operator(states, resolved, output_schema, operator, args, kwargs):
    if len(states) == 1:
        return operator.apply_table(states[0], *args, **kwargs)
    return operator.apply_table(*states, *args, composed_schema=output_schema, **kwargs)


# ---------------------------------------------------------------------------
# Couplings handlers
# ---------------------------------------------------------------------------


def _couplings_inherit(states, resolved, output_schema, operator, args, kwargs):
    return states[resolved.couplings.input].couplings


def _couplings_homogeneous(states, resolved, output_schema, operator, args, kwargs):
    if not states:
        return CouplingSet()
    first = states[0].couplings
    for state in states[1:]:
        if state.couplings != first:
            raise ValueError(
                f"{operator.name}: homogeneous couplings require all input "
                f"CouplingSets to be structurally equal. Consume coupled fields "
                f"before composition and reapply couplings afterwards."
            )
    return first


def _couplings_clear(states, resolved, output_schema, operator, args, kwargs):
    return CouplingSet()


def _couplings_derive(states, resolved, output_schema, operator, args, kwargs):
    if resolved.schema.mode == "compose":
        return _apply_compose_derive(states, resolved.couplings, output_schema)
    return _apply_unary_derive(
        states, resolved.couplings, output_schema, operator, args, kwargs
    )


def _couplings_call_operator(states, resolved, output_schema, operator, args, kwargs):
    if len(states) == 1:
        return operator.apply_couplings(states[0], *args, **kwargs)
    return operator.apply_couplings(*states, *args, composed_schema=output_schema, **kwargs)


def _apply_unary_derive(states, transition, output_schema, operator, args, kwargs):
    """Apply rename + drop lineage to input 0's couplings + append new_couplings.

    Reads the pre-computed ``rename_map`` and ``dropped`` info from
    ``resolve_derived_transitions``. Calls the operator's
    ``new_couplings`` hook if defined (DatasetOperator) to append any
    operator-introduced couplings (e.g. the bind_* family).
    """
    couplings = states[0].couplings
    if couplings.couplings:
        if transition.rename_map:
            couplings = couplings.rewrite_field_names(dict(transition.rename_map))
        surviving = set(output_schema.names())
        retained = couplings.retain(surviving)
        dropped_count = len(couplings.couplings) - len(retained.couplings)
        if dropped_count:
            warnings.warn(
                f"{operator.name}: dropped {dropped_count} coupling(s) "
                f"referencing fields that did not survive the schema "
                f"transition.",
                UserWarning,
                stacklevel=4,
            )
        couplings = retained

    new = ()
    new_hook = getattr(operator, "new_couplings", None)
    if new_hook is not None:
        new = new_hook(states[0], *args, **kwargs)
    if new:
        existing = set(couplings.couplings)
        additions = tuple(c for c in new if c not in existing)
        if additions:
            couplings = couplings.add(*additions)
    return couplings


def _apply_compose_derive(states, transition, output_schema):
    """Compose-derive couplings from every input using resolved schema lineage.

    Couplings are name-based, so final output-name retention handles fields
    dropped by composition without incorrectly pruning a same-name collision
    winner. ``superseded_per_input`` performs the input-specific collision
    pruning first; ``rename_map`` then rewrites surviving references.
    """
    superseded_map = dict(transition.superseded_per_input)
    rename_map = dict(transition.rename_map)
    surviving = set(output_schema.names())
    result: list[Coupling] = []
    for input_index, state in enumerate(states):
        lost = frozenset(superseded_map.get(input_index, ()))
        retained: list[Coupling] = []
        for coupling in state.couplings.couplings:
            touched = (coupling.output_field(), *coupling.input_fields())
            if lost and not lost.isdisjoint(touched):
                continue
            retained.append(coupling)
        couplings = CouplingSet(couplings=tuple(retained))
        if rename_map:
            couplings = couplings.rewrite_field_names(rename_map)
        couplings = couplings.retain(surviving)
        for coupling in couplings.couplings:
            if coupling not in result:
                result.append(coupling)
    return CouplingSet(couplings=tuple(result))


# ---------------------------------------------------------------------------
# Sources handlers
# ---------------------------------------------------------------------------


def _sources_inherit(states, resolved, output_schema, operator, args, kwargs):
    return states[resolved.sources.input].sources


def _sources_clear(states, resolved, output_schema, operator, args, kwargs):
    return ()


def _sources_derive(states, resolved, output_schema, operator, args, kwargs):
    """Return the deduped union of every input's sources."""
    seen: dict = {}
    for state in states:
        for src in state.sources:
            seen[src.source_id or src.source_uri] = src
    return tuple(seen.values())


def _sources_call_operator(states, resolved, output_schema, operator, args, kwargs):
    if len(states) == 1:
        apply_sources = getattr(operator, "apply_sources", None)
        if apply_sources is not None:
            return apply_sources(states[0], *args, **kwargs)
        return states[0].sources
    combine_sources = getattr(operator, "combine_sources", None)
    if combine_sources is not None:
        return combine_sources(*states, *args, composed_schema=output_schema, **kwargs)
    return states[0].sources if states else ()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_HANDLERS: dict[tuple[str, str], Handler] = {
    # ``preserve`` is the Phase-1-deprecated alias of ``inherit`` for index
    # identity; operators still using the bare TransitionPlan default carry
    # mode="preserve". Treat it as inherit until Phase 6 redeclares.
    ("index_identity", "preserve"): _identity_inherit,
    ("index_identity", "inherit"):  _identity_inherit,
    ("index_identity", "mint"):     _identity_mint,
    ("index_identity", "coalesce"): _identity_coalesce,
    ("index_identity", "custom"):   _identity_custom,
    ("table", "preserve"):  _table_preserve,
    ("table", "mutate"):    _table_call_operator,
    ("table", "construct"): _table_call_operator,
    ("couplings", "inherit"):     _couplings_inherit,
    ("couplings", "homogeneous"): _couplings_homogeneous,
    ("couplings", "clear"):       _couplings_clear,
    ("couplings", "derive"):      _couplings_derive,
    ("couplings", "construct"):   _couplings_call_operator,
    ("couplings", "custom"):      _couplings_call_operator,
    ("sources", "inherit"):   _sources_inherit,
    ("sources", "clear"):     _sources_clear,
    ("sources", "derive"):    _sources_derive,
    ("sources", "compose"):   _sources_derive,
    ("sources", "construct"): _sources_call_operator,
    ("sources", "custom"):    _sources_call_operator,
}


def register_aspect_handler(aspect: str, mode: str, handler: Handler) -> None:
    """Register or override a per-aspect-per-mode handler.

    Mirrors the ``register_field_policy`` extension pattern. Extensions
    use this to customize handling of a specific ``(aspect, mode)``
    combination, typically for custom field types or custom modes.
    """
    _HANDLERS[(aspect, mode)] = handler


def _dispatch(
    aspect: str,
    states: tuple[DatasetState, ...],
    resolved: TransitionPlan,
    output_schema: Schema,
    operator,
    args: tuple,
    kwargs: dict,
):
    transition = getattr(resolved, aspect)
    key = (aspect, transition.mode)
    try:
        handler = _HANDLERS[key]
    except KeyError as exc:
        raise ValueError(
            f"dispatch: no handler registered for {aspect}={transition.mode!r}."
        ) from exc
    return handler(states, resolved, output_schema, operator, args, kwargs)


# ---------------------------------------------------------------------------
# Unified flow
# ---------------------------------------------------------------------------


def compute_output_state(
    operator,
    states: tuple[DatasetState, ...],
    args: tuple,
    kwargs: dict,
) -> DatasetState:
    """Compute the output ``DatasetState`` for an operator call through dispatch.

    The unified flow that ``_apply`` and ``_compose`` will both call once
    wired in Phases 4 and 5:

    1. Ask the operator for its declared transition plan via
       ``resolve_transitions`` (may still contain ``derive`` modes).
    2. Build the output schema via ``apply_schema`` or the preserve
       short-circuit.
    3. Collapse the declared plan to a resolved plan via
       ``resolve_derived_transitions`` (schemas-only).
    4. Dispatch each aspect (identity, table, couplings, sources)
       through the registry.
    5. Collapse any ``MergedField`` in the schema (compose finalization).
    6. Return a fresh ``DatasetState``.

    Callers extract the four aspect values from the returned state and
    merge with the input dataset's metadata / side maps via
    ``replace_state`` if those need to be preserved.
    """
    declared = operator.resolve_transitions(*states, *args, **kwargs)
    output_schema = _build_schema(operator, declared, states, args, kwargs)
    input_schemas = tuple(state.schema for state in states)
    resolved = resolve_derived_transitions(
        declared,
        input_schemas=input_schemas,
        output_schema=output_schema,
    )
    output_schema = _dispatch(
        "index_identity", states, resolved, output_schema, operator, args, kwargs
    )
    output_table = _dispatch(
        "table", states, resolved, output_schema, operator, args, kwargs
    )
    output_couplings = _dispatch(
        "couplings", states, resolved, output_schema, operator, args, kwargs
    )
    output_sources = _dispatch(
        "sources", states, resolved, output_schema, operator, args, kwargs
    )
    output_schema = resolve_merged_fields(output_schema)
    return DatasetState(
        schema=output_schema,
        table=output_table,
        couplings=output_couplings,
        sources=output_sources,
    )
