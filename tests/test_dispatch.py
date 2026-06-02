"""Tests for the unified aspect dispatch (`patchframe.ops.dispatch`).

The dispatch helper is wired into ``_apply`` and ``_compose``. These tests
cover:

- The handler registry has the expected ``(aspect, mode)`` entries.
- Each handler in isolation returns the expected aspect value.
- ``compute_output_state`` against real unary operators matches the
  output that today's ``_apply`` flow produces (parity test, validates
  the rewrite preserves semantics).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd
import pytest

import patchframe as pf
from patchframe.dataset.couplings import Coupling, CouplingSet, Materialize
from patchframe.dataset.field_composition import (
    ColumnCollisionStrategy,
    FieldParent,
    MergedField,
)
from patchframe.dataset.fields import IndexField, ValueField
from patchframe.dataset.identity import primary_index_identity
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.dispatch import (
    _HANDLERS,
    _apply_compose_derive,
    _couplings_call_operator,
    _couplings_clear,
    _couplings_derive,
    _couplings_homogeneous,
    _couplings_inherit,
    _dispatch,
    _identity_coalesce,
    _identity_custom,
    _identity_inherit,
    _identity_mint,
    _sources_call_operator,
    _sources_clear,
    _sources_derive,
    _sources_inherit,
    _table_call_operator,
    _table_preserve,
    compute_output_state,
    register_aspect_handler,
)
from patchframe.ops.transitions import (
    CouplingsTransition,
    IndexIdentityTransition,
    SchemaTransition,
    SourcesTransition,
    TableTransition,
    TransitionPlan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(values=(1, 2, 3), index_name="id"):
    df = pd.DataFrame(
        {"a": values}, index=pd.Index(range(10, 10 + len(values)), name=index_name)
    )
    schema = Schema(
        fields=(
            IndexField(name=index_name, dtype="int64"),
            ValueField(name="a", dtype="int64"),
        )
    )
    return pf.make_from_dataframe(df, schema)


@dataclass(frozen=True, slots=True)
class _NamedCoupling(Coupling):
    """Minimal test-only Coupling that names one input/output field."""

    name: str = ""

    def input_fields(self) -> tuple[str, ...]:
        return (self.name,)

    def output_field(self) -> str:
        return self.name


def _empty_resolved(**overrides):
    """Build a plain resolved TransitionPlan for handler isolation tests."""
    plan = TransitionPlan(
        schema=SchemaTransition.preserve(),
        table=TableTransition.preserve(),
        couplings=CouplingsTransition.derive(),
        sources=SourcesTransition.inherit(),
        index_identity=IndexIdentityTransition.inherit(),
    )
    return plan._with(**overrides) if overrides else plan


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_covers_expected_aspect_mode_pairs():
    expected = {
        ("index_identity", "inherit"),
        ("index_identity", "mint"),
        ("index_identity", "coalesce"),
        ("index_identity", "custom"),
        ("table", "preserve"),
        ("table", "mutate"),
        ("table", "construct"),
        ("couplings", "inherit"),
        ("couplings", "homogeneous"),
        ("couplings", "clear"),
        ("couplings", "derive"),
        ("couplings", "construct"),
        ("couplings", "custom"),
        ("sources", "inherit"),
        ("sources", "clear"),
        ("sources", "derive"),
        ("sources", "compose"),
        ("sources", "construct"),
        ("sources", "custom"),
    }
    assert expected.issubset(_HANDLERS.keys())


def test_register_aspect_handler_overrides_existing_entry():
    sentinel = object()

    def custom_handler(states, resolved, output_schema, operator, args, kwargs):
        return sentinel

    original = _HANDLERS[("couplings", "clear")]
    try:
        register_aspect_handler("couplings", "clear", custom_handler)
        assert _HANDLERS[("couplings", "clear")] is custom_handler
    finally:
        register_aspect_handler("couplings", "clear", original)


def test_dispatch_raises_for_unknown_handler():
    state = _make_dataset().state
    plan = _empty_resolved(
        index_identity=IndexIdentityTransition(mode="inherit"),
        couplings=CouplingsTransition(mode="clear"),
    )
    # Manually corrupt the mode to test the missing-handler branch.
    bogus = TransitionPlan(
        schema=plan.schema,
        table=plan.table,
        couplings=CouplingsTransition(mode="clear"),
        sources=plan.sources,
        index_identity=plan.index_identity,
    )
    # Replace one of the modes with a string the registry doesn't have.
    bogus = bogus._with(couplings=object.__new__(CouplingsTransition))
    object.__setattr__(bogus.couplings, "mode", "bogus_mode")
    object.__setattr__(bogus.couplings, "input", 0)
    object.__setattr__(bogus.couplings, "rename_map", ())
    object.__setattr__(bogus.couplings, "dropped", ())
    object.__setattr__(bogus.couplings, "superseded_per_input", ())
    with pytest.raises(ValueError, match="no handler registered"):
        _dispatch(
            "couplings", (state,), bogus, state.schema, None, (), {}
        )


# ---------------------------------------------------------------------------
# Identity handlers
# ---------------------------------------------------------------------------


def test_identity_inherit_returns_input_identity():
    state = _make_dataset().state
    src_identity = primary_index_identity(state)
    resolved = _empty_resolved()
    result = _identity_inherit((state,), resolved, state.schema, None, (), {})
    # The output schema's primary identity matches the input.
    assert primary_index_identity(
        DatasetState(schema=result, table=state.table)
    ) == src_identity


def test_identity_mint_produces_fresh_identity():
    state = _make_dataset().state
    src_identity = primary_index_identity(state)
    resolved = _empty_resolved(index_identity=IndexIdentityTransition.mint())
    result = _identity_mint((state,), resolved, state.schema, None, (), {})
    minted = primary_index_identity(
        DatasetState(schema=result, table=state.table)
    )
    assert minted != src_identity


def test_identity_coalesce_preserves_when_inputs_agree():
    state = _make_dataset().state
    src_identity = primary_index_identity(state)
    resolved = _empty_resolved(index_identity=IndexIdentityTransition.coalesce())
    result = _identity_coalesce(
        (state, state), resolved, state.schema, None, (), {}
    )
    assert primary_index_identity(
        DatasetState(schema=result, table=state.table)
    ) == src_identity


def test_identity_coalesce_mints_when_inputs_diverge():
    state_a = _make_dataset().state
    state_b = _make_dataset(values=(4, 5)).state  # independent dataset → fresh identity
    resolved = _empty_resolved(index_identity=IndexIdentityTransition.coalesce())
    result = _identity_coalesce(
        (state_a, state_b), resolved, state_a.schema, None, (), {}
    )
    out_identity = primary_index_identity(
        DatasetState(schema=result, table=state_a.table)
    )
    assert out_identity != primary_index_identity(state_a)
    assert out_identity != primary_index_identity(state_b)


def test_identity_inherit_rejects_named_input_in_dispatch():
    state = _make_dataset().state
    resolved = _empty_resolved(
        index_identity=IndexIdentityTransition.inherit(input="plan")
    )

    class _Op:
        name = "_NamedInputOp"

    with pytest.raises(ValueError, match="named-input"):
        _identity_inherit((state,), resolved, state.schema, _Op(), (), {})


def test_identity_custom_calls_operator_hook():
    state = _make_dataset().state
    resolved = _empty_resolved(index_identity=IndexIdentityTransition.custom())

    class _Op:
        def apply_index_identity(self, input_state, schema, transition, *args, **kwargs):
            assert input_state is state
            assert transition is resolved.index_identity
            assert args == ("arg",)
            assert kwargs == {"flag": "value"}
            return schema

    result = _identity_custom(
        (state,), resolved, state.schema, _Op(), ("arg",), {"flag": "value"}
    )
    assert result is state.schema


# ---------------------------------------------------------------------------
# Table handler
# ---------------------------------------------------------------------------


def test_table_preserve_returns_input_table():
    state = _make_dataset().state
    resolved = _empty_resolved(table=TableTransition.preserve())
    result = _table_preserve((state,), resolved, state.schema, None, (), {})
    assert result is state.table


def test_table_call_operator_forwards_composition_args():
    state = _make_dataset().state
    resolved = _empty_resolved(table=TableTransition.construct())

    class _Op:
        def apply_table(self, *args, composed_schema=None, **kwargs):
            assert args == (state, state, "arg")
            assert composed_schema is state.schema
            assert kwargs == {"flag": "value"}
            return state.table

    result = _table_call_operator(
        (state, state), resolved, state.schema, _Op(), ("arg",), {"flag": "value"}
    )
    assert result is state.table


# ---------------------------------------------------------------------------
# Couplings handlers
# ---------------------------------------------------------------------------


def test_couplings_clear_returns_empty():
    state = _make_dataset().state
    resolved = _empty_resolved(couplings=CouplingsTransition.clear())
    result = _couplings_clear((state,), resolved, state.schema, None, (), {})
    assert isinstance(result, CouplingSet)
    assert len(result.couplings) == 0


def test_couplings_inherit_returns_selected_input_couplings():
    state_a = _make_dataset().state
    state_b = _make_dataset(values=(7, 8, 9)).state
    resolved = _empty_resolved(couplings=CouplingsTransition.inherit(input=1))
    result = _couplings_inherit(
        (state_a, state_b), resolved, state_b.schema, None, (), {}
    )
    assert result is state_b.couplings


def test_couplings_homogeneous_passes_when_inputs_match():
    state = _make_dataset().state
    resolved = _empty_resolved(couplings=CouplingsTransition.homogeneous())

    class _Op:
        name = "_HomogTest"

    result = _couplings_homogeneous(
        (state, state), resolved, state.schema, _Op(), (), {}
    )
    assert result is state.couplings


def test_couplings_homogeneous_raises_on_divergence():
    state = _make_dataset().state
    bound = replace(
        state,
        couplings=CouplingSet(couplings=(_NamedCoupling(name="a"),)),
    )

    class _Op:
        name = "_HomogTest"

    with pytest.raises(ValueError, match="structurally equal"):
        _couplings_homogeneous(
            (state, bound), _empty_resolved(), state.schema, _Op(), (), {}
        )


def test_apply_compose_derive_prunes_superseded_couplings():
    """A coupling from a losing-parent input whose field is superseded is pruned."""
    state = _make_dataset().state
    label_coupling = _NamedCoupling(name="label")
    other_coupling = _NamedCoupling(name="other")
    left = replace(state, couplings=CouplingSet(couplings=(label_coupling,)))
    right = replace(
        state,
        couplings=CouplingSet(couplings=(label_coupling, other_coupling)),
    )

    # Right loses "label" → right's label-coupling pruned; the same coupling
    # from left is kept (dedup union); right's "other" coupling survives.
    transition = CouplingsTransition(
        mode="derive", superseded_per_input=((1, ("label",)),)
    )
    output_schema = Schema(
        fields=(ValueField(name="label"), ValueField(name="other"))
    )
    result = _apply_compose_derive((left, right), transition, output_schema)
    names = sorted(c.output_field() for c in result.couplings)
    assert names == ["label", "other"]


def test_couplings_derive_under_compose_without_merged_fields_unions_inputs():
    state = _make_dataset().state
    left = replace(
        state,
        couplings=CouplingSet(couplings=(Materialize(field="a"),)),
    )
    right = replace(
        state,
        couplings=CouplingSet(couplings=(Materialize(field="b"),)),
    )
    resolved = _empty_resolved(
        schema=SchemaTransition.compose(),
        couplings=CouplingsTransition.derive(),
    )
    output_schema = Schema(
        fields=(
            state.schema.get("id"),
            state.schema.get("a"),
            ValueField(name="b"),
        )
    )

    result = _couplings_derive(
        (left, right), resolved, output_schema, None, (), {}
    )

    assert tuple(c.output_field() for c in result.couplings) == ("a", "b")


def test_apply_compose_derive_rewrites_and_prunes_references():
    state = _make_dataset().state
    composed_input = replace(
        state,
        couplings=CouplingSet(
            couplings=(Materialize(field="old"), Materialize(field="gone"))
        ),
    )
    transition = CouplingsTransition(
        mode="derive",
        rename_map=(("old", "new"),),
        dropped=("gone",),
    )
    output_schema = Schema(
        fields=(state.schema.get("id"), ValueField(name="new"))
    )

    result = _apply_compose_derive((composed_input,), transition, output_schema)

    assert result == CouplingSet(couplings=(Materialize(field="new"),))


def test_couplings_call_operator_forwards_composition_args():
    state = _make_dataset().state
    resolved = _empty_resolved(couplings=CouplingsTransition.custom())

    class _Op:
        def apply_couplings(self, *args, composed_schema=None, **kwargs):
            assert args == (state, state, "arg")
            assert composed_schema is state.schema
            assert kwargs == {"flag": "value"}
            return state.couplings

    result = _couplings_call_operator(
        (state, state), resolved, state.schema, _Op(), ("arg",), {"flag": "value"}
    )
    assert result is state.couplings


# ---------------------------------------------------------------------------
# Sources handlers
# ---------------------------------------------------------------------------


def test_sources_clear_returns_empty():
    state = _make_dataset().state
    result = _sources_clear((state,), _empty_resolved(), state.schema, None, (), {})
    assert result == ()


def test_sources_inherit_returns_selected_input_sources():
    state_a = _make_dataset().state
    state_b = _make_dataset(values=(7, 8)).state
    resolved = _empty_resolved(sources=SourcesTransition.inherit(input=1))
    result = _sources_inherit(
        (state_a, state_b), resolved, state_b.schema, None, (), {}
    )
    assert result is state_b.sources


def test_sources_derive_dedupes_union_by_source_id():
    state_a = _make_dataset().state
    # Repeating the same input must dedupe to one source.
    result_same = _sources_derive(
        (state_a, state_a), _empty_resolved(), state_a.schema, None, (), {}
    )
    assert len(result_same) == 1
    # Distinct sources stay distinct: synthesize a second state with a
    # different DatasetSourceInfo so the dedup key differs.
    from patchframe.dataset.provenance import DatasetSourceInfo

    alt_source = DatasetSourceInfo(
        source_id="alt", source_uri="memory://alt", source_type="dataframe"
    )
    state_b = DatasetState(
        schema=state_a.schema,
        table=state_a.table,
        couplings=state_a.couplings,
        sources=(alt_source,),
    )
    result_distinct = _sources_derive(
        (state_a, state_b), _empty_resolved(), state_a.schema, None, (), {}
    )
    assert len(result_distinct) == 2


def test_sources_call_operator_forwards_composition_args():
    state = _make_dataset().state
    resolved = _empty_resolved(sources=SourcesTransition.custom())

    class _Op:
        def combine_sources(self, *args, composed_schema=None, **kwargs):
            assert args == (state, state, "arg")
            assert composed_schema is state.schema
            assert kwargs == {"flag": "value"}
            return state.sources

    result = _sources_call_operator(
        (state, state), resolved, state.schema, _Op(), ("arg",), {"flag": "value"}
    )
    assert result is state.sources


# ---------------------------------------------------------------------------
# compute_output_state parity tests
#
# The same operator call through the existing _apply path and through the
# new compute_output_state must produce equivalent aspect values.
# ---------------------------------------------------------------------------


def _assert_states_equivalent(actual: DatasetState, expected: DatasetState):
    assert actual.schema.names() == expected.schema.names()
    assert actual.couplings == expected.couplings
    assert tuple(s.source_id for s in actual.sources) == tuple(
        s.source_id for s in expected.sources
    )
    pd.testing.assert_frame_equal(actual.table, expected.table)


def test_compute_output_state_parity_where_filter():
    ds = _make_dataset(values=(1, 2, 3, 4))
    op = pf.where.instance()
    pred = ds.table["a"] > 2
    expected_state = op(ds, pred).state
    actual_state = compute_output_state(op, (ds.state,), (pred,), {})
    _assert_states_equivalent(actual_state, expected_state)


def test_compute_output_state_parity_drop_field():
    schema = Schema(
        fields=(
            IndexField(name="id", dtype="int64"),
            ValueField(name="a", dtype="int64"),
            ValueField(name="b", dtype="int64"),
        )
    )
    df = pd.DataFrame(
        {"a": [1, 2], "b": [3, 4]}, index=pd.Index([10, 20], name="id")
    )
    ds = pf.make_from_dataframe(df, schema)
    op = pf.drop.instance()
    expected_state = op(ds, "b").state
    actual_state = compute_output_state(op, (ds.state,), ("b",), {})
    _assert_states_equivalent(actual_state, expected_state)


def test_compute_output_state_parity_rename_field():
    schema = Schema(
        fields=(
            IndexField(name="id", dtype="int64"),
            ValueField(name="a", dtype="int64"),
        )
    )
    df = pd.DataFrame({"a": [1, 2]}, index=pd.Index([10, 20], name="id"))
    ds = pf.make_from_dataframe(df, schema)
    op = pf.rename.instance()
    expected_state = op(ds, {"a": "alpha"}).state
    actual_state = compute_output_state(op, (ds.state,), ({"a": "alpha"},), {})
    _assert_states_equivalent(actual_state, expected_state)


def test_compute_output_state_resolves_merged_fields_in_final_schema():
    """The output schema returned by compute_output_state has no MergedFields.

    Even when an operator's apply_schema returns an intermediate schema
    carrying MergedField, compute_output_state collapses them via
    resolve_merged_fields before returning.
    """
    # Build a tiny composition operator that emits a MergedField.
    from patchframe.ops.base import CompositionOperator

    class _CombineFirst(CompositionOperator):
        transitions = TransitionPlan(
            schema=SchemaTransition.compose(),
            table=TableTransition.construct(),
            couplings=CouplingsTransition.clear(),
            sources=SourcesTransition.derive(),
            index_identity=IndexIdentityTransition.coalesce(),
        )

        def apply_schema(self, *states, **_):
            # Take first input's IndexField + a MergedField over the 'a' parents.
            left, right = states
            idx = left.schema.get("id")
            a_left = left.schema.get("a")
            a_right = right.schema.get("a")
            merged = MergedField.over(
                (FieldParent(0, a_left), FieldParent(1, a_right)),
                collision=ColumnCollisionStrategy(mode="keep", side="left"),
            )
            return Schema(fields=(idx, merged))

        def apply_table(self, *states, composed_schema=None, **_):
            return states[0].table  # left wins → reuse left's table

        def apply_couplings(self, *states, composed_schema=None, **_):
            return CouplingSet()

    state_a = _make_dataset().state
    state_b = _make_dataset(values=(9, 9, 9)).state
    op = _CombineFirst.instance()
    result = compute_output_state(op, (state_a, state_b), (), {})
    # MergedField must not appear in the final returned schema.
    assert not any(isinstance(f, MergedField) for f in result.schema)
