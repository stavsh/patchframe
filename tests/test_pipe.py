"""Tests for ``pipe`` — the constrained ``.table`` escape — and ``table_transform``.

``pipe`` runs a user ``Dataset -> DataFrame`` fn and re-validates the returned
table against a declared (or schema-inferred) ``TransitionPlan`` (the "anonymous
operator" model). These cover the two supported shapes (preserve / construct),
the schema-presence shortcut, the lineage carried forward (sources, couplings),
and the fail-loud guards that make the escape *constrained* rather than a raw
``make_from_dataframe`` rebuild.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

import patchframe as pf
from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.identity import primary_index_identity


def _members() -> pf.Dataset:
    df = pd.DataFrame(
        {"a": [1, 2, 3, 4], "g": ["x", "x", "y", "y"]},
        index=pd.Index([10, 11, 12, 13], name="id"),
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="id"),
            pf.ValueField(name="a", dtype=int),
            pf.ValueField(name="g", dtype=str),
        )
    )
    return pf.make_from_dataframe(df, schema)


_GROUP_SCHEMA = pf.Schema(
    fields=(pf.IndexField(name="g"), pf.ValueField(name="total", dtype=int))
)

#: An explicit construct plan (== the schema-inferred default), for the tests
#: that assert explicit transitions still work.
_CONSTRUCT = pf.TransitionPlan(
    schema=pf.SchemaTransition.construct(),
    couplings=pf.CouplingsTransition.clear(),
    index_identity=pf.IndexIdentityTransition.mint(),
    sources=pf.SourcesTransition.inherit(),
)


# Transform fns receive the input Dataset and return the new table.
def _drop_small(dataset: pf.Dataset) -> pd.DataFrame:
    table = dataset.table
    return table[table["a"] >= 2]


def _double_a(dataset: pf.Dataset) -> pd.DataFrame:
    table = dataset.table  # an isolated copy — mutating it is safe
    table["a"] = table["a"] * 2
    return table


def _rollup_by_group(dataset: pf.Dataset) -> pd.DataFrame:
    out = dataset.table.groupby("g", as_index=True)["a"].sum().to_frame("total")
    out.index.name = "g"
    return out


def _reindex_from_zero(dataset: pf.Dataset) -> pd.DataFrame:
    out = dataset.table.reset_index(drop=True)
    out.index.name = "id"
    return out


def _drop_g(dataset: pf.Dataset) -> pd.DataFrame:
    return dataset.table[["a"]]


def _identity(dataset: pf.Dataset) -> pd.DataFrame:
    return dataset.table


def _double(x: int) -> int:
    return x * 2


# --------------------------------------------------------------------------- #
# Shape 1 — preserve / inherit (a row/value transform; the no-schema default).
# --------------------------------------------------------------------------- #


class TestPreserve:
    def test_default_is_preserve_filter(self):
        kept = pf.pipe(_members(), _drop_small)
        assert list(kept.table.index) == [11, 12, 13]
        assert kept.schema.names() == ("id", "a", "g")

    def test_no_schema_copies_input_schema(self):
        ds = _members()
        kept = pf.pipe(ds, _drop_small)
        assert kept.schema.names() == ds.schema.names()
        assert primary_index_identity(kept.state) == primary_index_identity(ds.state)

    def test_sources_preserved(self):
        ds = _members()
        assert len(ds.sources) == 1
        assert len(pf.pipe(ds, _drop_small).sources) == 1

    def test_recompute_values(self):
        kept = pf.pipe(_members(), _double_a)
        assert list(kept.table["a"]) == [2, 4, 6, 8]
        assert kept.schema.names() == ("id", "a", "g")

    def test_input_table_not_mutated_in_place(self):
        ds = _members()
        pf.pipe(ds, _double_a)
        assert list(ds.table["a"]) == [1, 2, 3, 4]

    def test_preserve_mint_allows_renumbered_rows(self):
        ds = _members()
        plan = pf.TransitionPlan(
            schema=pf.SchemaTransition.preserve(),
            index_identity=pf.IndexIdentityTransition.mint(),
        )
        out = pf.pipe(ds, _reindex_from_zero, transitions=plan)
        assert list(out.table.index) == [0, 1, 2, 3]
        assert primary_index_identity(out.state) != primary_index_identity(ds.state)


# --------------------------------------------------------------------------- #
# Shape 2 — construct / mint (a rebuild; selected by supplying a schema).
# --------------------------------------------------------------------------- #


class TestConstruct:
    def test_schema_alone_infers_construct(self):
        # No transitions=, but a schema= -> construct (mint, validate, rebuild).
        out = pf.pipe(_members(), _rollup_by_group, schema=_GROUP_SCHEMA)
        table = out.table.sort_index()
        assert list(table.index) == ["x", "y"]
        assert table.index.name == "g"
        assert dict(table["total"]) == {"x": 3, "y": 7}
        assert out.schema.names() == ("g", "total")

    def test_mints_fresh_identity(self):
        ds = _members()
        out = pf.pipe(ds, _rollup_by_group, schema=_GROUP_SCHEMA)
        assert primary_index_identity(out.state) != primary_index_identity(ds.state)

    def test_sources_carried_forward(self):
        # The concrete win over make_from_dataframe: provenance survives a rebuild.
        out = pf.pipe(_members(), _rollup_by_group, schema=_GROUP_SCHEMA)
        assert len(out.sources) == 1

    def test_explicit_construct_transitions_match_inference(self):
        out = pf.pipe(_members(), _rollup_by_group, schema=_GROUP_SCHEMA, transitions=_CONSTRUCT)
        assert out.schema.names() == ("g", "total")
        assert dict(out.table.sort_index()["total"]) == {"x": 3, "y": 7}

    def test_sources_cleared_when_declared(self):
        plan = pf.TransitionPlan(
            schema=pf.SchemaTransition.construct(),
            couplings=pf.CouplingsTransition.clear(),
            index_identity=pf.IndexIdentityTransition.mint(),
            sources=pf.SourcesTransition.clear(),
        )
        out = pf.pipe(_members(), _rollup_by_group, schema=_GROUP_SCHEMA, transitions=plan)
        assert len(out.sources) == 0


# --------------------------------------------------------------------------- #
# Couplings.
# --------------------------------------------------------------------------- #


def _coupled_dataset() -> pf.Dataset:
    ds = _members()
    with ds.context() as ctx:
        pf.map_fields(ctx.field("a"), _double, out="b")
        return ctx.dataset


class TestCouplings:
    def test_preserve_derives_surviving_couplings(self):
        coupled = _coupled_dataset()
        assert len(coupled.couplings.couplings) == 1
        kept = pf.pipe(coupled, _drop_small)
        assert len(kept.couplings.couplings) == 1

    def test_clear_drops_couplings(self):
        coupled = _coupled_dataset()
        plan = pf.TransitionPlan(
            schema=pf.SchemaTransition.preserve(),
            index_identity=pf.IndexIdentityTransition.inherit(),
            couplings=pf.CouplingsTransition.clear(),
        )
        kept = pf.pipe(coupled, _drop_small, transitions=plan)
        assert kept.couplings == CouplingSet()

    def test_construct_clears_couplings_without_warning(self):
        coupled = _coupled_dataset()
        # Construct has no input field lineage, so the schema-inferred default
        # explicitly clears couplings.
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            out = pf.pipe(coupled, _rollup_by_group, schema=_GROUP_SCHEMA)
        assert len(out.couplings.couplings) == 0
        assert not [w for w in records if "coupling" in str(w.message)]

    def test_construct_rejects_derived_couplings(self):
        plan = pf.TransitionPlan(
            schema=pf.SchemaTransition.construct(),
            couplings=pf.CouplingsTransition.derive(),
            index_identity=pf.IndexIdentityTransition.mint(),
        )
        with pytest.raises(ValueError, match="couplings=derive"):
            pf.pipe(_coupled_dataset(), _rollup_by_group, schema=_GROUP_SCHEMA, transitions=plan)

    def test_inherit_validates_coupling_references_at_pipe_boundary(self):
        plan = pf.TransitionPlan(
            schema=pf.SchemaTransition.construct(),
            couplings=pf.CouplingsTransition.inherit(),
            index_identity=pf.IndexIdentityTransition.mint(),
        )
        with pytest.raises(ValueError, match="Coupling input not in schema"):
            pf.pipe(_coupled_dataset(), _rollup_by_group, schema=_GROUP_SCHEMA, transitions=plan)


# --------------------------------------------------------------------------- #
# Fail-loud guards — what makes the escape *constrained*.
# --------------------------------------------------------------------------- #


class TestFailLoud:
    def test_return_honesty_preserve_missing_column(self):
        # No schema -> preserve; the fn dropped a field -> schema mismatch.
        with pytest.raises(ValueError):
            pf.pipe(_members(), _drop_g)

    def test_inherit_rejects_labels_outside_input(self):
        with pytest.raises(ValueError, match="left the input identity namespace"):
            pf.pipe(_members(), _reindex_from_zero)

    def test_construct_requires_mint(self):
        plan = pf.TransitionPlan(
            schema=pf.SchemaTransition.construct(),
            index_identity=pf.IndexIdentityTransition.inherit(),
        )
        with pytest.raises(ValueError, match="mint"):
            pf.pipe(_members(), _identity, transitions=plan)

    def test_construct_requires_schema(self):
        plan = pf.TransitionPlan(
            schema=pf.SchemaTransition.construct(),
            couplings=pf.CouplingsTransition.clear(),
            index_identity=pf.IndexIdentityTransition.mint(),
        )
        with pytest.raises(ValueError, match="requires an explicit schema"):
            pf.pipe(_members(), _rollup_by_group, transitions=plan)

    def test_unsupported_schema_mode(self):
        plan = pf.TransitionPlan(schema=pf.SchemaTransition.rewrite())
        with pytest.raises(ValueError, match="not supported"):
            pf.pipe(_members(), _identity, transitions=plan)

    def test_non_dataframe_return(self):
        with pytest.raises(TypeError, match="DataFrame"):
            pf.pipe(_members(), lambda ds: 42)

    def test_returning_the_dataset_fails_loud(self):
        # Common mistake: returning the dataset instead of its table.
        with pytest.raises(TypeError, match="DataFrame"):
            pf.pipe(_members(), lambda ds: ds)

    def test_preserve_with_mismatched_schema_arg(self):
        # schema alone would infer construct, so assert preserve explicitly.
        plan = pf.TransitionPlan(schema=pf.SchemaTransition.preserve())
        with pytest.raises(ValueError, match="different fields"):
            pf.pipe(_members(), _identity, schema=_GROUP_SCHEMA, transitions=plan)

    def test_transitions_must_be_a_plan(self):
        with pytest.raises(TypeError, match="TransitionPlan"):
            pf.pipe(_members(), _identity, transitions="preserve")

    def test_schema_and_transitions_are_keyword_only(self):
        with pytest.raises(TypeError, match="keyword-only"):
            pf.pipe(_members(), _identity, _GROUP_SCHEMA)


# --------------------------------------------------------------------------- #
# table_transform — the named form (a pipe wrapper binding a static contract).
# --------------------------------------------------------------------------- #


@pf.table_transform
def _t_filter(dataset: pf.Dataset) -> pd.DataFrame:
    table = dataset.table
    return table[table["a"] >= 2]


@pf.table_transform(schema=_GROUP_SCHEMA)  # schema -> construct, inferred
def _t_rollup(dataset: pf.Dataset) -> pd.DataFrame:
    out = dataset.table.groupby("g", as_index=True)["a"].sum().to_frame("total")
    out.index.name = "g"
    return out


@pf.table_transform(schema=_GROUP_SCHEMA)
def _t_rollup_above(dataset: pf.Dataset, *, threshold: int) -> pd.DataFrame:
    table = dataset.table
    sub = table[table["a"] >= threshold]
    out = sub.groupby("g", as_index=True)["a"].sum().to_frame("total")
    out.index.name = "g"
    return out


@pf.table_transform
def _t_scaled_filter(dataset: pf.Dataset, minimum: int, scale: int) -> pd.DataFrame:
    table = dataset.table
    out = table[table["a"] >= minimum].copy()
    out["a"] = out["a"] * scale
    return out


class TestTableTransform:
    def test_bare_decorator_is_preserve_filter(self):
        out = _t_filter(_members())
        assert list(out.table.index) == [11, 12, 13]
        assert out.schema.names() == ("id", "a", "g")
        assert len(out.sources) == 1  # provenance carried by the default

    def test_static_schema_rebuild_preserves_sources(self):
        out = _t_rollup(_members())
        assert dict(out.table.sort_index()["total"]) == {"x": 3, "y": 7}
        assert out.schema.names() == ("g", "total")
        assert len(out.sources) == 1

    def test_forwards_extra_kwargs_to_fn(self):
        out = _t_rollup_above(_members(), threshold=3)
        # Only a >= 3 survives: rows 12,13 (both g="y").
        assert dict(out.table["total"]) == {"y": 7}

    def test_forwards_extra_positional_args_after_dataset(self):
        out = _t_scaled_filter(_members(), 3, 10)
        assert list(out.table.index) == [12, 13]
        assert list(out.table["a"]) == [30, 40]

    def test_reusable_across_datasets(self):
        first = _t_filter(_members())
        second = _t_filter(_members())
        assert list(first.table.index) == list(second.table.index) == [11, 12, 13]

    def test_keeps_underlying_fn_accessible(self):
        assert _t_rollup.__pipe_fn__.__name__ == "_t_rollup"
