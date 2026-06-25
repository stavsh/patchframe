"""Tests for the reduce operator and the reducing-operator vocabulary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import patchframe as pf
from patchframe.ops.builtin.reduce import (
    Count,
    Distinct,
    Max,
    Mean,
    Min,
    ReduceCoupling,
    ReducingOperator,
    Sum,
)
from patchframe.ops.transitions import Cardinality


def _members() -> pf.Dataset:
    df = pd.DataFrame(
        {"k": ["a", "a", "b"], "v": [1.0, 2.0, 5.0], "u": ["x", "x", "y"]},
        index=pd.Index(["r0", "r1", "r2"], name="id"),
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="id"),
            pf.ValueField(name="k", dtype=str),
            pf.ValueField(name="v", dtype=float),
            pf.ValueField(name="u", dtype=str),
        )
    )
    return pf.make_from_dataframe(df, schema)


class TestReduce:
    def test_multi_agg_values_and_honest_dtypes(self):
        out = pf.reduce(
            _members(),
            "k",
            aggs={
                "total_v": Sum.on("v"),
                "n": Count.on(),
                "uniq_u": Distinct.on("u"),
                "avg_v": Mean.on("v"),
                "min_v": Min.on("v"),
            },
        )
        table = out.table.sort_index()
        assert list(table.index) == ["a", "b"]
        assert table.loc["a", "total_v"] == 3.0
        assert table.loc["a", "n"] == 2
        assert table.loc["a", "uniq_u"] == 1
        assert table.loc["a", "avg_v"] == 1.5
        assert table.loc["b", "total_v"] == 5.0
        assert table.loc["b", "n"] == 1
        # Honest output dtypes come from the reductions' declarations.
        assert str(table["total_v"].dtype) == "Float64"
        assert str(table["n"].dtype) == "Int64"
        assert str(table["uniq_u"].dtype) == "Int64"

    def test_ratios_are_post_reduce_expressions(self):
        # A ratio is not a reduction: aggregate the additive components, divide after.
        out = pf.reduce(_members(), "k", aggs={"total_v": Sum.on("v"), "n": Count.on()})
        per_event = (out.table["total_v"] / out.table["n"]).sort_index()
        assert per_event.loc["a"] == 1.5
        assert per_event.loc["b"] == 5.0

    def test_standalone_reducer_over_bundle_handle(self):
        # A reducing operator only accepts a BundleField handle and returns a
        # FieldReturn — the lift (partition) happens before it.
        groups = pf.partition(_members(), "k", into="fiber")
        handle = Sum.on("v")(groups.field("fiber"), out="total_v")
        assert isinstance(handle, pf.FieldHandle)
        realized = handle.collect()
        assert realized.table["total_v"].sort_index().tolist() == [3.0, 5.0]

    def test_domain_totality_empty_fiber(self):
        members = _members()
        keys = pf.make_from_dataframe(
            pd.DataFrame(index=pd.Index(["a", "b", "c"], name="k")),
            pf.Schema(fields=(pf.IndexField(name="k"),)),
        )
        linked = pf.link(members, keys, "k")
        out = pf.reduce(
            linked, "k", aggs={"total_v": Sum.on("v"), "n": Count.on()}, domain=keys
        )
        table = out.table
        assert list(table.index) == ["a", "b", "c"]  # total over the domain
        # 'c' had no members -> empty fiber -> additive reductions are 0.
        assert table.loc["c", "total_v"] == 0.0
        assert table.loc["c", "n"] == 0

    def test_reduce_inherits_domain_index_identity(self):
        members = _members()
        keys = pf.make_from_dataframe(
            pd.DataFrame(index=pd.Index(["a", "b"], name="k")),
            pf.Schema(fields=(pf.IndexField(name="k"),)),
        )
        linked = pf.link(members, keys, "k")
        out = pf.reduce(linked, "k", aggs={"n": Count.on()}, domain=keys)
        from patchframe.dataset.identity import primary_index_identity

        assert primary_index_identity(out.state) == primary_index_identity(keys.state)

    def test_reductions_declare_reduce_cardinality_and_kernel(self):
        assert Sum.cardinality is Cardinality.REDUCE
        assert Distinct.cardinality is Cardinality.REDUCE
        # The reserved bulk kernel for a future engine fuser (declared, not consumed).
        assert Sum.bulk_kernel == "sum"
        assert Distinct.bulk_kernel == "nunique"
        assert Count.bulk_kernel == "size"

    def test_reduce_coupling_carries_the_typed_reducer(self):
        # The graph carries the reducing operator, so a fuser can introspect it.
        reducer = Sum.on("v")
        coupling = ReduceCoupling(fiber="fiber", output="total_v", reducer=reducer, column="v")
        assert coupling.input_fields() == ("fiber",)
        assert coupling.output_field() == "total_v"
        assert coupling.reducer.bulk_kernel == "sum"

    def test_rejects_non_reduction_agg(self):
        with pytest.raises(TypeError, match="must be a reduction"):
            pf.reduce(_members(), "k", aggs={"bad": "not a reduction"})

    def test_sum_requires_a_column(self):
        with pytest.raises(ValueError, match="requires a column"):
            pf.reduce(_members(), "k", aggs={"bad": Sum.on()})

    def test_reduce_is_an_operator(self):
        assert issubclass(Sum, ReducingOperator)
        assert issubclass(ReducingOperator, pf.DatasetOperator)
