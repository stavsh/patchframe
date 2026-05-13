"""Tests for merge composition operator."""

from __future__ import annotations

import pandas as pd
import pytest

from patchframe.data.dimensions import IndexDimension
from patchframe.dataset.couplings import CouplingSet, Materialize
from patchframe.dataset.field_composition import ColumnCollisionStrategy
from patchframe.dataset.fields import DimensionField, IndexField, ValueField
from patchframe.dataset.schema import Schema
from patchframe.ops.builtin.join import join
from patchframe.ops.builtin.make_from_dataframe import make_from_dataframe
from patchframe.ops.builtin.merge import merge


def _dataset(table: pd.DataFrame, *fields, couplings: CouplingSet | None = None):
    return make_from_dataframe(
        table,
        Schema(fields=(IndexField(name="item_id"), *fields)),
        couplings=couplings,
    )


def _join_plan(table: pd.DataFrame):
    return make_from_dataframe(
        table,
        Schema(
            fields=(
                IndexField(name="join_id"),
                ValueField(name="left_index"),
                ValueField(name="right_index"),
            )
        ),
    )


class TestMerge:
    def test_materializes_outer_index_join_plan(self):
        left = _dataset(
            pd.DataFrame({"score": [10, 20]}, index=["a", "b"]),
            ValueField(name="score", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"label": [200, 300]}, index=["b", "c"]),
            ValueField(name="label", dtype=int),
        )
        plan = join(left, right, how="outer")

        result = merge(left, right, plan)

        assert result.schema.names() == (
            "join_id",
            "left_index",
            "right_index",
            "score",
            "label",
        )
        assert result.table.index.name == "join_id"
        assert result.table["left_index"].isna().tolist() == [False, False, True]
        assert result.table["right_index"].isna().tolist() == [True, False, False]
        assert result.table["score"].isna().tolist() == [False, False, True]
        assert result.table["label"].isna().tolist() == [True, False, False]
        assert result.table["score"].iloc[:2].tolist() == [10, 20]
        assert result.table["label"].iloc[1:].tolist() == [200, 300]

    def test_preserves_join_plan_metadata_columns(self):
        left = _dataset(
            pd.DataFrame({"left_score": [10]}, index=["a"]),
            ValueField(name="left_score", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"right_score": [20]}, index=["a"]),
            ValueField(name="right_score", dtype=int),
        )
        plan = join(left, right).replace_state(
            schema=join(left, right).schema.add(ValueField(name="match_score", dtype=float)),
            table=join(left, right).table.assign(match_score=pd.Series([0.5], dtype="Float64")),
        )

        result = merge(left, right, plan)

        assert result.table["match_score"].tolist() == [0.5]
        assert result.schema.names() == (
            "join_id",
            "left_index",
            "right_index",
            "match_score",
            "left_score",
            "right_score",
        )

    def test_default_collision_policy_raises_for_same_name_columns(self):
        left = _dataset(
            pd.DataFrame({"score": [10]}, index=["a"]),
            ValueField(name="score", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"score": [20]}, index=["a"]),
            ValueField(name="score", dtype=int),
        )
        plan = join(left, right)

        with pytest.raises(ValueError, match="field collision"):
            merge(left, right, plan)

    def test_update_missing_collision_policy_fills_left_nulls(self):
        left = _dataset(
            pd.DataFrame(
                {"score": pd.Series([10, pd.NA], index=["a", "b"], dtype="Int64")}
            ),
            ValueField(name="score", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"score": pd.Series([99, 20], index=["a", "b"], dtype="Int64")}),
            ValueField(name="score", dtype=int),
        )
        plan = join(left, right)

        result = merge(
            left,
            right,
            plan,
            collision=ColumnCollisionStrategy(mode="update_missing", side="left"),
        )

        assert result.table["score"].tolist() == [10, 20]

    def test_dimension_collision_requires_matching_dimension(self):
        x = IndexDimension(name="x")
        y = IndexDimension(name="y")
        left = _dataset(
            pd.DataFrame({"position": [1]}, index=["a"]),
            DimensionField.from_dim(x, "position", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"position": [1]}, index=["a"]),
            DimensionField.from_dim(y, "position", dtype=int),
        )
        plan = join(left, right)

        with pytest.raises(TypeError, match="dimension"):
            merge(left, right, plan, collision="update_missing")

    def test_missing_join_plan_mapping_column_raises(self):
        left = _dataset(pd.DataFrame({"score": [10]}, index=["a"]), ValueField(name="score"))
        right = _dataset(pd.DataFrame({"label": [20]}, index=["a"]), ValueField(name="label"))
        plan = make_from_dataframe(
            pd.DataFrame({"left_index": ["a"]}, index=pd.RangeIndex(1, name="join_id")),
            Schema(fields=(IndexField(name="join_id"), ValueField(name="left_index"))),
        )

        with pytest.raises(ValueError, match="missing required mapping columns"):
            merge(left, right, plan)

    def test_missing_input_label_raises(self):
        left = _dataset(pd.DataFrame({"score": [10]}, index=["a"]), ValueField(name="score"))
        right = _dataset(pd.DataFrame({"label": [20]}, index=["a"]), ValueField(name="label"))
        plan = _join_plan(
            pd.DataFrame(
                {"left_index": ["missing"], "right_index": ["a"]},
                index=pd.RangeIndex(1, name="join_id"),
            )
        )

        with pytest.raises(ValueError, match="missing from left dataset"):
            merge(left, right, plan)

    def test_reserved_mapping_names_are_rejected_in_inputs(self):
        left = _dataset(
            pd.DataFrame({"left_index": ["x"]}, index=["a"]),
            ValueField(name="left_index"),
        )
        right = _dataset(pd.DataFrame({"label": [20]}, index=["a"]), ValueField(name="label"))
        plan = join(left, right)

        with pytest.raises(ValueError, match="reserved join-plan mapping names"):
            merge(left, right, plan)

    def test_couplings_are_unioned_like_column_composition(self):
        left_coupling = Materialize(field="left_score")
        right_coupling = Materialize(field="right_score")
        left = _dataset(
            pd.DataFrame({"left_score": [10]}, index=["a"]),
            ValueField(name="left_score", dtype=int),
            couplings=CouplingSet(couplings=(left_coupling,)),
        )
        right = _dataset(
            pd.DataFrame({"right_score": [20]}, index=["a"]),
            ValueField(name="right_score", dtype=int),
            couplings=CouplingSet(couplings=(right_coupling,)),
        )
        plan = join(left, right)

        result = merge(left, right, plan)

        assert result.couplings.couplings == (left_coupling, right_coupling)
        result.coupling_engine()
