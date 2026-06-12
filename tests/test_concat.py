"""Tests for concat composition operators."""

from __future__ import annotations

import pandas as pd
import pytest

from patchframe.data.dimensions import IndexDimension
from patchframe.dataset.couplings import CouplingSet, Materialize
from patchframe.dataset.field_composition import ColumnCollisionStrategy
from patchframe.dataset.fields import DimensionField, IndexColumnField, IndexField, ValueField
from patchframe.dataset.identity import primary_index_identity
from patchframe.dataset.schema import Schema
from patchframe.ops.builtin.concat import concat, concat_columns, concat_rows
from patchframe.ops.builtin.keep import keep
from patchframe.ops.builtin.make_from_dataframe import make_from_dataframe


def _dataset(table: pd.DataFrame, *fields):
    return make_from_dataframe(table, Schema(fields=(IndexField(name="item_id"), *fields)))


class TestConcatRows:
    def test_stacks_rows_and_composes_value_field_dtype(self):
        left = _dataset(
            pd.DataFrame({"score": [1]}, index=["a"]),
            ValueField(name="score", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"score": [2.5]}, index=["b"]),
            ValueField(name="score", dtype=float),
        )

        result = concat_rows(left, right)

        assert result.schema.get("score").dtype is None
        assert result.table["score"].tolist() == [1, 2.5]

    def test_missing_columns_are_nullable(self):
        left = _dataset(
            pd.DataFrame({"score": [1]}, index=["a"]),
            ValueField(name="score", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"label": ["x"]}, index=["b"]),
            ValueField(name="label", dtype=str),
        )

        result = concat(left, right, axis=0)

        assert result.table["score"].isna().tolist() == [False, True]
        assert result.table["label"].isna().tolist() == [True, False]

    def test_dimension_fields_require_same_dimension(self):
        x = IndexDimension(name="x")
        y = IndexDimension(name="y")
        left = _dataset(
            pd.DataFrame({"start": [0]}, index=["a"]),
            DimensionField.from_dim(x, "start", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"start": [0]}, index=["b"]),
            DimensionField.from_dim(y, "start", dtype=int),
        )

        with pytest.raises(TypeError, match="dimension"):
            concat_rows(left, right)

    def test_rejects_duplicate_output_index(self):
        left = _dataset(
            pd.DataFrame({"score": [1]}, index=["a"]),
            ValueField(name="score", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"score": [2]}, index=["a"]),
            ValueField(name="score", dtype=int),
        )

        with pytest.raises(ValueError, match="index must be unique"):
            concat_rows(left, right)


class TestConcatColumns:
    def test_same_identity_index_fields_unify_without_strategy(self):
        # Identity alignment (join-dimensions-identity.md §5): same-name
        # IndexFields sharing one IndexIdentity are the same namespace, so
        # aligning them is not a collision and needs no strategy.
        base = _dataset(
            pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}, index=["i", "j"]),
            ValueField(name="a", dtype=int),
            ValueField(name="b", dtype=str),
        )
        left = keep(base, ["item_id", "a"])
        right = keep(base, ["item_id", "b"])

        result = concat_columns(left, right)

        assert result.schema.names() == ("item_id", "a", "b")
        assert result.table["a"].tolist() == [1, 2]
        assert result.table["b"].tolist() == ["x", "y"]
        assert primary_index_identity(result) == primary_index_identity(base)

    def test_different_identity_index_fields_still_collide(self):
        # Distinct namespaces whose labels merely agree as values stay a real
        # collision: the caller must choose a strategy explicitly.
        left = _dataset(
            pd.DataFrame({"a": [1]}, index=["i"]),
            ValueField(name="a", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"b": [2]}, index=["i"]),
            ValueField(name="b", dtype=int),
        )

        with pytest.raises(ValueError, match="field collision for 'item_id'"):
            concat_columns(left, right)

    def test_default_collision_policy_raises(self):
        left = _dataset(
            pd.DataFrame({"score": [1]}, index=["a"]),
            ValueField(name="score", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"score": [2]}, index=["a"]),
            ValueField(name="score", dtype=int),
        )

        with pytest.raises(ValueError, match="field collision"):
            concat_columns(left, right)

    def test_update_missing_collision_policy_fills_left_nulls(self):
        left = _dataset(
            pd.DataFrame({"score": pd.Series([1, pd.NA], index=["a", "b"], dtype="Int64")}),
            ValueField(name="score", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"score": pd.Series([9, 2], index=["a", "b"], dtype="Int64")}),
            ValueField(name="score", dtype=int),
        )

        result = concat(
            left,
            right,
            axis=1,
            collision=ColumnCollisionStrategy(mode="update_missing", side="left"),
        )

        assert result.table["score"].tolist() == [1, 2]

    def test_primary_value_field_is_downgraded_on_column_add(self):
        left = _dataset(
            pd.DataFrame({"left_score": [1]}, index=["a"]),
            ValueField(name="left_score", dtype=int, primary=True),
        )
        right = _dataset(
            pd.DataFrame({"right_score": [2]}, index=["a"]),
            ValueField(name="right_score", dtype=int, primary=True),
        )

        result = concat_columns(left, right, collision="keep")

        assert result.schema.get("left_score").primary
        assert not result.schema.get("right_score").primary

    def test_secondary_index_field_becomes_table_backed_index_column(self):
        left = _dataset(
            pd.DataFrame({"left_score": [1, 2]}, index=["a", "b"]),
            ValueField(name="left_score", dtype=int),
        )
        right = make_from_dataframe(
            pd.DataFrame({"right_score": [10, 30]}, index=["a", "c"]),
            Schema(
                fields=(
                    IndexField(name="right_id"),
                    ValueField(name="right_score", dtype=int),
                )
            ),
        )

        result = concat_columns(left, right)

        assert isinstance(result.schema.get("item_id"), IndexField)
        assert isinstance(result.schema.get("right_id"), IndexColumnField)
        assert result.table["right_id"].isna().tolist() == [False, True, False]
        assert result.table["right_id"].iloc[0] == "a"
        assert result.table["right_id"].iloc[2] == "c"


class TestConcatRowCouplings:
    def test_preserves_identical_coupling_sets(self):
        coupling_set = CouplingSet(couplings=(Materialize(field="score"),))
        left = _dataset(
            pd.DataFrame({"score": [1]}, index=["a"]),
            ValueField(name="score", dtype=int),
        ).replace_state(couplings=coupling_set)
        right = _dataset(
            pd.DataFrame({"score": [2]}, index=["b"]),
            ValueField(name="score", dtype=int),
        ).replace_state(couplings=coupling_set)

        result = concat_rows(left, right)

        assert result.couplings == coupling_set

    def test_rejects_different_row_couplings(self):
        left = _dataset(
            pd.DataFrame({"score": [1]}, index=["a"]),
            ValueField(name="score", dtype=int),
        ).replace_state(couplings=CouplingSet(couplings=(Materialize(field="score"),)))
        right = _dataset(
            pd.DataFrame({"score": [2]}, index=["b"]),
            ValueField(name="score", dtype=int),
        )

        with pytest.raises(ValueError, match="Consume coupled fields"):
            concat_rows(left, right)
