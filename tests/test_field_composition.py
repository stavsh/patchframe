"""Tests for registered field composition policies."""

from __future__ import annotations

import pandas as pd
import pytest

from patchframe.data.dimensions import IndexDimension, TemporalDimension
from patchframe.dataset.field_composition import (
    ColumnCollisionStrategy,
    compose_column,
    compose_key,
    compose_rows,
    field_policy_for,
    resolve_column_collision,
)
from patchframe.dataset.fields import (
    DimensionField,
    Field,
    IndexColumnField,
    IndexField,
    ValueField,
)


class TestPolicyRegistry:
    def test_returns_registered_policy_for_field_subclass(self):
        policy = field_policy_for(ValueField(name="score", dtype=float))
        assert type(policy).__name__ == "ValueFieldCompositionPolicy"

    def test_falls_back_to_base_field_policy(self):
        policy = field_policy_for(Field(name="raw", dtype=str))
        assert type(policy).__name__ == "FieldCompositionPolicy"


class TestComposeRows:
    def test_value_fields_allow_dtype_widening(self):
        result = compose_rows(
            (
                ValueField(name="score", dtype=int),
                ValueField(name="score", dtype=float),
            )
        )

        assert isinstance(result, ValueField)
        assert result.dtype is None
        assert result.nullable

    def test_base_policy_requires_matching_dtype(self):
        with pytest.raises(TypeError, match="dtype"):
            compose_rows((Field(name="x", dtype=int), Field(name="x", dtype=float)))

    def test_dimension_fields_require_same_dimension(self):
        x = IndexDimension(name="x")
        result = compose_rows(
            (
                DimensionField.from_dim(x, "start", dtype=int),
                DimensionField.from_dim(x, "start", dtype=int),
            )
        )

        assert isinstance(result, DimensionField)
        assert result.dimension == x

    def test_dimension_fields_reject_different_dimensions(self):
        x = IndexDimension(name="x")
        time = TemporalDimension(name="time", sample_rate=16000)

        with pytest.raises(TypeError, match="dimension"):
            compose_rows(
                (
                    DimensionField.from_dim(x, "start", dtype=int),
                    DimensionField.from_dim(time, "start", dtype=int),
                )
            )


class TestComposeColumn:
    def test_primary_field_is_downgraded_when_same_primary_type_exists(self):
        existing = (ValueField(name="left_score", dtype=float, primary=True),)
        result = compose_column(
            ValueField(name="right_score", dtype=float, primary=True),
            existing,
        )

        assert isinstance(result, ValueField)
        assert not result.primary

    def test_primary_field_is_preserved_when_no_same_primary_type_exists(self):
        result = compose_column(ValueField(name="score", dtype=float, primary=True), ())

        assert result.primary

    def test_index_field_downgrades_to_index_column_field(self):
        result = compose_column(IndexField(name="right_id"), (IndexField(name="left_id"),))

        assert isinstance(result, IndexColumnField)
        assert not result.primary
        assert result.nullable


class TestComposeKey:
    def test_key_composition_uses_row_compatibility(self):
        result = compose_key(
            (
                ValueField(name="id", dtype=int),
                ValueField(name="id", dtype=float),
            )
        )

        assert isinstance(result, ValueField)
        assert result.dtype is None

    def test_dimension_key_composition_rejects_mismatched_dimensions(self):
        x = IndexDimension(name="x")
        y = IndexDimension(name="y")

        with pytest.raises(TypeError, match="dimension"):
            compose_key(
                (
                    DimensionField.from_dim(x, "position", dtype=int),
                    DimensionField.from_dim(y, "position", dtype=int),
                )
            )


class TestColumnCollisionStrategy:
    def test_update_missing_fills_null_values_from_other_side(self):
        left = pd.Series([1, None, 3, pd.NA], dtype="Int64")
        right = pd.Series([9, 2, 8, 4], dtype="Int64")

        result = resolve_column_collision(
            left,
            right,
            ColumnCollisionStrategy(mode="update_missing", side="left"),
        )

        assert result.tolist() == [1, 2, 3, 4]

    def test_update_missing_can_raise_on_conflicting_non_null_values(self):
        left = pd.Series([1, None], dtype="Int64")
        right = pd.Series([9, 2], dtype="Int64")

        with pytest.raises(ValueError, match="conflicting"):
            resolve_column_collision(
                left,
                right,
                ColumnCollisionStrategy(
                    mode="update_missing",
                    side="left",
                    on_conflict="raise",
                ),
            )

    def test_keep_uses_selected_side(self):
        left = pd.Series([1, None], dtype="Int64")
        right = pd.Series([9, 2], dtype="Int64")

        result = resolve_column_collision(
            left,
            right,
            ColumnCollisionStrategy(mode="keep", side="right"),
        )

        assert result.tolist() == [9, 2]

    def test_error_mode_raises(self):
        with pytest.raises(ValueError, match="mode='error'"):
            resolve_column_collision(
                pd.Series([1]),
                pd.Series([2]),
                ColumnCollisionStrategy(mode="error"),
            )
