"""Tests for join-plan construction."""

from __future__ import annotations

import pandas as pd
import pytest

from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.dataset.fields import DimensionedSliceField, IndexField, ValueField
from patchframe.dataset.provenance import DatasetSourceInfo
from patchframe.dataset.schema import Schema
from patchframe.ops.builtin.join import DimensionJoin, FieldEqualityJoin, join
from patchframe.ops.builtin.make_from_dataframe import make_from_dataframe


def _dataset(table: pd.DataFrame, *fields):
    return make_from_dataframe(table, Schema(fields=(IndexField(name="item_id"), *fields)))


class TestIndexJoin:
    def test_inner_join_by_index_labels(self):
        left = _dataset(
            pd.DataFrame({"score": [1, 2, 3]}, index=["a", "b", "c"]),
            ValueField(name="score", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"label": [10, 20, 30]}, index=["b", "c", "d"]),
            ValueField(name="label", dtype=int),
        )

        result = join(left, right)

        assert result.schema.names() == ("join_id", "left_index", "right_index")
        assert isinstance(result.schema.get("join_id"), IndexField)
        assert result.table.index.name == "join_id"
        assert result.table.index.is_unique
        assert result.table.to_dict("list") == {
            "left_index": ["b", "c"],
            "right_index": ["b", "c"],
        }

    def test_left_join_by_index_labels(self):
        left = _dataset(
            pd.DataFrame({"score": [1, 2, 3]}, index=["a", "b", "c"]),
            ValueField(name="score", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"label": [10, 20]}, index=["b", "c"]),
            ValueField(name="label", dtype=int),
        )

        result = join(left, right, how="left")

        assert result.table["left_index"].tolist() == ["a", "b", "c"]
        assert result.table["right_index"].isna().tolist() == [True, False, False]
        assert result.table["right_index"].iloc[1:].tolist() == ["b", "c"]

    def test_right_join_by_index_labels(self):
        left = _dataset(
            pd.DataFrame({"score": [1]}, index=["a"]),
            ValueField(name="score", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"label": [10, 20]}, index=["a", "b"]),
            ValueField(name="label", dtype=int),
        )

        result = join(left, right, how="right")

        assert result.table["left_index"].isna().tolist() == [False, True]
        assert result.table["left_index"].iloc[0] == "a"
        assert result.table["right_index"].tolist() == ["a", "b"]

    def test_outer_join_by_index_labels(self):
        left = _dataset(
            pd.DataFrame({"score": [1, 2]}, index=["a", "b"]),
            ValueField(name="score", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"label": [10, 20]}, index=["b", "c"]),
            ValueField(name="label", dtype=int),
        )

        result = join(left, right, how="outer")

        assert result.table["left_index"].isna().tolist() == [False, False, True]
        assert result.table["right_index"].isna().tolist() == [True, False, False]
        assert result.table["left_index"].iloc[:2].tolist() == ["a", "b"]
        assert result.table["right_index"].iloc[1:].tolist() == ["b", "c"]

    def test_combines_input_sources(self):
        schema = Schema(fields=(IndexField(name="item_id"), ValueField(name="x", dtype=int)))
        left = make_from_dataframe(
            pd.DataFrame({"x": [1]}, index=["a"]),
            schema,
            source_info=DatasetSourceInfo(
                source_id="left",
                source_uri="memory://left",
                source_type="dataframe",
            ),
        )
        right = make_from_dataframe(
            pd.DataFrame({"x": [2]}, index=["a"]),
            schema,
            source_info=DatasetSourceInfo(
                source_id="right",
                source_uri="memory://right",
                source_type="dataframe",
            ),
        )

        result = join(left, right)

        assert tuple(source.source_id for source in result.sources) == ("left", "right")


class TestFieldEqualityJoin:
    def test_on_shorthand_uses_field_equality_strategy(self):
        left = _dataset(
            pd.DataFrame({"group": [1, 2, 2]}, index=["l1", "l2", "l3"]),
            ValueField(name="group", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"group": [2, 2]}, index=["r1", "r2"]),
            ValueField(name="group", dtype=int),
        )

        result = join(left, right, on="group")

        assert result.table.to_dict("list") == {
            "left_index": ["l2", "l2", "l3", "l3"],
            "right_index": ["r1", "r2", "r1", "r2"],
        }

    def test_explicit_field_equality_strategy(self):
        left = _dataset(
            pd.DataFrame({"group": [1, 2]}, index=["l1", "l2"]),
            ValueField(name="group", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"group": [2, 3]}, index=["r1", "r2"]),
            ValueField(name="group", dtype=int),
        )

        result = join(left, right, strategy=FieldEqualityJoin(on=("group",)))

        assert result.table.to_dict("list") == {
            "left_index": ["l2"],
            "right_index": ["r1"],
        }

    def test_field_equality_left_join_keeps_unmatched_left_rows(self):
        left = _dataset(
            pd.DataFrame({"group": [1, 2]}, index=["l1", "l2"]),
            ValueField(name="group", dtype=int),
        )
        right = _dataset(
            pd.DataFrame({"group": [2]}, index=["r1"]),
            ValueField(name="group", dtype=int),
        )

        result = join(left, right, on="group", how="left")

        assert result.table["left_index"].tolist() == ["l1", "l2"]
        assert result.table["right_index"].isna().tolist() == [True, False]
        assert result.table["right_index"].iloc[1] == "r1"


class TestDimensionJoin:
    def test_matches_half_open_interval_overlap_across_all_dimensions(self):
        left = _dataset(
            pd.DataFrame(
                {
                    "extent": [
                        DimensionedSlice(dims={"y": slice(0, 10), "x": slice(0, 10)}),
                        DimensionedSlice(dims={"y": slice(10, 20), "x": slice(0, 10)}),
                    ]
                },
                index=["l1", "l2"],
            ),
            DimensionedSliceField(name="extent"),
        )
        right = _dataset(
            pd.DataFrame(
                {
                    "bbox": [
                        DimensionedSlice(dims={"y": slice(5, 15), "x": slice(2, 4)}),
                        DimensionedSlice(dims={"y": slice(20, 25), "x": slice(0, 5)}),
                    ]
                },
                index=["r1", "r2"],
            ),
            DimensionedSliceField(name="bbox"),
        )

        result = join(
            left,
            right,
            strategy=DimensionJoin(
                left_field="extent",
                right_field="bbox",
                dimensions=("y", "x"),
            ),
        )

        assert result.table.to_dict("list") == {
            "left_index": ["l1", "l2"],
            "right_index": ["r1", "r1"],
        }

    def test_scopes_tile_local_coordinates_by_equality_fields(self):
        left = _dataset(
            pd.DataFrame(
                {
                    "source_image_id": ["tile_a", "tile_b"],
                    "extent": [
                        DimensionedSlice(dims={"y": slice(0, 10), "x": slice(0, 10)}),
                        DimensionedSlice(dims={"y": slice(0, 10), "x": slice(0, 10)}),
                    ],
                },
                index=["l1", "l2"],
            ),
            ValueField(name="source_image_id", dtype=str),
            DimensionedSliceField(name="extent"),
        )
        right = _dataset(
            pd.DataFrame(
                {
                    "source_image_id": ["tile_b"],
                    "bbox": [
                        DimensionedSlice(dims={"y": slice(1, 2), "x": slice(1, 2)}),
                    ],
                },
                index=["r1"],
            ),
            ValueField(name="source_image_id", dtype=str),
            DimensionedSliceField(name="bbox"),
        )

        result = join(
            left,
            right,
            strategy=DimensionJoin(
                left_field="extent",
                right_field="bbox",
                dimensions=("y", "x"),
                on="source_image_id",
            ),
        )

        assert result.table.to_dict("list") == {
            "left_index": ["l2"],
            "right_index": ["r1"],
        }

    def test_left_join_keeps_unmatched_slices(self):
        left = _dataset(
            pd.DataFrame(
                {
                    "extent": [
                        DimensionedSlice(dims={"y": slice(0, 2), "x": slice(0, 2)}),
                        DimensionedSlice(dims={"y": slice(2, 4), "x": slice(0, 2)}),
                    ]
                },
                index=["l1", "l2"],
            ),
            DimensionedSliceField(name="extent"),
        )
        right = _dataset(
            pd.DataFrame(
                {
                    "bbox": [
                        DimensionedSlice(dims={"y": slice(0, 1), "x": slice(0, 1)}),
                    ]
                },
                index=["r1"],
            ),
            DimensionedSliceField(name="bbox"),
        )

        result = join(
            left,
            right,
            strategy=DimensionJoin(
                how="left",
                left_field="extent",
                right_field="bbox",
                dimensions=("y", "x"),
            ),
        )

        assert result.table["left_index"].tolist() == ["l1", "l2"]
        assert result.table["right_index"].isna().tolist() == [False, True]


class TestJoinErrors:
    def test_rejects_invalid_how(self):
        left = _dataset(pd.DataFrame({"x": [1]}, index=["a"]), ValueField(name="x", dtype=int))
        right = _dataset(pd.DataFrame({"x": [1]}, index=["a"]), ValueField(name="x", dtype=int))

        with pytest.raises(ValueError, match="how"):
            join(left, right, how="sideways")

    def test_requires_two_datasets(self):
        left = _dataset(pd.DataFrame({"x": [1]}, index=["a"]), ValueField(name="x", dtype=int))

        with pytest.raises(ValueError, match="exactly two"):
            join(left)

    def test_rejects_missing_on_field(self):
        left = _dataset(pd.DataFrame({"x": [1]}, index=["a"]), ValueField(name="x", dtype=int))
        right = _dataset(pd.DataFrame({"x": [1]}, index=["a"]), ValueField(name="x", dtype=int))

        with pytest.raises(ValueError, match="not present"):
            join(left, right, on="missing")

    def test_rejects_index_field_as_on_field_for_mvp(self):
        left = _dataset(pd.DataFrame({"x": [1]}, index=["a"]), ValueField(name="x", dtype=int))
        right = _dataset(pd.DataFrame({"x": [1]}, index=["a"]), ValueField(name="x", dtype=int))

        with pytest.raises(ValueError, match="table columns"):
            join(left, right, on="item_id")

    def test_rejects_strategy_and_on_together(self):
        left = _dataset(pd.DataFrame({"x": [1]}, index=["a"]), ValueField(name="x", dtype=int))
        right = _dataset(pd.DataFrame({"x": [1]}, index=["a"]), ValueField(name="x", dtype=int))

        with pytest.raises(ValueError, match="either 'strategy' or 'on'"):
            join(left, right, strategy=FieldEqualityJoin(on=("x",)), on="x")

    def test_dimension_join_requires_slice_fields(self):
        left = _dataset(pd.DataFrame({"x": [1]}, index=["a"]), ValueField(name="x", dtype=int))
        right = _dataset(pd.DataFrame({"x": [1]}, index=["a"]), ValueField(name="x", dtype=int))

        with pytest.raises(TypeError, match="DimensionedSliceField"):
            join(
                left,
                right,
                strategy=DimensionJoin(
                    left_field="x",
                    right_field="x",
                    dimensions=("x",),
                ),
            )

    def test_join_plan_has_empty_couplings(self):
        left = _dataset(pd.DataFrame({"x": [1]}, index=["a"]), ValueField(name="x", dtype=int))
        right = _dataset(pd.DataFrame({"x": [1]}, index=["a"]), ValueField(name="x", dtype=int))

        result = join(left, right)

        assert result.couplings.couplings == ()
