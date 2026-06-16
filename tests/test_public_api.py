"""Tests for the public convenience imports."""

from __future__ import annotations

import pytest

import patchframe as pf


def test_top_level_exports_core_dataset_objects():
    assert pf.Dataset
    assert pf.DatasetContext
    assert pf.DatasetState
    assert pf.Schema
    assert pf.IndexField
    assert pf.IndexIdentity
    assert pf.ForeignIndexField
    assert pf.FieldHandle
    assert pf.ValueField
    assert pf.ArrayDataSource
    assert pf.AxisWindow
    assert pf.DataAccessor
    assert pf.ResolvedSlice
    assert pf.Dimensions
    assert pf.get_active_dataset_context


def test_top_level_exports_user_facing_operators():
    assert pf.make_from_dataframe
    assert pf.make_plan
    assert pf.assign
    assert pf.where
    assert pf.concat
    assert pf.join
    assert pf.merge
    assert pf.explode
    assert pf.compose_slice
    assert pf.slice_data
    assert pf.link
    assert pf.map_fields
    assert pf.partition
    assert pf.dimension_join
    assert pf.match
    assert pf.implode
    assert pf.equals
    assert pf.overlap
    assert pf.assert_predicate_contract
    assert pf.window_expansion_plan
    assert pf.ContextEffect
    assert pf.OperatorCall
    assert pf.PlanOperator
    assert pf.register_aspect_handler


@pytest.mark.parametrize(
    "old_name, new_name",
    [
        ("bind_materialize", "materialize"),
        ("bind_slice", "slice_data"),
        ("bind_dimensions", "compose_slice"),
    ],
)
def test_renamed_bind_operators_keep_deprecated_aliases(old_name, new_name):
    # The bind_ prefix was dropped; the old names stay as deprecated aliases that
    # warn and resolve to the renamed operator.
    with pytest.warns(DeprecationWarning, match=old_name):
        aliased = getattr(pf, old_name)
    assert aliased is getattr(pf, new_name)
