"""Tests for the public convenience imports."""

from __future__ import annotations

import patchframe as pf


def test_top_level_exports_core_dataset_objects():
    assert pf.Dataset
    assert pf.DatasetState
    assert pf.Schema
    assert pf.IndexField
    assert pf.ValueField
    assert pf.ArrayDataSource
    assert pf.AxisWindow
    assert pf.DataAccessor
    assert pf.ResolvedSlice
    assert pf.Dimensions


def test_top_level_exports_user_facing_operators():
    assert pf.make_from_dataframe
    assert pf.where
    assert pf.concat
    assert pf.join
    assert pf.merge
    assert pf.bind_dimensions
    assert pf.bind_slice
    assert pf.make_dimensional_plan
    assert pf.PlanOperator
