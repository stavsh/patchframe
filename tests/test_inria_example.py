"""Tests for the Inria Aerial Image Labeling example workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import patchframe as pf
from examples.inria import (
    CITY_FIELD,
    IMAGE_EXTENT_FIELD,
    IMAGE_FIELD,
    IMAGE_HEIGHT_FIELD,
    IMAGE_WIDTH_FIELD,
    MASK_FIELD,
    MASK_PATH_FIELD,
    PATCH_FIELD,
    SPLIT_FIELD,
    InriaAerialDataSource,
    bind_inria_patches,
    make_inria,
    make_inria_patch_plan,
)
from patchframe.data.manager import reset_default_manager
from patchframe.testing import assert_source_contract


@pytest.fixture(autouse=True)
def fresh_manager():
    reset_default_manager()


@pytest.fixture
def inria_root(tmp_path, monkeypatch):
    arrays = {
        "train/images/austin1.tif": np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3),
        "train/gt/austin1.tif": np.arange(4 * 6, dtype=np.uint8).reshape(4, 6) % 2 == 0,
        "train/images/vienna2.tif": np.arange(5 * 5 * 3, dtype=np.uint8).reshape(5, 5, 3),
        "train/gt/vienna2.tif": np.arange(5 * 5, dtype=np.uint8).reshape(5, 5) % 3 == 0,
        "test/images/bellingham1.tif": np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3),
    }
    for relative_path in arrays:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def _array(path):
        relative_path = Path(path).relative_to(tmp_path).as_posix()
        return arrays[relative_path]

    def _inspect(path):
        array = _array(path)
        channels = 1 if array.ndim == 2 else array.shape[2]
        return array.shape[0], array.shape[1], channels, str(array.dtype)

    def _read_full(path, *, asset_id):
        return _array(path).copy()

    def _read_window(path, *, y_slice, x_slice, asset_id):
        return _array(path)[y_slice, x_slice].copy()

    monkeypatch.setattr("examples.inria._inspect_tiff", _inspect)
    monkeypatch.setattr("examples.inria._read_tiff_full", _read_full)
    monkeypatch.setattr("examples.inria._read_tiff_window", _read_window)
    return tmp_path, arrays


def test_make_inria_discovers_training_tiles_and_aligned_masks(inria_root):
    root, _ = inria_root

    images = make_inria(root, split="train")

    assert images.table.index.tolist() == ["austin1", "vienna2"]
    assert images.table.index.name == "image_id"
    assert images.table[CITY_FIELD].tolist() == ["austin", "vienna"]
    assert images.table[SPLIT_FIELD].tolist() == ["train", "train"]
    assert images.table[IMAGE_HEIGHT_FIELD].tolist() == [4, 5]
    assert images.table[IMAGE_WIDTH_FIELD].tolist() == [6, 5]
    assert images.schema.has(MASK_PATH_FIELD)
    assert images.schema.has(MASK_FIELD)
    assert isinstance(images.table[IMAGE_EXTENT_FIELD].array, pf.DimensionedSliceArray)
    assert images.table[IMAGE_EXTENT_FIELD].iloc[0].dims == {
        "y": slice(0, 4),
        "x": slice(0, 6),
    }
    assert isinstance(images.table[IMAGE_FIELD].iloc[0], pf.DataAccessor)
    assert isinstance(images.table[MASK_FIELD].iloc[0], pf.DataAccessor)


def test_make_inria_test_split_omits_unpublished_masks(inria_root):
    root, _ = inria_root

    images = make_inria(root, split="test")

    assert images.table.index.tolist() == ["bellingham1"]
    assert not images.schema.has(MASK_PATH_FIELD)
    assert not images.schema.has(MASK_FIELD)


def test_make_inria_rejects_missing_training_mask(inria_root):
    root, _ = inria_root
    (root / "train" / "gt" / "austin1.tif").unlink()

    with pytest.raises(ValueError, match="training masks are missing"):
        make_inria(root)


def test_make_inria_rejects_mismatched_training_mask_dimensions(inria_root):
    root, arrays = inria_root
    arrays["train/gt/austin1.tif"] = np.zeros((3, 6), dtype=bool)

    with pytest.raises(ValueError, match="mask dimensions do not match"):
        make_inria(root)


def test_inria_source_satisfies_partial_read_contract(inria_root):
    root, _ = inria_root
    source = InriaAerialDataSource(root=str(root))
    dim_slice = pf.DimensionedSlice(dims={"y": slice(1, 3), "x": slice(2, 5)})

    assert_source_contract(
        source,
        item_id="austin1",
        dim_slice=dim_slice,
        compare_partial=True,
    )


def test_filtered_plan_loads_only_selected_image_and_mask_windows(inria_root, monkeypatch):
    root, arrays = inria_root
    partial_reads = []
    original = __import__("examples.inria", fromlist=["_read_tiff_window"])._read_tiff_window

    def _read_window(path, *, y_slice, x_slice, asset_id):
        partial_reads.append((Path(path).relative_to(root).as_posix(), y_slice, x_slice))
        return original(path, y_slice=y_slice, x_slice=x_slice, asset_id=asset_id)

    monkeypatch.setattr("examples.inria._read_tiff_window", _read_window)

    images = make_inria(root)
    plan = make_inria_patch_plan(images, patch_size=(2, 3), stride=(2, 3))
    selected = pf.where(plan, plan.table.index < 1)
    patches = bind_inria_patches(images, selected)

    assert len(plan.table) == 6
    assert len(patches.table) == 1
    assert partial_reads == []

    row = patches[0]

    np.testing.assert_array_equal(row[IMAGE_FIELD], arrays["train/images/austin1.tif"][0:2, 0:3])
    np.testing.assert_array_equal(row[MASK_FIELD], arrays["train/gt/austin1.tif"][0:2, 0:3])
    assert row[PATCH_FIELD].dims == {"y": slice(0, 2), "x": slice(0, 3)}
    assert partial_reads == [
        ("train/images/austin1.tif", slice(0, 2), slice(0, 3)),
        ("train/gt/austin1.tif", slice(0, 2), slice(0, 3)),
    ]


def test_bind_inria_patches_can_leave_sliced_accessors_lazy(inria_root):
    root, arrays = inria_root
    images = make_inria(root)

    patches = bind_inria_patches(
        images,
        patch_size=(2, 3),
        stride=(2, 3),
        materialize_patches=False,
    )
    row = patches[1]

    assert isinstance(row[IMAGE_FIELD], pf.DataAccessor)
    assert isinstance(row[MASK_FIELD], pf.DataAccessor)
    assert row[IMAGE_FIELD].dimensioned_slice == pf.DimensionedSlice(
        dims={"y": slice(0, 2), "x": slice(3, 6)}
    )
    np.testing.assert_array_equal(
        row[IMAGE_FIELD].materialize(),
        arrays["train/images/austin1.tif"][0:2, 3:6],
    )


def test_bind_inria_patches_preserves_explicit_empty_plan(inria_root):
    root, _ = inria_root
    images = make_inria(root)
    plan = make_inria_patch_plan(images, patch_size=(2, 3))
    selected = pf.where(plan, plan.table.index < 0)

    patches = bind_inria_patches(images, selected)

    assert patches.table.empty


def test_patch_plan_can_include_partial_edge_tiles(inria_root):
    root, _ = inria_root
    images = pf.where(make_inria(root), lambda table: table.index == "vienna2")

    plan = make_inria_patch_plan(
        images,
        patch_size=(3, 3),
        stride=(3, 3),
        include_partial=True,
    )

    assert len(plan.table) == 4
    assert plan.table[PATCH_FIELD].iloc[-1].dims == {
        "y": slice(3, 5),
        "x": slice(3, 5),
    }
