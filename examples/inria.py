"""Inria Aerial Image Labeling example built around lazy GeoTIFF patches.

Download and extract the dataset from:

    https://project.inria.fr/aerialimagelabeling/

The expected directory layout is:

    AerialImageDataset/
        train/
            images/
                austin1.tif
                ...
            gt/
                austin1.tif
                ...
        test/
            images/
                bellingham1.tif
                ...

The conventional usage path stays short:

    images = make_inria("AerialImageDataset", split="train")
    plan = make_inria_patch_plan(images, patch_size=512)
    patches = bind_inria_patches(images, plan)
    row = patches[0]
    image = row["image"]
    mask = row["mask"]

The table stores lazy accessors. Row access reads only the selected GeoTIFF
window for both the RGB image and aligned building mask. The example stays in
raster pixel coordinates. Geometry-aware matching becomes necessary when
labels arrive as an independent spatial source rather than an aligned mask.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

import patchframe as pf

InriaSplit = Literal["train", "test"]
RasterSize = int | tuple[int, int]

IMAGE_FIELD = "image"
MASK_FIELD = "mask"
PATCH_FIELD = "patch"
IMAGE_EXTENT_FIELD = "image_extent"
SOURCE_IMAGE_ID_FIELD = "source_image_id"
CITY_FIELD = "city"
SPLIT_FIELD = "split"
IMAGE_PATH_FIELD = "image_path"
MASK_PATH_FIELD = "mask_path"
IMAGE_HEIGHT_FIELD = "image_height"
IMAGE_WIDTH_FIELD = "image_width"
CHANNEL_COUNT_FIELD = "channel_count"
METERS_PER_PIXEL_FIELD = "meters_per_pixel"

_IMAGE_ASSET_ID = 0
_MASK_ASSET_ID = 1
_METERS_PER_PIXEL = 0.3


class InriaAerialDataSource(pf.ArrayDataSource):
    """DataSource that reads Inria RGB imagery and building masks on demand."""

    source_type = "inria_aerial"
    thread_safe: bool = True
    fork_safe: bool = True
    config_fields = ("root", "split")
    supports_partial_read = True

    def __init__(
        self,
        *,
        root: str,
        split: InriaSplit = "train",
        dimensions: pf.Dimensions | None = None,
        source_id: str | None = None,
    ) -> None:
        _validate_split(split)
        super().__init__(
            dimensions=dimensions or _raster_dimensions(),
            source_id=source_id,
            root=os.path.abspath(root),
            split=split,
        )

    def read_full(self, item_id: Any, accessor: pf.DataAccessor) -> np.ndarray:
        return _read_tiff_full(
            self._path_for(item_id, accessor.asset_id),
            asset_id=accessor.asset_id,
        )

    def read_partial(
        self,
        item_id: Any,
        resolved_slice: pf.ResolvedSlice,
        accessor: pf.DataAccessor,
    ) -> np.ndarray:
        return _read_tiff_window(
            self._path_for(item_id, accessor.asset_id),
            y_slice=_slice_for_axis(resolved_slice, "y"),
            x_slice=_slice_for_axis(resolved_slice, "x"),
            asset_id=accessor.asset_id,
        )

    def inspect(self, accessor: pf.DataAccessor) -> dict[str, Any]:
        path = self._path_for(accessor.item_id, accessor.asset_id)
        height, width, channel_count, dtype = _inspect_tiff(path)
        is_mask = accessor.asset_id == _MASK_ASSET_ID
        return {
            "item_id": accessor.item_id,
            "asset": _asset_name(accessor.asset_id),
            "path": path,
            "shape": (height, width) if is_mask else (height, width, channel_count),
            "dtype": "bool" if is_mask else dtype,
            "extent": _extent_for_shape((height, width)),
            "dimensioned_slice": accessor.dimensioned_slice,
        }

    def extent_for(self, item_id: Any) -> pf.DimensionedSlice:
        height, width, _, _ = _inspect_tiff(self._path_for(item_id, _IMAGE_ASSET_ID))
        return _extent_for_shape((height, width))

    def _path_for(self, item_id: Any, asset_id: int) -> str:
        if asset_id == _IMAGE_ASSET_ID:
            directory = "images"
        elif asset_id == _MASK_ASSET_ID and self.split == "train":
            directory = "gt"
        elif asset_id == _MASK_ASSET_ID:
            raise ValueError("Inria test imagery does not include public ground-truth masks.")
        else:
            raise ValueError(f"Unknown Inria asset id: {asset_id!r}.")
        return os.path.join(self.root, self.split, directory, f"{item_id}.tif")


class make_inria(pf.CreationOperator):
    """Build a lazy dataset over an extracted Inria dataset directory."""

    split = pf.Parameter(default="train")

    def make_source(
        self,
        root: str | Path,
        *,
        split: InriaSplit | object = pf.MISSING,
        **_: Any,
    ) -> InriaAerialDataSource:
        return InriaAerialDataSource(
            root=str(root),
            split=self.resolve_param("split", split),
        )

    def generate_source_info(
        self,
        root: str | Path,
        *,
        split: InriaSplit | object = pf.MISSING,
        **_: Any,
    ) -> pf.DatasetSourceInfo:
        resolved_split = self.resolve_param("split", split)
        return pf.DatasetSourceInfo(
            source_uri=Path(root).resolve().as_uri(),
            source_type="inria_aerial",
            source_name=f"Inria Aerial Image Labeling ({resolved_split})",
            metadata={
                "split": resolved_split,
                "meters_per_pixel": _METERS_PER_PIXEL,
                "dataset_homepage": "https://project.inria.fr/aerialimagelabeling/",
            },
        )

    def build(
        self,
        root: str | Path,
        *,
        split: InriaSplit | object = pf.MISSING,
        source_desc_id: int | None = None,
        source_manager: Any = None,
        **_: Any,
    ) -> pf.DatasetState:
        resolved_split = self.resolve_param("split", split)
        _validate_split(resolved_split)
        root_path = Path(root).resolve()
        image_paths = tuple(sorted((root_path / resolved_split / "images").glob("*.tif")))
        if not image_paths:
            raise ValueError(
                f"No Inria GeoTIFF images found under {root_path / resolved_split / 'images'}."
            )

        ids = tuple(path.stem for path in image_paths)
        raster_info = tuple(_inspect_tiff(str(path)) for path in image_paths)
        heights = tuple(info[0] for info in raster_info)
        widths = tuple(info[1] for info in raster_info)
        channel_counts = tuple(info[2] for info in raster_info)
        if resolved_split == "train":
            _validate_training_masks(root_path, ids, heights=heights, widths=widths)
        dimensions = _raster_dimensions()
        index = pd.Index(ids, name="image_id")

        table = pd.DataFrame(index=index)
        table[SOURCE_IMAGE_ID_FIELD] = pd.Series(ids, index=index, dtype="string")
        table[CITY_FIELD] = pd.Series(
            (_city_for(image_id) for image_id in ids),
            index=index,
            dtype="string",
        )
        table[SPLIT_FIELD] = pd.Series(resolved_split, index=index, dtype="string")
        table[IMAGE_PATH_FIELD] = pd.Series(
            (str(path) for path in image_paths),
            index=index,
            dtype="string",
        )
        if resolved_split == "train":
            table[MASK_PATH_FIELD] = pd.Series(
                (str(root_path / resolved_split / "gt" / f"{image_id}.tif") for image_id in ids),
                index=index,
                dtype="string",
            )
        table[IMAGE_HEIGHT_FIELD] = pd.Series(heights, index=index, dtype="Int64")
        table[IMAGE_WIDTH_FIELD] = pd.Series(widths, index=index, dtype="Int64")
        table[CHANNEL_COUNT_FIELD] = pd.Series(channel_counts, index=index, dtype="Int64")
        table[METERS_PER_PIXEL_FIELD] = pd.Series(
            _METERS_PER_PIXEL,
            index=index,
            dtype="Float64",
        )
        table[IMAGE_EXTENT_FIELD] = pd.Series(
            pf.DimensionedSliceArray.from_columns(
                dimensions=dimensions.dims,
                selector_columns=(
                    (np.zeros(len(ids), dtype=np.int64), np.asarray(heights, dtype=np.int64)),
                    (np.zeros(len(ids), dtype=np.int64), np.asarray(widths, dtype=np.int64)),
                ),
            ),
            index=index,
        )
        table[PATCH_FIELD] = pd.Series(
            pf.DimensionedSliceArray(mask=np.ones(len(ids), dtype=bool)),
            index=index,
        )
        table[IMAGE_FIELD] = [
            pf.DataAccessor(
                source_desc_id=source_desc_id,
                item_id=image_id,
                asset_id=_IMAGE_ASSET_ID,
                manager_hint=source_manager,
            )
            for image_id in ids
        ]
        if resolved_split == "train":
            table[MASK_FIELD] = [
                pf.DataAccessor(
                    source_desc_id=source_desc_id,
                    item_id=image_id,
                    asset_id=_MASK_ASSET_ID,
                    manager_hint=source_manager,
                )
                for image_id in ids
            ]

        schema = pf.Schema(
            fields=(
                pf.IndexField(name="image_id"),
                pf.DimensionField.from_dim(
                    pf.CategoricalDimension(name="source_image_id"),
                    SOURCE_IMAGE_ID_FIELD,
                    dtype=str,
                ),
                pf.ValueField(name=CITY_FIELD, dtype=str),
                pf.ValueField(name=SPLIT_FIELD, dtype=str),
                pf.ValueField(name=IMAGE_PATH_FIELD, dtype=str),
                *(
                    (pf.ValueField(name=MASK_PATH_FIELD, dtype=str),)
                    if resolved_split == "train"
                    else ()
                ),
                pf.ValueField(name=IMAGE_HEIGHT_FIELD, dtype=int),
                pf.ValueField(name=IMAGE_WIDTH_FIELD, dtype=int),
                pf.ValueField(name=CHANNEL_COUNT_FIELD, dtype=int),
                pf.ValueField(name=METERS_PER_PIXEL_FIELD, dtype=float),
                pf.DimensionedSliceField(name=IMAGE_EXTENT_FIELD, nullable=False),
                pf.DimensionedSliceField(name=PATCH_FIELD),
                pf.DataField(name=IMAGE_FIELD),
                *((pf.DataField(name=MASK_FIELD),) if resolved_split == "train" else ()),
            )
        )
        return pf.DatasetState(schema=schema, table=table)


def make_inria_patch_plan(
    images: pf.Dataset,
    *,
    patch_size: RasterSize,
    stride: RasterSize | None = None,
    include_partial: bool = False,
    extent_field: str = IMAGE_EXTENT_FIELD,
    patch_field: str = PATCH_FIELD,
) -> pf.Dataset:
    """Create a regular pixel-space patch plan for Inria tiles."""

    patch_height, patch_width = _normalize_size("patch_size", patch_size)
    stride_height, stride_width = _normalize_size(
        "stride",
        patch_size if stride is None else stride,
    )
    return pf.window_expansion_plan(
        images,
        field=extent_field,
        windows={
            "y": pf.AxisWindow(patch_height, stride_height, include_partial=include_partial),
            "x": pf.AxisWindow(patch_width, stride_width, include_partial=include_partial),
        },
        slice_field=patch_field,
    )


def bind_inria_patches(
    images: pf.Dataset,
    plan: pf.Dataset | None = None,
    *,
    patch_size: RasterSize = 512,
    stride: RasterSize | None = None,
    include_partial: bool = False,
    extent_field: str = IMAGE_EXTENT_FIELD,
    patch_field: str = PATCH_FIELD,
    image_field: str = IMAGE_FIELD,
    mask_field: str = MASK_FIELD,
    materialize_patches: bool = True,
) -> pf.Dataset:
    """Apply a patch plan and bind lazy image and mask reads to row access."""

    patch_plan = (
        plan
        if plan is not None
        else make_inria_patch_plan(
            images,
            patch_size=patch_size,
            stride=stride,
            include_partial=include_partial,
            extent_field=extent_field,
            patch_field=patch_field,
        )
    )
    data_fields = tuple(
        name for name in (image_field, mask_field) if images.schema.has(name)
    )
    patches = pf.explode(images, patch_plan)
    for data_field in data_fields:
        patches = pf.bind_slice(patches, slice_field=patch_field, data_field=data_field)
    if materialize_patches:
        patches = pf.bind_materialize(patches, data_fields)
    return patches


def plot_patch(
    patches: pf.Dataset,
    item_id: Any,
    *,
    image_field: str = IMAGE_FIELD,
    mask_field: str = MASK_FIELD,
    axes: Any = None,
) -> Any:
    """Plot one lazily loaded Inria image patch and optional building mask."""

    plt = _pyplot()
    row = patches[item_id]
    has_mask = mask_field in row
    if axes is None:
        _, axes = plt.subplots(1, 2 if has_mask else 1, squeeze=False)
        axes = axes[0]
    axes[0].imshow(np.asarray(row[image_field]))
    axes[0].set_title(f"{item_id}: RGB")
    axes[0].set_axis_off()
    if has_mask:
        axes[1].imshow(np.asarray(row[mask_field]), cmap="gray")
        axes[1].set_title(f"{item_id}: buildings")
        axes[1].set_axis_off()
    return axes


def _raster_dimensions() -> pf.Dimensions:
    return pf.Dimensions((pf.IndexDimension(name="y"), pf.IndexDimension(name="x")))


def _extent_for_shape(shape: tuple[int, ...]) -> pf.DimensionedSlice:
    return pf.DimensionedSlice(dims={"y": slice(0, shape[0]), "x": slice(0, shape[1])})


def _slice_for_axis(resolved_slice: pf.ResolvedSlice, name: str) -> slice:
    value = resolved_slice.get(name, slice(None))
    if not isinstance(value, slice):
        raise TypeError(f"Inria GeoTIFF reads require slice selectors; got {name}={value!r}.")
    return value


def _asset_name(asset_id: int) -> str:
    if asset_id == _IMAGE_ASSET_ID:
        return IMAGE_FIELD
    if asset_id == _MASK_ASSET_ID:
        return MASK_FIELD
    raise ValueError(f"Unknown Inria asset id: {asset_id!r}.")


def _read_tiff_full(path: str, *, asset_id: int) -> np.ndarray:
    rasterio = _rasterio()
    with rasterio.open(path) as src:
        values = src.read()
    return _format_asset(values, asset_id=asset_id)


def _read_tiff_window(
    path: str,
    *,
    y_slice: slice,
    x_slice: slice,
    asset_id: int,
) -> np.ndarray:
    rasterio = _rasterio()
    with rasterio.open(path) as src:
        window = rasterio.windows.Window.from_slices(
            y_slice,
            x_slice,
            height=src.height,
            width=src.width,
        )
        values = src.read(window=window)
    return _format_asset(values, asset_id=asset_id)


def _inspect_tiff(path: str) -> tuple[int, int, int, str]:
    rasterio = _rasterio()
    with rasterio.open(path) as src:
        return src.height, src.width, src.count, src.dtypes[0]


def _format_asset(values: np.ndarray, *, asset_id: int) -> np.ndarray:
    if asset_id == _IMAGE_ASSET_ID:
        return np.moveaxis(values, 0, -1)
    if asset_id == _MASK_ASSET_ID:
        return values[0] > 0
    raise ValueError(f"Unknown Inria asset id: {asset_id!r}.")


def _city_for(image_id: str) -> str:
    match = re.fullmatch(r"(.+?)(\d+)", image_id)
    if match is None:
        raise ValueError(f"Cannot infer Inria city from image id {image_id!r}.")
    return match.group(1)


def _validate_training_masks(
    root: Path,
    image_ids: tuple[str, ...],
    *,
    heights: tuple[int, ...],
    widths: tuple[int, ...],
) -> None:
    mask_dir = root / "train" / "gt"
    missing = [image_id for image_id in image_ids if not (mask_dir / f"{image_id}.tif").is_file()]
    if missing:
        raise ValueError(f"Inria training masks are missing for image ids: {missing}.")
    mismatched = []
    for image_id, height, width in zip(image_ids, heights, widths, strict=True):
        mask_height, mask_width, _, _ = _inspect_tiff(str(mask_dir / f"{image_id}.tif"))
        if (mask_height, mask_width) != (height, width):
            mismatched.append(image_id)
    if mismatched:
        raise ValueError(f"Inria training mask dimensions do not match images: {mismatched}.")


def _normalize_size(name: str, value: RasterSize) -> tuple[int, int]:
    result = (value, value) if isinstance(value, int) else tuple(value)
    if len(result) != 2 or any(size <= 0 for size in result):
        raise ValueError(f"{name} must contain two positive integers.")
    return result


def _validate_split(split: str) -> None:
    if split not in {"train", "test"}:
        raise ValueError("split must be either 'train' or 'test'.")


def _rasterio() -> Any:
    try:
        import rasterio
    except ImportError as err:
        raise ImportError("rasterio is required for the Inria aerial imagery example.") from err
    return rasterio


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("matplotlib is required for Inria visualization.") from err
    return plt
