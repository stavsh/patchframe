"""Runnable Inria Aerial Image Labeling patch example.

Download and extract the dataset from:

    https://project.inria.fr/aerialimagelabeling/

Then run:

    python examples/inria_usage.py --root path/to/AerialImageDataset --city austin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import patchframe as pf
from examples.inria import (
    CITY_FIELD,
    IMAGE_FIELD,
    MASK_FIELD,
    PATCH_FIELD,
    bind_inria_patches,
    make_inria,
    make_inria_mask_patch_plan,
    make_inria_patch_plan,
    plot_patch,
)


def main(
    root: str | Path,
    *,
    split: str,
    city: str | None,
    patch_size: int,
    stride: int,
    max_patches: int | None,
    min_component_pixels: int,
    include_partial: bool,
    plot: bool,
) -> pf.Dataset:
    images = make_inria(root, split=split)
    print(f"Loaded {len(images.table)} Inria {split} tiles")

    if city is not None:
        images = pf.where(images, images.table[CITY_FIELD] == city)
        print(f"Selected {len(images.table)} tiles for city {city!r}")

    if split == "train":
        plan = make_inria_mask_patch_plan(
            images,
            patch_size=patch_size,
            stride=stride,
            include_partial=include_partial,
            min_component_pixels=min_component_pixels,
        )
        print(f"Planned {len(plan.table)} mask-containing patches")
    else:
        plan = make_inria_patch_plan(
            images,
            patch_size=patch_size,
            stride=stride,
            include_partial=include_partial,
        )
        print(f"Planned {len(plan.table)} patches")

    if max_patches is not None:
        plan = pf.where(plan, plan.table.index < max_patches)
        print(f"Selected {len(plan.table)} patches")

    patches = bind_inria_patches(images, plan)
    if patches.table.empty:
        print("No patches matched.")
        return patches

    patch_id = patches.table.index[0]
    row = patches[patch_id]
    print(f"\nFirst patch id: {patch_id}")
    print(f"Pixel slice: {row[PATCH_FIELD].dims}")
    print(f"Loaded image shape: {row[IMAGE_FIELD].shape} dtype={row[IMAGE_FIELD].dtype}")
    if MASK_FIELD in row:
        mask = np.asarray(row[MASK_FIELD])
        print(f"Building coverage: {mask.mean():.2%}")

    if plot:
        import matplotlib.pyplot as plt

        plot_patch(patches, patch_id)
        plt.show()

    return patches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Extracted Inria AerialImageDataset root")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--city", help="Optional city prefix, for example 'austin' or 'vienna'")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-patches", type=int, default=4)
    parser.add_argument("--min-component-pixels", type=int, default=64)
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    main(
        args.root,
        split=args.split,
        city=args.city,
        patch_size=args.patch_size,
        stride=args.stride,
        max_patches=args.max_patches,
        min_component_pixels=args.min_component_pixels,
        include_partial=args.include_partial,
        plot=args.plot,
    )
