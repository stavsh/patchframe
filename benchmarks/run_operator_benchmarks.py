"""Run manual patchframe operator benchmarks.

Example:
    python -m benchmarks.run_operator_benchmarks --rows 100000 --ops merge_inner --profile
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import json
import platform
import pstats
import sys
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.factories import (
    dimension_bindings,
    make_collision_pair,
    make_index_pair,
    make_multidim_dataset,
)
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.field_composition import ColumnCollisionStrategy
from patchframe.dataset.state import DatasetState
from patchframe.ops.builtin.bind_dimensions import bind_dimensions
from patchframe.ops.builtin.bind_slice import bind_slice
from patchframe.ops.builtin.concat import concat_columns, concat_rows
from patchframe.ops.builtin.consume import consume
from patchframe.ops.builtin.join import FieldEqualityJoin, IndexJoin, join
from patchframe.ops.builtin.merge import merge

OperationRunner = Callable[[], Dataset]
PhaseRunner = Callable[[], tuple[Dataset, dict[str, float]]]

DEFAULT_OPS = (
    "concat_rows",
    "concat_columns",
    "join_index",
    "join_field",
    "merge_inner",
    "merge_outer",
    "merge_collision_update_missing",
    "consume_bind_dimensions",
    "consume_chained_dimensions",
    "consume_bind_slice",
)


@dataclass(frozen=True, slots=True)
class OperationCase:
    """Prepared benchmark operation with setup excluded from timing."""

    name: str
    run: OperationRunner
    phase_run: PhaseRunner | None = None


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ops = _parse_ops(args.ops)
    results = []
    for op_name in selected_ops:
        for repeat in range(args.repeat):
            case = _make_case(op_name, args)
            result = _run_case(case, args, output_dir=output_dir, repeat=repeat)
            results.append(result)
            print(
                f"{op_name}[{repeat}]: {result['elapsed_sec']:.6f}s "
                f"shape={tuple(result['result_shape'])}"
            )

    payload = {
        "metadata": _metadata(),
        "config": _config(args),
        "results": results,
    }
    path = output_dir / f"operator_bench_{_timestamp()}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--value-cols", type=int, default=16)
    parser.add_argument("--string-cols", type=int, default=1)
    parser.add_argument("--group-mod", type=int, default=1024)
    parser.add_argument("--null-every", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--ops", default="all")
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-limit", type=int, default=15)
    parser.add_argument("--phase-profile", action="store_true")
    parser.add_argument("--tracemalloc", action="store_true")
    parser.add_argument("--no-data", action="store_true")
    parser.add_argument("--no-dimensions", action="store_true")
    return parser.parse_args(argv)


def _parse_ops(value: str) -> tuple[str, ...]:
    if value == "all":
        return DEFAULT_OPS
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(selected) - set(DEFAULT_OPS))
    if unknown:
        raise ValueError(f"Unknown benchmark operations: {unknown}")
    return selected


def _make_case(name: str, args: argparse.Namespace) -> OperationCase:
    rows = args.rows
    common = {
        "value_cols": args.value_cols,
        "string_cols": args.string_cols,
        "include_data": not args.no_data,
        "include_dimensions": not args.no_dimensions,
        "null_every": args.null_every,
    }

    if name == "concat_rows":
        pair = make_index_pair(rows, overlap="none", **common)
        return _composition_case(name, concat_rows.instance(), (pair.left, pair.right), {})

    if name == "concat_columns":
        pair = make_index_pair(rows, overlap="full", right_index_field="right_id", **common)
        collision = ColumnCollisionStrategy(mode="keep", side="left")
        return _composition_case(
            name,
            concat_columns.instance(),
            (pair.left, pair.right),
            {"collision": collision},
        )

    if name == "join_index":
        pair = make_index_pair(rows, overlap="half", **common)
        strategy = IndexJoin(how="inner")
        return _composition_case(
            name,
            join.instance(),
            (pair.left, pair.right),
            {"strategy": strategy},
        )

    if name == "join_field":
        pair = make_index_pair(rows, overlap="full", **common)
        strategy = FieldEqualityJoin(on=("group",), how="inner")
        return _composition_case(
            name,
            join.instance(),
            (pair.left, pair.right),
            {"strategy": strategy},
        )

    if name == "merge_inner":
        pair = make_index_pair(rows, overlap="full", **common)
        plan = join(pair.left, pair.right)
        collision = ColumnCollisionStrategy(mode="keep", side="left")
        return _composition_case(
            name,
            merge.instance(),
            (pair.left, pair.right, plan),
            {"collision": collision},
        )

    if name == "merge_outer":
        pair = make_index_pair(rows, overlap="half", **common)
        plan = join(pair.left, pair.right, how="outer")
        collision = ColumnCollisionStrategy(mode="keep", side="left")
        return _composition_case(
            name,
            merge.instance(),
            (pair.left, pair.right, plan),
            {"collision": collision},
        )

    if name == "merge_collision_update_missing":
        pair = make_collision_pair(rows, **common)
        plan = join(pair.left, pair.right)
        collision = ColumnCollisionStrategy(mode="update_missing", side="left")
        return _composition_case(
            name,
            merge.instance(),
            (pair.left, pair.right, plan),
            {"collision": collision},
        )

    if name == "consume_bind_dimensions":
        ds = _consume_dataset(rows, args)
        ds = bind_dimensions(ds, slice_field="slice", bindings=dimension_bindings())
        return _dataset_case(name, consume.instance(), ds, {"target": "slice"})

    if name == "consume_chained_dimensions":
        ds = _consume_dataset(rows, args)
        ds = bind_dimensions(
            ds,
            slice_field="slice",
            bindings={"time": dimension_bindings()["time"]},
        )
        ds = bind_dimensions(
            ds,
            slice_field="slice",
            bindings={
                name: fields
                for name, fields in dimension_bindings().items()
                if name != "time"
            },
        )
        return _dataset_case(name, consume.instance(), ds, {"target": "slice"})

    if name == "consume_bind_slice":
        ds = _consume_dataset(rows, args)
        ds = bind_dimensions(ds, slice_field="slice", bindings=dimension_bindings())
        ds = bind_slice(ds, slice_field="slice", data_field="data")
        target = ds.couplings.couplings[-1]
        return _dataset_case(name, consume.instance(), ds, {"target": target})

    raise ValueError(f"Unknown benchmark operation: {name}")


def _consume_dataset(rows: int, args: argparse.Namespace) -> Dataset:
    return make_multidim_dataset(
        rows,
        value_cols=args.value_cols,
        string_cols=args.string_cols,
        group_mod=args.group_mod,
        include_dimensions=True,
        include_data=True,
        null_every=args.null_every,
    )


def _composition_case(
    name: str,
    operator: Any,
    datasets: tuple[Dataset, ...],
    kwargs: dict[str, Any],
) -> OperationCase:
    return OperationCase(
        name=name,
        run=lambda: operator(*datasets, **kwargs),
        phase_run=lambda: _run_composition_phases(operator, datasets, kwargs),
    )


def _dataset_case(
    name: str,
    operator: Any,
    dataset: Dataset,
    kwargs: dict[str, Any],
) -> OperationCase:
    return OperationCase(
        name=name,
        run=lambda: operator(dataset, **kwargs),
        phase_run=lambda: _run_dataset_phases(operator, dataset, kwargs),
    )


def _run_case(
    case: OperationCase,
    args: argparse.Namespace,
    *,
    output_dir: Path,
    repeat: int,
) -> dict[str, Any]:
    gc.collect()
    if args.tracemalloc:
        tracemalloc.start()

    profile = cProfile.Profile() if args.profile else None
    start = time.perf_counter()
    if profile is not None:
        profile.enable()
    result, phase_timings = _execute(case, phase_profile=args.phase_profile)
    if profile is not None:
        profile.disable()
    elapsed = time.perf_counter() - start

    peak_memory = None
    if args.tracemalloc:
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    profile_path = None
    profile_summary = None
    if profile is not None:
        profile_path = output_dir / f"{case.name}_{repeat}_{_timestamp()}.prof"
        profile.dump_stats(str(profile_path))
        profile_summary = {
            "top_overall": _profile_entries(profile, None, args.profile_limit),
            "top_patchframe": _profile_entries(profile, "patchframe", args.profile_limit),
            "top_pandas": _profile_entries(profile, "pandas", args.profile_limit),
        }

    return {
        "operation": case.name,
        "repeat": repeat,
        "elapsed_sec": elapsed,
        "phase_timings": phase_timings,
        "result_shape": list(result.table.shape),
        "result_index_size": len(result.table.index),
        "schema_fields": len(result.schema.fields),
        "peak_tracemalloc_bytes": peak_memory,
        "profile_path": str(profile_path) if profile_path is not None else None,
        "profile_summary": profile_summary,
    }


def _execute(case: OperationCase, *, phase_profile: bool) -> tuple[Dataset, dict[str, float]]:
    if phase_profile and case.phase_run is not None:
        return case.phase_run()
    return case.run(), {}


def _run_composition_phases(
    operator: Any,
    datasets: tuple[Dataset, ...],
    kwargs: dict[str, Any],
) -> tuple[Dataset, dict[str, float]]:
    states = tuple(dataset.state for dataset in datasets)
    timings: dict[str, float] = {}

    schema, timings["apply_schema"] = _timed(lambda: operator.apply_schema(*states, **kwargs))
    table, timings["apply_table"] = _timed(lambda: operator.apply_table(*states, **kwargs))
    couplings, timings["apply_couplings"] = _timed(
        lambda: operator.apply_couplings(*states, **kwargs)
    )
    sources, timings["combine_sources"] = _timed(lambda: operator.combine_sources(*states))
    state, timings["state_build"] = _timed(
        lambda: DatasetState(
            schema=schema,
            table=table,
            couplings=couplings,
            sources=sources,
        )
    )
    return Dataset(state=state), timings


def _run_dataset_phases(
    operator: Any,
    dataset: Dataset,
    kwargs: dict[str, Any],
) -> tuple[Dataset, dict[str, float]]:
    state = dataset.state
    timings: dict[str, float] = {}
    table, timings["apply_table"] = _timed(lambda: operator.apply_table(state, **kwargs))
    result_state, timings["state_build"] = _timed(
        lambda: DatasetState(
            schema=state.schema,
            table=table,
            couplings=state.couplings,
            sources=state.sources,
            source_descriptors=state.source_descriptors,
            assets=state.assets,
            views=state.views,
        )
    )
    return Dataset(state=result_state, source_manager=dataset.source_manager), timings


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def _profile_entries(
    profile: cProfile.Profile,
    package_filter: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    stats = pstats.Stats(profile)
    entries = []
    for (filename, line, func_name), stat in stats.stats.items():
        primitive_calls, total_calls, total_time, cumulative_time, _ = stat
        normalized = filename.replace("\\", "/")
        if package_filter is not None and package_filter not in normalized:
            continue
        entries.append(
            {
                "function": f"{normalized}:{line}:{func_name}",
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "total_time_sec": total_time,
                "cumulative_time_sec": cumulative_time,
            }
        )
    entries.sort(key=lambda item: item["cumulative_time_sec"], reverse=True)
    return entries[:limit]


def _metadata() -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")


if __name__ == "__main__":
    raise SystemExit(main())
