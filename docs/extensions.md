# Extensions

Reference + work index for **extending patchframe** — new field types, dimensions,
match predicates, window/fiber specs, reducing operators, sources, operators,
aspect handlers. Small on purpose: it maps the surfaces that exist and names the
ones that do not. For operator internals see `operator-authoring.md`; for sources,
CLAUDE.md "Source Authoring".

## Current problems

- **Scattered.** The hooks live across `data/`, `dataset/`, `ops/`, `testing/`;
  there is no single map, so an author cannot find them. This doc is step one.
- **Inconsistent mechanism.** Three shapes coexist with no stated rule: *subclass-
  only* (`Dimension`, `MatchPredicate`, `ReducingOperator`), *subclass +
  `register_*`* (`Field` wants `register_field_policy` + `register_field_exit`),
  and *subclass + contract test* (`DataSource` + `assert_source_contract`). Which
  to use when is tribal knowledge.
- **Aspirational protocols.** Some surfaces are named in design but unbuilt:
  `WindowSpec` (only `AxisWindow` exists — the dense path is effectively
  hardcoded), `FiberSpec` (unbuilt), `register_predicate` (deferred),
  `assert_operator_contract` / `register_aspect_check` (planned). Extending those
  has no seam yet.
- **Uneven verification.** Predicates and sources have offline contract tests;
  fields, dimensions, and operators do not — an extension can silently break an
  invariant.

## Extension surfaces (code references)

Exported under `patchframe.data` / `.dataset` / `.ops` (sources via
`patchframe.testing` for the test helper).

| extend | base / mechanism | contract test | module |
|---|---|---|---|
| field type | subclass `Field` + `register_field_policy` + `register_field_exit` | — (gap) | `dataset/fields.py`, `dataset/field_composition.py` |
| dimension type | subclass `Dimension` (`spec` / `to_index` / `comparable_with`) | — (gap) | `data/dimensions.py` |
| match predicate | subclass `MatchPredicate` (`matches` / `correspond` / `stage` / `applies_to`) | `assert_predicate_contract` | `data/predicates.py` |
| window spec | `AxisWindow` only — **no `WindowSpec` base yet** | — | `data/windows.py` |
| reducing op | subclass `ReducingOperator` (impl `apply`; reserve `bulk_kernel`) | — (gap) | `ops/builtin/reduce.py` |
| source | subclass `DataSource` / `ArrayDataSource` (`read_full` / `read_partial`) | `assert_source_contract` | `data/source.py`, `data/array_source.py` |
| operator | declare `OperatorSignature` + a `TransitionPlan` | planned (`assert_operator_contract`) | `operator-authoring.md` |
| aspect handler | `register_aspect_handler(aspect, mode, fn)` | — | `ops/dispatch.py` |

## What needs to be done

- **Decide the uniform shape** (subclass vs registry) and make the per-X protocols
  *rhyme*: each surface = *semantics + (optional) vectorized strategy + a contract
  test*, as `MatchPredicate` and reducing ops already are. Bring the rest into line.
- **Build the missing protocols:** `WindowSpec` / `WindowExpansion` to replace the
  `AxisWindow`-only path (before geometry/sparse planners); `FiberSpec` (the
  `partition` / `chunk` fibering sibling); `register_predicate` (when an extension
  needs polymorphic discovery).
- **Close the verification gaps:** `assert_operator_contract` + per-aspect checkers
  (`register_aspect_check`), and contract tests for field and dimension types, so
  every surface is mechanically checkable (`design/aspect-transition.md`).
- **Driving consumers** (force the uniform shape): geometry (`GeometryDimension`,
  geometry window/predicate specs), sparse points (`SparseDimensions`,
  bounding-volume sources), Dask (executor) — all extension-owned
  (`design/design-constraints.md` §6, §11; `design/roadmap.md` "Extension surfaces").
