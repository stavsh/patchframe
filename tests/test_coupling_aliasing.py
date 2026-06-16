"""Reject in-place read/write aliasing in a pending coupling set (fail-loud).

The forked-handle hazard: two handles forked off one field where the second op
rewrites that field in place while the first still reads it —
``h2 = op(ds.field("f")); op(ds.field("f"), out="f")``. Because couplings
reference fields by name with no versioning, the topo-sort silently runs the
in-place write before the reader, feeding the reader the after-value. We reject
the ambiguous set rather than miscompute (the consume-is-literal fork makes the
failure mode worse: collecting one handle advances the shared cursor under the
other). These tests pin the engine-level invariant and that it does *not* flag
the legitimate patterns the engine deliberately supports.
"""

from __future__ import annotations

import pytest

import patchframe as pf
from patchframe.dataset.coupling_engine import CouplingEngine
from patchframe.dataset.couplings import CallSpec, CouplingSet, MapCoupling


def _id(x):
    return x


def _ds(**columns) -> pf.Dataset:
    import pandas as pd

    frame = pd.DataFrame(columns, index=["r0", "r1"])
    fields = (pf.IndexField(name="item_id"), *(pf.ValueField(name=n) for n in columns))
    return pf.make_from_dataframe(frame, pf.Schema(fields=fields))


def _map(inputs, output) -> MapCoupling:
    return MapCoupling(inputs=tuple(inputs), output=output, call=CallSpec(operator=_id))


def test_inplace_write_after_external_reader_is_rejected():
    # reader recorded first (reads f1 → out2), then an in-place rewrite of f1.
    schema = _ds(f1=[1, 2], out2=[None, None]).schema
    reader = _map(("f1",), "out2")
    inplace = _map(("f1",), "f1")  # reads and writes f1 → ambiguous for `reader`

    with pytest.raises(ValueError, match="transformed in place"):
        CouplingEngine(schema=schema, couplings=CouplingSet((reader, inplace)))


def test_inplace_write_before_reader_is_allowed():
    # The materialize → map shape: the in-place transform is recorded first, so a
    # later reader is meant to see the result. Not a hazard.
    schema = _ds(f1=[1, 2], out2=[None, None]).schema
    inplace = _map(("f1",), "f1")
    reader = _map(("f1",), "out2")

    engine = CouplingEngine(schema=schema, couplings=CouplingSet((inplace, reader)))

    assert engine.order == (inplace, reader)


def test_out_of_order_clean_chain_is_allowed():
    # c = f(b); b = g(a) declared out of order — b has a single producer that does
    # not read b, so it is unambiguous and the engine reorders it.
    schema = _ds(a=[1, 2], b=[None, None], c=[None, None]).schema
    second = _map(("b",), "c")
    first = _map(("a",), "b")

    engine = CouplingEngine(schema=schema, couplings=CouplingSet((second, first)))

    assert engine.order == (first, second)


def test_inplace_chain_mates_are_not_external_readers():
    # Two in-place transforms on f1 form a same-output chain; the earlier one
    # reading f1 is a chain mate, not an external reader. No false positive.
    schema = _ds(f1=[1, 2]).schema
    first = _map(("f1",), "f1")
    second = _map(("f1",), "f1")

    engine = CouplingEngine(schema=schema, couplings=CouplingSet((first, second)))

    assert engine.order == (first, second)


def test_consume_surfaces_the_hazard():
    # The public path: an ambiguous pending set raises when consume builds the engine.
    ds = _ds(f1=[1, 2], out2=[None, None])
    aliased = ds.replace_state(
        couplings=CouplingSet((_map(("f1",), "out2"), _map(("f1",), "f1")))
    )

    with pytest.raises(ValueError, match="ambiguous"):
        pf.consume(aliased, "out2")
