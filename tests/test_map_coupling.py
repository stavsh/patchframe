"""MapCoupling: the deferred per-row function over field values (map_fields' coupling).

MapCoupling is the per-row sibling of ApplyOperator — it carries a CallSpec whose
operator is a plain function, reads input *column values* per row, and writes
``fn(*values)``. These tests pin the coupling mechanics directly (compute,
apply_row, field refs, pickle, topo participation); the operator-level behaviour
lives in test_map_fields.py.
"""

from __future__ import annotations

import pickle

import pandas as pd

import patchframe as pf
from patchframe.dataset.coupling_engine import CouplingEngine
from patchframe.dataset.couplings import CallSpec, CouplingSet, MapCoupling


def _add(a, b):
    return a + b


def _inc(x):
    return x + 1


def _ds(**columns) -> pf.Dataset:
    frame = pd.DataFrame(columns, index=["r0", "r1"])
    fields = (pf.IndexField(name="item_id"), *(pf.ValueField(name=n) for n in columns))
    return pf.make_from_dataframe(frame, pf.Schema(fields=fields))


def test_compute_applies_fn_per_row():
    ds = _ds(a=[1, 2], b=[10, 20])
    coupling = MapCoupling(inputs=("a", "b"), output="c", call=CallSpec(operator=_add))

    result = coupling.compute(ds.state)

    assert result.tolist() == [11, 22]
    assert list(result.index) == ["r0", "r1"]


def test_apply_row_writes_output():
    ds = _ds(a=[1, 2], b=[10, 20])
    coupling = MapCoupling(inputs=("a", "b"), output="c", call=CallSpec(operator=_add))

    out = coupling.apply_row({"a": 5, "b": 7}, ds.state)

    assert out["c"] == 12


def test_input_output_fields_and_fn_view():
    coupling = MapCoupling(inputs=("a", "b"), output="c", call=CallSpec(operator=_add))

    assert coupling.input_fields() == ("a", "b")
    assert coupling.output_field() == "c"
    assert coupling.fn is _add


def test_pickle_round_trips_with_module_level_fn():
    coupling = MapCoupling(inputs=("a", "b"), output="c", call=CallSpec(operator=_add))

    restored = pickle.loads(pickle.dumps(coupling))

    assert restored.fn is _add
    assert restored.input_fields() == ("a", "b")
    assert restored.output_field() == "c"


def test_engine_orders_map_after_its_upstream_producer():
    # c = inc(b), b = inc(a): declared out of order, the engine must still run the
    # producer of `b` before the coupling that reads `b`.
    ds = _ds(a=[1, 2], b=[None, None], c=[None, None])
    first = MapCoupling(inputs=("a",), output="b", call=CallSpec(operator=_inc))
    second = MapCoupling(inputs=("b",), output="c", call=CallSpec(operator=_inc))

    engine = CouplingEngine(schema=ds.schema, couplings=CouplingSet((second, first)))

    assert engine.order == (first, second)


def test_consume_runs_a_map_coupling_chain():
    ds = _ds(a=[1, 2], b=[None, None], c=[None, None])
    first = MapCoupling(inputs=("a",), output="b", call=CallSpec(operator=_inc))
    second = MapCoupling(inputs=("b",), output="c", call=CallSpec(operator=_inc))
    chained = ds.replace_state(couplings=CouplingSet((first, second)))

    result = pf.consume(chained, "c")

    assert result.table["b"].tolist() == [2, 3]
    assert result.table["c"].tolist() == [3, 4]
