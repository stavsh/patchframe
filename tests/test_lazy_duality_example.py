"""Guards the runnable lazy<->eager duality example against regressions."""

from __future__ import annotations

import pandas as pd

from examples.lazy_eager_duality_usage import (
    _labels,
    _scores,
    deferred_pipeline,
    eager_pipeline,
    same_level_deferral,
)


def test_deferred_bundle_pipeline_matches_eager():
    scores, labels = _scores(), _labels()

    eager = eager_pipeline(scores, labels)
    deferred = deferred_pipeline(scores, labels)

    assert eager.schema.names() == deferred.schema.names()
    pd.testing.assert_frame_equal(eager.table, deferred.table)


def test_same_level_deferral_materializes_clips():
    result = same_level_deferral()

    assert result.table.at["a", "clip"].dims["t"] == slice(0, 4)
    assert result.table.at["b", "clip"].dims["t"] == slice(5, 9)
