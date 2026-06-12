"""Similarity ranking tests.

A tiny hand-built index (5 synthetic sounds, 3 features) replaces the real
DB + model, so these run on a fresh clone. Sounds 1+2 are designed to be
near-duplicates, 3+4 likewise, and 5 is unlike everything — making the
correct ranking unambiguous.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.similarity import SimilarityIndex

FEATURE_COLS = ["f1", "f2", "f3"]
RAW = pd.DataFrame(
    [
        [0.0, 0.0, 10.0],   # id 1 ─┐ near-duplicates
        [0.1, -0.2, 9.0],   # id 2 ─┘
        [5.0, 5.0, 0.0],    # id 3 ─┐ near-duplicates
        [4.8, 5.2, 0.3],    # id 4 ─┘
        [-5.0, 5.0, 5.0],   # id 5: unlike everything
    ],
    columns=FEATURE_COLS,
)


@pytest.fixture
def index() -> SimilarityIndex:
    scaler = StandardScaler().fit(RAW)
    X = scaler.transform(RAW)
    unit = X / np.linalg.norm(X, axis=1, keepdims=True)
    meta = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": [f"sound{i}" for i in range(1, 6)],
        "category": ["a", "a", "b", "b", "c"],
    })
    bundle = {"scaler": scaler, "feature_cols": FEATURE_COLS}
    return SimilarityIndex(meta, unit, bundle)


def query(row: int) -> dict:
    return RAW.iloc[row].to_dict()


def test_self_query_without_exclusion_returns_self_first(index):
    results = index.search(query(0), k=1)
    assert results.loc[0, "id"] == 1
    assert results.loc[0, "similarity"] == pytest.approx(1.0)


def test_excluding_self_promotes_the_near_duplicate(index):
    results = index.search(query(0), k=2, exclude_sound_id=1)
    assert 1 not in results["id"].values
    assert results.loc[0, "id"] == 2  # the designed near-duplicate


def test_results_are_sorted_and_k_is_respected(index):
    results = index.search(query(2), k=3)
    assert len(results) == 3
    sims = results["similarity"].to_numpy()
    assert all(sims[:-1] >= sims[1:]), "similarities must be descending"


def test_designed_pairs_rank_above_the_outlier(index):
    # Query sound 3: its pair (4) must beat the outlier (5).
    results = index.search(query(2), k=4, exclude_sound_id=3)
    ids = list(results["id"])
    assert ids.index(4) < ids.index(5)
