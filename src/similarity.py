"""Similarity search over the sound library.

"Find me sounds like this one": represent every sound as its standardized
37-dim feature vector and rank by cosine similarity.

Why cosine (vs. plain Euclidean distance)? Cosine compares the *direction*
of two vectors and ignores their overall magnitude. After standardization
the direction encodes the sound's acoustic profile (bright vs. dark,
steady vs. bursty, ...) while magnitude mostly reflects how *extreme* a
sound is overall. Two quiet and loud recordings of the same kind of sound
point the same way and score as similar — which matches what a human
means by "sounds like". On standardized features the two metrics are
related anyway (for unit-length vectors, Euclidean distance is a monotone
function of cosine), so this choice matters at the margins; cosine is the
conventional pick for nearest-neighbor search in feature space.

Usage as a module (the Streamlit app does this):

    index = SimilarityIndex.load()
    results = index.search(feature_dict, k=5)

CLI demo — pick a random library sound and show its neighbors:

    python -m src.similarity [--k 5] [--sound-id 123]
"""

import argparse
import json
import random

import joblib
import numpy as np
import pandas as pd

from src.db import get_connection
from src.train import MODEL_PATH


class SimilarityIndex:
    """In-memory index: metadata DataFrame + matrix of unit-length vectors."""

    def __init__(self, meta: pd.DataFrame, unit_vectors: np.ndarray, bundle: dict):
        self.meta = meta
        self.unit_vectors = unit_vectors
        self.bundle = bundle

    @classmethod
    def load(cls) -> "SimilarityIndex":
        """Build the index from the DB + the trained model bundle.

        We reuse the bundle's scaler (fit during training) rather than
        fitting a new one here, so a query goes through *exactly* the same
        preprocessing as the library — and as the classifier. One pipeline,
        no train/serve skew.
        """
        bundle = joblib.load(MODEL_PATH)
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT s.id, s.freesound_id, s.name, s.category, s.license,
                          s.username, s.duration, s.filepath, f.feature_json
                   FROM sounds s JOIN features f ON f.sound_id = s.id
                   ORDER BY s.id"""
            ).fetchall()

        meta = pd.DataFrame([dict(r) for r in rows]).drop(columns="feature_json")
        feats = pd.json_normalize([json.loads(r["feature_json"]) for r in rows])
        # Column order must match what the scaler/model were fit with.
        X = bundle["scaler"].transform(feats[bundle["feature_cols"]])

        # Pre-normalize every vector to unit length. Cosine similarity is
        # then just a dot product, so searching the whole library becomes
        # one matrix-vector multiply.
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # guard: an all-zero vector stays zero
        return cls(meta, X / norms, bundle)

    def vectorize(self, feature_dict: dict[str, float]) -> np.ndarray:
        """Standardize + unit-normalize one raw feature dict (e.g. from an
        uploaded file) into query space."""
        # One-row DataFrame (not a bare list) so column names line up with
        # what the scaler was fit on — sklearn checks and warns otherwise.
        row = pd.DataFrame([feature_dict])[self.bundle["feature_cols"]]
        v = self.bundle["scaler"].transform(row)[0]
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def search(
        self,
        feature_dict: dict[str, float],
        k: int = 5,
        exclude_sound_id: int | None = None,
    ) -> pd.DataFrame:
        """Top-k most similar library sounds, with metadata + similarity.

        exclude_sound_id: when the query *is* a library sound, exclude it —
        otherwise it would trivially return itself with similarity 1.0.
        """
        q = self.vectorize(feature_dict)
        sims = self.unit_vectors @ q  # cosine similarity to every sound

        results = self.meta.assign(similarity=sims)
        if exclude_sound_id is not None:
            results = results[results["id"] != exclude_sound_id]
        return results.nlargest(k, "similarity").reset_index(drop=True)


def _demo(k: int, sound_id: int | None) -> None:
    """Pick one library sound and print its nearest neighbors."""
    index = SimilarityIndex.load()

    with get_connection() as conn:
        if sound_id is None:
            row = conn.execute(
                """SELECT s.id, s.name, s.category, f.feature_json
                   FROM sounds s JOIN features f ON f.sound_id = s.id
                   ORDER BY RANDOM() LIMIT 1"""
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT s.id, s.name, s.category, f.feature_json
                   FROM sounds s JOIN features f ON f.sound_id = s.id
                   WHERE s.id = ?""",
                (sound_id,),
            ).fetchone()
        if row is None:
            raise SystemExit(f"no sound with id={sound_id} (or it lacks features)")

    print(f"query: [{row['category']}] {row['name']} (id={row['id']})\n")
    results = index.search(json.loads(row["feature_json"]), k=k,
                           exclude_sound_id=row["id"])
    for rank, r in results.iterrows():
        print(
            f"  {rank + 1}. sim={r['similarity']:.3f}  [{r['category']:<10}] "
            f"{r['name'][:55]}  (by {r['username']})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Similarity search demo.")
    parser.add_argument("--k", type=int, default=5, help="number of neighbors")
    parser.add_argument("--sound-id", type=int, default=None,
                        help="query sound id (random if omitted)")
    args = parser.parse_args()
    random.seed()  # explicit: demo is intentionally different each run
    _demo(args.k, args.sound_id)
