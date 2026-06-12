"""Package a small demo dataset + trained model for cloud deployment.

Produces demo_bundle.zip (gitignored) containing, with paths relative to
the project root:

    data/audiodna.db          <- pruned copy: demo sounds only
    data/raw/<cat>/<id>.mp3   <- the demo sounds' audio
    models/audiodna_model.joblib

Only CC0 (public domain) sounds are included, so the bundle can be
redistributed as a GitHub Release asset with no attribution obligations —
though the app still shows attribution for every sound, as good practice.

Usage (from the project root):
    python -m scripts.make_demo_bundle
"""

import shutil
import sqlite3
import zipfile

from src.config import CATEGORIES, DB_PATH, PROJECT_ROOT
from src.train import MODEL_PATH

N_PER_CATEGORY = 12
OUT_ZIP = PROJECT_ROOT / "demo_bundle.zip"


def pick_demo_ids(conn) -> list[int]:
    """Up to N CC0-licensed, feature-complete sounds per category."""
    ids: list[int] = []
    for cat in CATEGORIES:
        rows = conn.execute(
            """SELECT s.id FROM sounds s
               JOIN features f ON f.sound_id = s.id
               WHERE s.category = ? AND s.license LIKE '%publicdomain/zero%'
               ORDER BY s.id LIMIT ?""",
            (cat, N_PER_CATEGORY),
        ).fetchall()
        if len(rows) < N_PER_CATEGORY:
            print(f"warning: only {len(rows)} CC0 sounds available for '{cat}'")
        ids.extend(r[0] for r in rows)
    return ids


def main() -> None:
    staging = PROJECT_ROOT / "demo_staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "data").mkdir(parents=True)
    (staging / "models").mkdir()

    # Prune a COPY of the database down to the demo sounds. Copy-then-delete
    # (rather than rebuilding) guarantees the demo schema is identical to
    # the real one. VACUUM rewrites the file so the deleted rows' space is
    # actually reclaimed.
    demo_db = staging / "data" / "audiodna.db"
    shutil.copy2(DB_PATH, demo_db)
    conn = sqlite3.connect(demo_db)
    keep = pick_demo_ids(conn)
    placeholders = ",".join("?" * len(keep))
    conn.execute(f"DELETE FROM features WHERE sound_id NOT IN ({placeholders})", keep)
    conn.execute(f"DELETE FROM sounds WHERE id NOT IN ({placeholders})", keep)
    conn.commit()
    rows = conn.execute("SELECT category, filepath FROM sounds").fetchall()
    conn.execute("VACUUM")
    conn.close()

    for _, rel_path in rows:
        src = PROJECT_ROOT / rel_path
        dest = staging / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    shutil.copy2(MODEL_PATH, staging / "models" / MODEL_PATH.name)

    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging))
    shutil.rmtree(staging)

    size_mb = OUT_ZIP.stat().st_size / 1e6
    print(f"wrote {OUT_ZIP.name}: {len(rows)} sounds, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
