"""Sanity check for the AudioDNA environment.

Run after `pip install -r requirements.txt`:

    python check_setup.py

Verifies (1) all dependencies import, (2) librosa can load audio and
compute MFCCs, (3) the database schema can be created, and (4) the
Freesound API key is readable from .env. Exits non-zero on failure so
it can double as a CI smoke test.
"""

import sys


def check(label: str, fn) -> bool:
    """Run one check, print a pass/fail line, and never crash the script."""
    try:
        detail = fn()
        print(f"  [OK]   {label}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as exc:
        print(f"  [FAIL] {label} — {exc}")
        return False


def check_imports():
    import librosa, matplotlib, numpy, pandas, requests, seaborn  # noqa: F401
    import sklearn, soundfile, streamlit  # noqa: F401
    return "all dependencies import"


def check_librosa_mfcc():
    import librosa

    # librosa ships small example recordings (downloaded and cached on
    # first use) — handy for testing DSP code before we have real data.
    path = librosa.example("trumpet")
    # sr=22050 resamples to our project rate; mono=True averages stereo
    # channels into one signal, matching how we'll process every file.
    y, sr = librosa.load(path, sr=22050, mono=True)
    # MFCCs summarize the spectral envelope (the "shape" of the sound's
    # frequency content) per short time frame -> matrix (n_mfcc, n_frames).
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    assert mfcc.shape[0] == 13, f"expected 13 MFCC rows, got {mfcc.shape}"
    return f"loaded {len(y)/sr:.1f}s of audio, MFCC shape {mfcc.shape}"


def check_database():
    from src.db import get_connection, init_db

    init_db()
    with get_connection() as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    missing = {"sounds", "features"} - tables
    assert not missing, f"missing tables: {missing}"
    return "schema OK (tables: sounds, features)"


def check_api_key():
    from src.config import get_api_key

    key = get_api_key()  # raises with instructions if unset
    # Show just enough to confirm it's the right key without printing it.
    return f"FREESOUND_API_KEY found (ends in ...{key[-4:]})"


def main() -> int:
    print("AudioDNA setup check")
    print(f"  Python {sys.version.split()[0]}")
    results = [
        check("dependency imports", check_imports),
        check("librosa load + MFCC", check_librosa_mfcc),
        check("database schema", check_database),
        check("Freesound API key", check_api_key),
    ]
    if all(results):
        print("All checks passed — ready for Phase 2 (data collection).")
        return 0
    print("Some checks failed — fix the issues above before continuing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
