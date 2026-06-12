"""Audio feature extraction pipeline.

Turns each downloaded MP3 into a fixed-length numeric "fingerprint" and
stores it as JSON in the `features` table.

Usage:
    python -m src.features

Idempotent: sounds that already have features are skipped. Corrupt or
unreadable files are logged and skipped, never crash the run.

--- Why these features? (DSP crash course) ---------------------------------
Audio is just a long array of amplitude samples — far too long and too
raw to feed a classical ML model directly. The standard trick is:

1. Slice the signal into short overlapping frames (~23 ms each).
2. Compute descriptors per frame (so each descriptor becomes a time series).
3. Summarize each time series with statistics (mean + std here), giving a
   fixed-length vector regardless of how long the original clip was.

The mean captures "what the sound is like on average"; the std captures
"how much it changes over time" (a steady ambience hum vs. a sharp impact
have very different stds even if their means are similar).
"""

import json
import logging
from datetime import datetime, timezone

import librosa
import numpy as np

from src.config import N_MFCC, PROJECT_ROOT, SAMPLE_RATE
from src.db import get_connection, init_db

log = logging.getLogger("features")

# Frames quieter than (peak - 30 dB) count as silence and get trimmed.
# Trimming matters because dead air at the start/end of an upload would
# drag every "mean" feature toward silence and add noise to the labels'
# real acoustic differences. 30 dB is librosa's sensible default.
TRIM_DB = 30


def extract_feature_vector(filepath) -> dict[str, float]:
    """Compute the full feature dict for one audio file.

    Returns a flat {name: value} dict — named keys (rather than a bare
    list) make the JSON self-documenting and let us build a labeled
    DataFrame for EDA/training without guessing column order.
    """
    # mono=True mixes channels down to one signal; sr=SAMPLE_RATE resamples
    # everything to a common rate so features are comparable across files.
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TRIM_DB)

    if len(y) < 2048:  # shorter than one analysis frame -> nothing to measure
        raise ValueError("audio is empty or near-empty after silence trimming")

    features: dict[str, float] = {}

    # MFCCs (Mel-Frequency Cepstral Coefficients) describe the *spectral
    # envelope* — the broad shape of energy across frequencies, on the mel
    # scale, which spaces frequencies the way human hearing does. They are
    # the workhorse "timbre" feature: metal clangs, footsteps, and beeps
    # all carve very different envelope shapes.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)  # (13, n_frames)
    for i in range(N_MFCC):
        features[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))

    # Spectral centroid: the "center of mass" of the spectrum, in Hz.
    # Perceptually this is brightness — UI beeps sit high, explosions low.
    # Spectral bandwidth: how spread out energy is around that center —
    # noisy/broadband sounds (rain, gunshots) vs. tonal ones (beeps).
    # Spectral rolloff: the frequency below which 85% of energy lives —
    # another brightness/high-frequency-content measure.
    # Zero-crossing rate: how often the waveform crosses zero — high for
    # noisy/percussive content, low for smooth low-pitched sounds.
    # RMS energy: loudness over time — its std separates one sharp burst
    # (impact) from sustained level (ambience).
    framewise = {
        "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr),
        "spectral_rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr),
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(y),
        "rms": librosa.feature.rms(y=y),
    }
    for name, values in framewise.items():
        features[f"{name}_mean"] = float(np.mean(values))
        features[f"{name}_std"] = float(np.std(values))

    # Duration after trimming (seconds). Categories differ systematically
    # here (UI blips ~1s, ambience clips much longer), so it is a real
    # signal — though we should stay aware the model may lean on it.
    features["duration"] = float(len(y) / sr)

    return features


def sounds_missing_features(conn) -> list:
    """Rows from `sounds` that have no feature vector yet (the idempotency
    check — re-runs only process what's new or previously failed)."""
    return conn.execute(
        """SELECT s.id, s.filepath FROM sounds s
           LEFT JOIN features f ON f.sound_id = s.id
           WHERE f.sound_id IS NULL
           ORDER BY s.id"""
    ).fetchall()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_db()

    with get_connection() as conn:
        todo = sounds_missing_features(conn)
        log.info("%d sounds need feature extraction", len(todo))

        done = failed = 0
        for row in todo:
            path = PROJECT_ROOT / row["filepath"]
            try:
                features = extract_feature_vector(path)
            except Exception as exc:
                # Corrupt download, unsupported encoding, or empty audio:
                # log it and move on. The LEFT JOIN means we'll retry these
                # on the next run, and they'll fail again harmlessly.
                log.warning("skipping sound id=%s (%s): %s", row["id"], path.name, exc)
                failed += 1
                continue

            conn.execute(
                "INSERT OR REPLACE INTO features (sound_id, feature_json, extracted_at) "
                "VALUES (?, ?, ?)",
                (
                    row["id"],
                    json.dumps(features),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
            conn.commit()  # per-file commit: an interrupted run keeps its progress
            done += 1
            if done % 100 == 0:
                log.info("%d/%d extracted", done, len(todo))

    log.info("Done — %d extracted, %d skipped/failed.", done, failed)


if __name__ == "__main__":
    main()
