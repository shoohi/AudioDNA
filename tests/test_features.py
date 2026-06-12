"""Feature extraction tests.

These use *synthetic* signals (sine waves, white noise) rather than real
files, so they run on a fresh clone with no downloaded data — and because
we know exactly what the features SHOULD say about a pure tone vs. noise,
they verify the features mean what we claim they mean.
"""

import numpy as np
import pytest

from src.config import N_MFCC, SAMPLE_RATE
from src.features import features_from_audio

EXPECTED_DIM = 2 * N_MFCC + 2 * 5 + 1  # mfcc mean+std, 5 spectral mean+std, duration


def sine(freq: float, seconds: float = 1.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_vector_shape_and_finiteness():
    feats = features_from_audio(sine(440), SAMPLE_RATE)
    assert len(feats) == EXPECTED_DIM == 37
    assert all(np.isfinite(v) for v in feats.values())
    # spot-check the naming convention the rest of the project relies on
    for key in ("mfcc1_mean", "mfcc13_std", "spectral_centroid_mean",
                "zero_crossing_rate_std", "rms_mean", "duration"):
        assert key in feats


def test_duration_matches_signal_length():
    feats = features_from_audio(sine(440, seconds=2.0), SAMPLE_RATE)
    assert feats["duration"] == pytest.approx(2.0, abs=0.01)


def test_centroid_tracks_pitch():
    # A 4000 Hz tone is "brighter" than a 200 Hz tone: its spectral
    # centroid and zero-crossing rate must both be higher.
    low = features_from_audio(sine(200), SAMPLE_RATE)
    high = features_from_audio(sine(4000), SAMPLE_RATE)
    assert high["spectral_centroid_mean"] > low["spectral_centroid_mean"]
    assert high["zero_crossing_rate_mean"] > low["zero_crossing_rate_mean"]


def test_noise_is_broader_band_than_tone():
    # White noise spreads energy across all frequencies; a sine puts it
    # all in one place. Bandwidth must reflect that.
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(SAMPLE_RATE).astype(np.float32)
    tone_f = features_from_audio(sine(440), SAMPLE_RATE)
    noise_f = features_from_audio(noise, SAMPLE_RATE)
    assert noise_f["spectral_bandwidth_mean"] > tone_f["spectral_bandwidth_mean"]
    assert noise_f["zero_crossing_rate_mean"] > tone_f["zero_crossing_rate_mean"]


def test_near_empty_audio_raises():
    with pytest.raises(ValueError):
        features_from_audio(np.zeros(100, dtype=np.float32), SAMPLE_RATE)
