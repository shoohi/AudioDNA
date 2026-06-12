"""Central configuration for AudioDNA.

Every other module imports paths and constants from here, so there is a
single place to change them. Nothing in this file does heavy work at
import time (no DB connections, no audio loading) — just constants.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# config.py lives in src/, so the project root is one directory up.
# Using __file__ (instead of os.getcwd()) means paths work no matter
# which directory you run scripts from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "audiodna.db"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
# load_dotenv() reads key=value pairs from a `.env` file in the project
# root into environment variables. The `.env` file is gitignored, so the
# key never ends up in version control.
load_dotenv(PROJECT_ROOT / ".env")


def get_api_key() -> str:
    """Return the Freesound API key, or raise with a helpful message.

    A function (rather than a module-level constant) so that importing
    config never crashes — only code that actually needs the key fails
    when it is missing.
    """
    key = os.getenv("FREESOUND_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FREESOUND_API_KEY is not set. Copy .env.example to .env and "
            "add your key from https://freesound.org/apiv2/apply/"
        )
    return key


# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
CATEGORIES = ["impact", "footsteps", "ambience", "ui", "explosion", "weapon"]

# How many sounds to collect per category (Phase 2).
SOUNDS_PER_CATEGORY = 150

# Only collect short clips: game SFX are typically brief, and short files
# keep download size and feature-extraction time manageable.
MAX_DURATION_SECONDS = 10.0

# ---------------------------------------------------------------------------
# Audio / DSP constants
# ---------------------------------------------------------------------------
# 22050 Hz is half of CD quality (44100 Hz). By the Nyquist theorem it
# still captures frequencies up to ~11 kHz — plenty for distinguishing
# sound-effect categories — while halving memory and compute. It is also
# librosa's default, so examples and docs line up with our numbers.
SAMPLE_RATE = 22050

# Number of MFCC coefficients to extract per frame (Phase 3).
# 13 is the classic choice from speech recognition: the first ~13
# coefficients capture the broad spectral envelope ("timbre") while
# higher ones mostly add noise for classification tasks.
N_MFCC = 13
