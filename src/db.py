"""SQLite connection helper and schema creation.

Usage from other modules:

    from src.db import get_connection, init_db

Or from the command line, to (re)create the schema:

    python -m src.db
"""

import sqlite3

from src.config import DATA_DIR, DB_PATH

# Schema notes (one deliberate change from the original spec):
#
# * `features.sound_id` is the PRIMARY KEY, not just a foreign key.
#   Each sound has exactly one feature vector, and making sound_id the
#   primary key enforces that at the database level. It also makes the
#   extraction script naturally idempotent: re-running it can use
#   "INSERT OR REPLACE" and can cheaply ask "which sounds lack features?"
#   without ever creating duplicate rows.
#
# * Timestamps are stored as TEXT in ISO-8601 format (SQLite has no
#   native datetime type; ISO strings sort correctly and are readable).
SCHEMA = """
CREATE TABLE IF NOT EXISTS sounds (
    id            INTEGER PRIMARY KEY,
    freesound_id  INTEGER UNIQUE NOT NULL,
    name          TEXT,
    category      TEXT NOT NULL,
    license       TEXT,
    username      TEXT,
    duration      REAL,
    filepath      TEXT,
    downloaded_at TEXT
);

CREATE TABLE IF NOT EXISTS features (
    sound_id     INTEGER PRIMARY KEY REFERENCES sounds(id),
    feature_json TEXT NOT NULL,
    extracted_at TEXT
);

-- Most queries filter or group by category (per-category collection,
-- class-balance checks, the library explorer tab), so index it.
CREATE INDEX IF NOT EXISTS idx_sounds_category ON sounds(category);
"""


def get_connection() -> sqlite3.Connection:
    """Open a connection to the project database.

    - Row factory lets us access columns by name (row["category"])
      instead of by position, which keeps calling code readable.
    - SQLite does NOT enforce foreign keys unless you ask per-connection,
      so we turn that on here once for everyone.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables if they do not exist. Safe to call repeatedly."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
