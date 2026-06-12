"""Freesound data collector.

Searches Freesound for game sound effects in each category and downloads
the *preview* MP3s (token auth is enough for previews; downloading
original files would require the full OAuth2 flow).

Usage:
    python -m src.collect --category impact --limit 150
    python -m src.collect --all

Idempotent: sounds already in the database are skipped, so re-running
after an interruption just picks up where it left off.
"""

import argparse
import logging
import time
from datetime import datetime, timezone

import requests

from src.config import (
    MAX_DURATION_SECONDS,
    RAW_AUDIO_DIR,
    SOUNDS_PER_CATEGORY,
    CATEGORIES,
    get_api_key,
)
from src.db import get_connection, init_db

log = logging.getLogger("collect")

# ---------------------------------------------------------------------------
# Search queries per category
# ---------------------------------------------------------------------------
# Our labels are "weak": a sound is labeled `impact` because it matched an
# impact-themed search, not because a human verified it. Using 2-3 varied
# queries per category diversifies the results (different uploaders,
# recording styles) so the classifier learns the category, not one query.
CATEGORY_QUERIES = {
    "impact": ["impact hit", "thud impact", "metal impact hit"],
    "footsteps": ["footsteps", "footsteps gravel", "footsteps wood walk"],
    "ambience": ["game ambience loop", "forest ambience", "city ambience"],
    "ui": ["ui click button", "menu select interface", "ui notification beep"],
    "explosion": ["explosion", "explosion blast", "distant explosion"],
    "weapon": ["gunshot", "sword swing weapon", "laser gun shot"],
}

API_BASE = "https://freesound.org/apiv2"

# Metadata fields we ask the API to return for each search result.
# Requesting only what we need keeps responses small.
FIELDS = "id,name,license,username,duration,previews"

# Freesound's token-auth rate limit is ~60 requests/minute. Sleeping
# ~1.1s between API calls keeps us safely under it without bursts.
THROTTLE_SECONDS = 1.1

# Max results per page the API allows is 150; fewer round-trips per query.
PAGE_SIZE = 150

REQUEST_TIMEOUT = 30  # seconds; never hang forever on a dead connection
MAX_RETRIES = 5


class FreesoundClient:
    """Thin wrapper around the Freesound REST API with throttling/backoff."""

    def __init__(self, api_key: str):
        self.session = requests.Session()
        # Token in a header (not the URL) so it never shows up in logs.
        self.session.headers["Authorization"] = f"Token {api_key}"

    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        """GET with throttling and exponential backoff on 429/5xx.

        429 means "too many requests" — the polite response is to wait
        (the server may tell us how long via the Retry-After header)
        and try again, doubling the wait each attempt.
        """
        for attempt in range(MAX_RETRIES):
            resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            time.sleep(THROTTLE_SECONDS)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = float(resp.headers.get("Retry-After", 2**attempt * 5))
                log.warning(
                    "HTTP %s from API, backing off %.0fs (attempt %d/%d)",
                    resp.status_code, wait, attempt + 1, MAX_RETRIES,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"Giving up on {url} after {MAX_RETRIES} retries")

    def search(self, query: str, page: int) -> dict:
        """One page of text-search results, filtered to short sounds."""
        params = {
            "query": query,
            # Freesound filter syntax: range filter on duration (seconds).
            # 0.1s lower bound excludes empty/broken uploads.
            "filter": f"duration:[0.1 TO {MAX_DURATION_SECONDS}]",
            "fields": FIELDS,
            "page_size": PAGE_SIZE,
            "page": page,
        }
        return self._get(f"{API_BASE}/search/text/", params).json()

    def download(self, url: str, dest_path) -> None:
        """Download a preview MP3 to dest_path (atomically via a temp name).

        Writing to '<name>.part' first and renaming on success means an
        interrupted run never leaves a half-written file that looks valid.
        """
        tmp = dest_path.with_suffix(dest_path.suffix + ".part")
        resp = self._get(url)
        tmp.write_bytes(resp.content)
        tmp.replace(dest_path)


# ---------------------------------------------------------------------------
# Collection logic
# ---------------------------------------------------------------------------

def existing_freesound_ids(conn) -> set[int]:
    """All freesound_ids already collected (any category)."""
    return {row["freesound_id"] for row in conn.execute("SELECT freesound_id FROM sounds")}


def count_in_category(conn, category: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS n FROM sounds WHERE category = ?", (category,)
    ).fetchone()["n"]


def collect_category(client: FreesoundClient, category: str, limit: int) -> int:
    """Download sounds for one category until `limit` rows exist in the DB.

    Returns the number of new sounds added in this run.
    """
    category_dir = RAW_AUDIO_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    with get_connection() as conn:
        seen = existing_freesound_ids(conn)
        have = count_in_category(conn, category)
        if have >= limit:
            log.info("[%s] already have %d/%d sounds — nothing to do", category, have, limit)
            return 0
        log.info("[%s] have %d/%d, collecting %d more", category, have, limit, limit - have)

        for query in CATEGORY_QUERIES[category]:
            page = 1
            while have < limit:
                results = client.search(query, page)
                if not results.get("results"):
                    break  # this query is exhausted, move to the next one

                for item in results["results"]:
                    if have >= limit:
                        break
                    fs_id = item["id"]
                    # Skip anything we already have — including sounds that
                    # matched an earlier category's query. This keeps every
                    # sound in exactly one category (no label conflicts).
                    if fs_id in seen:
                        continue
                    # preview-hq-mp3: ~128kbps MP3 the API provides for
                    # every sound. Good enough for feature extraction and
                    # far smaller than originals (which can be huge WAVs).
                    preview_url = item.get("previews", {}).get("preview-hq-mp3")
                    if not preview_url:
                        continue

                    dest = category_dir / f"{fs_id}.mp3"
                    try:
                        client.download(preview_url, dest)
                    except Exception as exc:
                        log.warning("[%s] download failed for id=%s: %s", category, fs_id, exc)
                        continue

                    conn.execute(
                        """INSERT INTO sounds
                           (freesound_id, name, category, license, username,
                            duration, filepath, downloaded_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            fs_id,
                            item.get("name"),
                            category,
                            item.get("license"),
                            item.get("username"),
                            item.get("duration"),
                            # Store the path relative to the project root so
                            # the DB stays valid if the project folder moves.
                            str(dest.relative_to(RAW_AUDIO_DIR.parent.parent)),
                            datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        ),
                    )
                    conn.commit()  # commit per sound: a crash loses nothing
                    seen.add(fs_id)
                    have += 1
                    added += 1
                    if have % 25 == 0:
                        log.info("[%s] %d/%d collected", category, have, limit)

                if results.get("next") is None:
                    break  # no more pages for this query
                page += 1
            if have >= limit:
                break

        if have < limit:
            log.warning(
                "[%s] only reached %d/%d — queries exhausted. "
                "Consider adding query terms to CATEGORY_QUERIES.",
                category, have, limit,
            )
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect sounds from Freesound.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--category", choices=CATEGORIES, help="collect one category")
    group.add_argument("--all", action="store_true", help="collect every category")
    parser.add_argument(
        "--limit", type=int, default=SOUNDS_PER_CATEGORY,
        help=f"target sounds per category (default {SOUNDS_PER_CATEGORY})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_db()
    client = FreesoundClient(get_api_key())

    categories = CATEGORIES if args.all else [args.category]
    total = 0
    for category in categories:
        total += collect_category(client, category, args.limit)
    log.info("Done — %d new sounds added.", total)


if __name__ == "__main__":
    main()
