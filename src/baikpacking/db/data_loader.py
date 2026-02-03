import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import psycopg2
from psycopg2.extras import Json

from baikpacking.db.db_connection import DB_DSN


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


DEFAULT_SNAP_DIR = Path("data/snapshots/clean")
DEFAULT_PATTERN_JSON = "dotwatcher_bikes_cleaned_new_*.json"
DEFAULT_PATTERN_JSONL = "dotwatcher_bikes_cleaned_new_*.jsonl"

# Database connection string
DB_DSN=DB_DSN

# ---------------------------------------------------------------------------
# SQL definitions
# ---------------------------------------------------------------------------

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    body TEXT,
    raw JSONB
);

CREATE TABLE IF NOT EXISTS riders (
    id SERIAL PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    name TEXT,
    age INT,
    location TEXT,
    bike TEXT,
    key_items TEXT,
    frame_type TEXT,
    frame_material TEXT,
    wheel_size TEXT,
    tyre_width TEXT,
    electronic_shifting BOOLEAN,
    raw JSONB
);
"""

ARTICLE_INSERT_IF_NEW_SQL = """
INSERT INTO articles (title, url, body, raw)
VALUES (%(title)s, %(url)s, %(body)s, %(raw)s)
ON CONFLICT (url) DO NOTHING
RETURNING id;
"""

INSERT_RIDER_SQL = """
INSERT INTO riders (
    article_id,
    name,
    age,
    location,
    bike,
    key_items,
    frame_type,
    frame_material,
    wheel_size,
    tyre_width,
    electronic_shifting,
    raw
) VALUES (
    %(article_id)s,
    %(name)s,
    %(age)s,
    %(location)s,
    %(bike)s,
    %(key_items)s,
    %(frame_type)s,
    %(frame_material)s,
    %(wheel_size)s,
    %(tyre_width)s,
    %(electronic_shifting)s,
    %(raw)s
);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_input(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.suffix.lower() == ".jsonl":
        return list(_iter_jsonl(path))

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Unexpected JSON structure: expected dict or list.")


def find_latest_new_snapshot(snap_dir: Path) -> Path:
    files = sorted(list(snap_dir.glob(DEFAULT_PATTERN_JSON)) + list(snap_dir.glob(DEFAULT_PATTERN_JSONL)))
    if not files:
        raise FileNotFoundError(
            f"No new-only cleaned snapshots found in {snap_dir}. "
            f"Expected {DEFAULT_PATTERN_JSON} or {DEFAULT_PATTERN_JSONL}"
        )
    return files[-1]


def _fetch_existing_urls(cur) -> Set[str]:
    cur.execute("SELECT url FROM articles WHERE url IS NOT NULL;")
    return {row[0] for row in cur.fetchall()}

def _to_int_or_none(value: Any) -> int | None:
    """Convert a value to int if possible, otherwise return None."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def normalize_rider(rider: Dict[str, Any], article_id: int) -> Dict[str, Any]:
    """Map a rider JSON dict into DB columns."""
    return {
        "article_id": article_id,
        "name": rider.get("name"),
        "age": _to_int_or_none(rider.get("age")),
        "location": rider.get("location"),
        "bike": rider.get("bike"),
        "key_items": rider.get("key_items"),
        "frame_type": rider.get("frame_type"),
        "frame_material": rider.get("frame_material"),
        "wheel_size": rider.get("wheel_size"),
        "tyre_width": rider.get("tyre_width"),
        "electronic_shifting": rider.get("electronic_shifting"),
        "raw": Json(rider),
    }


# ---------------------------------------------------------------------------
# Main loading logic
# ---------------------------------------------------------------------------

def _fetch_existing_urls(cur) -> Set[str]:
    cur.execute("SELECT url FROM articles WHERE url IS NOT NULL;")
    return {row[0] for row in cur.fetchall()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Load ONLY new cleaned DotWatcher articles into DB.")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Path to a cleaned new-only snapshot (.json or .jsonl). If omitted, loads the latest snapshot in data/snapshots/clean.",
    )
    args = parser.parse_args()

    snap_dir = DEFAULT_SNAP_DIR
    input_path = Path(args.input) if args.input else find_latest_new_snapshot(snap_dir)

    print(f"Loading new-only snapshot: {input_path}")

    articles = _load_input(input_path)
    if not articles:
        print("Snapshot is empty. Nothing to load.")
        return

    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = False

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLES_SQL)

                existing_urls = _fetch_existing_urls(cur)
                print(f"Existing articles in DB: {len(existing_urls)}")

                inserted_articles = 0
                inserted_riders = 0
                skipped_articles = 0

                for article in articles:
                    url = article.get("url")
                    if not url:
                        skipped_articles += 1
                        continue

                    if url in existing_urls:
                        skipped_articles += 1
                        continue

                    article_row = {
                        "title": article.get("title") or "Untitled",
                        "url": url,
                        "body": article.get("body"),
                        "raw": Json(article),
                    }

                    cur.execute(ARTICLE_INSERT_IF_NEW_SQL, article_row)
                    res = cur.fetchone()

                    if res is None:
                        skipped_articles += 1
                        continue

                    article_id = res[0]
                    inserted_articles += 1
                    existing_urls.add(url)

                    riders = article.get("riders", []) or []
                    for rider in riders:
                        cur.execute(INSERT_RIDER_SQL, normalize_rider(rider, article_id))
                        inserted_riders += 1

        print(
            f"Inserted {inserted_articles} new articles and {inserted_riders} riders "
            f"(skipped {skipped_articles})."
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
