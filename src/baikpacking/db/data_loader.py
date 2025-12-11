import json
import os
from pathlib import Path
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import Json
from db_connection import DB_DSN


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


DATA_PATH = Path("data/dotwatcher_bikes_cleaned.json")

# Database connection string
DB_DSN=DB_DSN

# ---------------------------------------------------------------------------
# SQL definitions
# ---------------------------------------------------------------------------

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT UNIQUE,
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

ARTICLE_UPSERT_SQL = """
INSERT INTO articles (title, url, body, raw)
VALUES (%(title)s, %(url)s, %(body)s, %(raw)s)
ON CONFLICT (url)
DO UPDATE SET
    title = EXCLUDED.title,
    body  = EXCLUDED.body,
    raw   = EXCLUDED.raw
RETURNING id;
"""

DELETE_RIDERS_FOR_ARTICLE_SQL = """
DELETE FROM riders WHERE article_id = %s;
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


def _load_json(path: Path) -> List[Dict[str, Any]]:
    """
    Load the cleaned JSON file.

    Supports:
    - A list of article dicts: [ {...}, {...}, ... ]
    - Or a single article dict: { ... }
    """
    if not path.exists():
        raise FileNotFoundError(f"Cleaned JSON not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Single article
        return [data]
    if isinstance(data, list):
        return data

    raise ValueError("Unexpected JSON structure: expected dict or list at top level.")


# ---------------------------------------------------------------------------
# Main loading logic
# ---------------------------------------------------------------------------

def main() -> None:
    articles = _load_json(DATA_PATH)
    if not articles:
        print("No articles found in JSON. Nothing to do.")
        return

    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = False

    try:
        with conn:
            with conn.cursor() as cur:
                # Make sure tables exist
                cur.execute(CREATE_TABLES_SQL)

                total_articles = 0
                total_riders = 0

                for article in articles:
                    # Upsert article
                    article_row = {
                        "title": article.get("title") or "Untitled",
                        "url": article.get("url"),
                        "body": article.get("body"),
                        "raw": Json(article),
                    }

                    cur.execute(ARTICLE_UPSERT_SQL, article_row)
                    article_id = cur.fetchone()[0]
                    total_articles += 1

                    # Replace riders for this article
                    cur.execute(DELETE_RIDERS_FOR_ARTICLE_SQL, (article_id,))

                    riders = article.get("riders", []) or []
                    for rider in riders:
                        cur.execute(INSERT_RIDER_SQL, normalize_rider(rider, article_id))
                        total_riders += 1

        print(f"Loaded/updated {total_articles} articles and {total_riders} riders into PostgreSQL.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
