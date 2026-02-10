import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from psycopg2.extras import Json, execute_values

from baikpacking.db.db_connection import get_pg_connection

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SNAP_DIR = Path("data/snapshots/clean")
DEFAULT_PATTERN_JSON = "dotwatcher_bikes_cleaned_new_*.json"
DEFAULT_PATTERN_JSONL = "dotwatcher_bikes_cleaned_new_*.jsonl"

# ---------------------------------------------------------------------------
# Helpers: snapshot discovery + parsing
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r".*_(\d{8})_(\d{6})\.(json|jsonl)$", re.IGNORECASE)


def _extract_ts(path: Path) -> Optional[float]:
    """
    Extract YYYYMMDD_HHMMSS from filename (preferred), else None.
    """
    m = _TS_RE.match(path.name)
    if not m:
        return None
    try:
        dt = datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
        return dt.timestamp()
    except Exception:
        return None


def find_latest_new_snapshot(snap_dir: Path) -> Path:
    """
    Pick the newest snapshot:
    - Prefer timestamp in filename (..._YYYYMMDD_HHMMSS.json|jsonl)
    - Fallback to file modified time
    """
    files = list(snap_dir.glob(DEFAULT_PATTERN_JSON)) + list(snap_dir.glob(DEFAULT_PATTERN_JSONL))
    if not files:
        raise FileNotFoundError(
            f"No cleaned snapshots found in {snap_dir}. "
            f"Expected {DEFAULT_PATTERN_JSON} or {DEFAULT_PATTERN_JSONL}"
        )

    def sort_key(p: Path) -> Tuple[int, float]:
        ts = _extract_ts(p)
        if ts is not None:
            return (1, ts)
        return (0, p.stat().st_mtime)

    return sorted(files, key=sort_key, reverse=True)[0]


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
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

    # Accept:
    # - list[article]
    # - {"articles":[...]}
    # - dict single article
    if isinstance(data, dict) and "articles" in data and isinstance(data["articles"], list):
        return data["articles"]
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data

    raise ValueError("Unexpected JSON structure: expected dict, list, or {'articles':[...]}.")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def normalize_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map an article JSON dict into DB columns.
    Keep full payload in raw for traceability.
    """
    title = article.get("title") or article.get("article_title") or "Untitled"
    url = article.get("url") or article.get("article_url")
    body = article.get("body") or article.get("content") or article.get("text")
    return {"title": title, "url": url, "body": body, "raw": Json(article)}


def normalize_rider(rider: Dict[str, Any], article_id: int) -> Dict[str, Any]:
    def g(*keys: str) -> Any:
        for k in keys:
            if k in rider and rider[k] is not None:
                return rider[k]
        return None

    return {
        "article_id": article_id,
        "name": g("name", "rider_name"),
        "age": _to_int_or_none(g("age")),
        "location": g("location", "country", "region"),
        "bike": g("bike", "bike_model"),
        "key_items": g("key_items", "keyItems", "highlights"),
        "frame_type": g("frame_type", "frameType"),
        "frame_material": g("frame_material", "frameMaterial"),
        "wheel_size": g("wheel_size", "wheelSize"),
        "tyre_width": g("tyre_width", "tire_width", "tyreWidth"),
        "electronic_shifting": g("electronic_shifting", "electronicShifting"),
        "raw": Json(rider),
    }


def extract_riders(article: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Your cleaned snapshots may store riders under different keys.
    Keep this centralized so we don’t scatter heuristics everywhere.
    """
    for key in ("riders", "rider_setups", "setups", "profiles", "people"):
        val = article.get(key)
        if isinstance(val, list):
            return [r for r in val if isinstance(r, dict)]
    return []


# ---------------------------------------------------------------------------
# SQL (incremental)
# ---------------------------------------------------------------------------

# IMPORTANT:
# - If url is NULL/empty, we skip (we can't link/dedupe reliably).
# - We use ON CONFLICT DO NOTHING (no target) for compatibility with UNIQUE INDEX.
# - Incremental behavior expects a uniqueness guarantee on articles.url (index or constraint).
ARTICLE_UPSERT_SQL = """
INSERT INTO articles (title, url, body, raw)
VALUES %s
ON CONFLICT DO NOTHING
RETURNING id, url;
"""

RIDER_INSERT_SQL = """
INSERT INTO riders (
  article_id, name, age, location, bike, key_items,
  frame_type, frame_material, wheel_size, tyre_width, electronic_shifting, raw
) VALUES %s;
"""

REQUIRED_TABLES = ("articles", "riders")

FETCH_RIDERS_SQL = """
SELECT
  id,
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
FROM riders
ORDER BY id;
"""

TRUNCATE_RIDER_EMBEDDINGS_SQL = "TRUNCATE TABLE rider_embeddings;"

UPSERT_RIDER_EMBEDDINGS_SQL = """
INSERT INTO rider_embeddings (rider_id, embedding, model)
VALUES %s
ON CONFLICT (rider_id)
DO UPDATE SET
  embedding = EXCLUDED.embedding,
  model = EXCLUDED.model,
  updated_at = now();
"""



def assert_articles_url_unique(cur) -> None:
    """
    Ensure there is a uniqueness guarantee for articles.url.

    Since the schema creates a UNIQUE INDEX named idx_articles_url,
    we assert that either:
      - a UNIQUE constraint exists on (url), OR
      - the UNIQUE INDEX idx_articles_url exists.
    """
    # Unique constraint on url?
    cur.execute(
        """
        select 1
        from pg_constraint c
        join pg_class t on t.oid = c.conrelid
        join pg_namespace n on n.oid = t.relnamespace
        where n.nspname = 'public'
          and t.relname = 'articles'
          and c.contype = 'u'
          and pg_get_constraintdef(c.oid) = 'UNIQUE (url)'
        limit 1;
        """
    )
    if cur.fetchone() is not None:
        return

    # Unique index by the expected name?
    cur.execute(
        """
        select 1
        from pg_indexes
        where schemaname='public'
          and tablename='articles'
          and indexname='idx_articles_url';
        """
    )
    if cur.fetchone() is not None:
        return

    raise RuntimeError(
        "Expected a uniqueness guarantee for public.articles(url), but none was found.\n"
        "Create one using either:\n"
        "  CREATE UNIQUE INDEX idx_articles_url ON articles(url);\n"
        "or\n"
        "  ALTER TABLE articles ADD CONSTRAINT articles_url_unique UNIQUE (url);"
    )


def assert_tables_exist(cur) -> None:
    cur.execute(
        """
        select tablename
        from pg_tables
        where schemaname='public' and tablename = any(%s)
        """,
        (list(REQUIRED_TABLES),),
    )
    found = {r[0] for r in cur.fetchall()}
    missing = [t for t in REQUIRED_TABLES if t not in found]
    if missing:
        raise RuntimeError(
            f"Missing tables in DB: {missing}. Run your schema/migrations first."
        )


def load_latest_snapshot_into_db(
    snap_dir: Path = DEFAULT_SNAP_DIR,
    input_path: Optional[Path] = None,
    dry_run: bool = False,
) -> dict:
    """
    Incrementally load the newest cleaned snapshot into Postgres.
    Returns a small stats dict.
    """
    path = input_path or find_latest_new_snapshot(snap_dir)
    articles_in = _load_input(path)
    if not articles_in:
        return {"snapshot": str(path), "inserted_articles": 0, "inserted_riders": 0, "skipped_no_url": 0}

    normalized_articles = []
    for a in articles_in:
        na = normalize_article(a)
        if not na["url"]:
            continue
        normalized_articles.append((na, a))

    inserted_articles = 0
    inserted_riders = 0
    skipped_no_url = len(articles_in) - len(normalized_articles)

    with get_pg_connection(autocommit=False) as conn:
        with conn.cursor() as cur:
            assert_tables_exist(cur)
            assert_articles_url_unique(cur)

            article_rows = [(na["title"], na["url"], na["body"], na["raw"]) for (na, _) in normalized_articles]
            execute_values(cur, ARTICLE_UPSERT_SQL, article_rows, page_size=200)
            inserted = cur.fetchall()
            url_to_article_id = {url: aid for (aid, url) in inserted}
            inserted_articles = len(url_to_article_id)

            rider_rows = []
            for (na, original_article) in normalized_articles:
                aid = url_to_article_id.get(na["url"])
                if not aid:
                    continue
                for r in extract_riders(original_article):
                    nr = normalize_rider(r, aid)
                    rider_rows.append(
                        (
                            nr["article_id"], nr["name"], nr["age"], nr["location"], nr["bike"], nr["key_items"],
                            nr["frame_type"], nr["frame_material"], nr["wheel_size"], nr["tyre_width"],
                            nr["electronic_shifting"], nr["raw"],
                        )
                    )

            if rider_rows:
                execute_values(cur, RIDER_INSERT_SQL, rider_rows, page_size=500)
                inserted_riders = len(rider_rows)

            if dry_run:
                conn.rollback()
            else:
                conn.commit()

    return {
        "snapshot": str(path),
        "inserted_articles": inserted_articles,
        "inserted_riders": inserted_riders,
        "skipped_no_url": skipped_no_url,
    }

def fetch_riders(conn) -> List[Dict[str, Any]]:
    """
    Fetch riders from Postgres as list[dict].
    This is the source for embedding.
    """
    with conn.cursor() as cur:
        cur.execute(FETCH_RIDERS_SQL)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def truncate_rider_embeddings(conn) -> None:
    """
    Clear rider_embeddings table (for a full rebuild).
    """
    with conn.cursor() as cur:
        cur.execute(TRUNCATE_RIDER_EMBEDDINGS_SQL)
    conn.commit()


def upsert_rider_embeddings(
    conn,
    records: List[Tuple[int, List[float], str]],
    page_size: int = 500,
) -> int:
    """
    Upsert embeddings into rider_embeddings in batch.

    Args:
        conn: psycopg2 connection
        records: list of (rider_id, embedding_vector, model_name)
        page_size: batch size for execute_values

    Returns:
        Number of rows attempted (len(records)).
    """
    if not records:
        return 0

    with conn.cursor() as cur:
        execute_values(
            cur,
            UPSERT_RIDER_EMBEDDINGS_SQL,
            records,
            page_size=page_size,
        )
    conn.commit()
    return len(records)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally load newest cleaned DotWatcher snapshot into DB.")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Optional path to a cleaned snapshot (.json or .jsonl). If omitted, loads the latest in data/snapshots/clean.",
    )
    parser.add_argument(
        "--snap-dir",
        type=str,
        default=str(DEFAULT_SNAP_DIR),
        help="Directory where cleaned snapshots live.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report how many rows would be inserted, but do not write to DB.",
    )
    args = parser.parse_args()

    snap_dir = Path(args.snap_dir)
    input_path = Path(args.input) if args.input else find_latest_new_snapshot(snap_dir)

    print(f"Loading snapshot: {input_path}")

    articles_in = _load_input(input_path)
    if not articles_in:
        print("Snapshot is empty. Nothing to load.")
        return

    # Normalize + filter out missing URLs (cannot dedupe)
    normalized_articles = []
    for a in articles_in:
        na = normalize_article(a)
        if not na["url"]:
            continue
        normalized_articles.append((na, a))  # keep original for rider extraction

    if args.dry_run:
        print(f"[dry-run] parsed articles with url: {len(normalized_articles)} (out of {len(articles_in)})")

    inserted_articles = 0
    inserted_riders = 0
    skipped_no_url = len(articles_in) - len(normalized_articles)

    with get_pg_connection(autocommit=False) as conn:
        with conn.cursor() as cur:
            assert_tables_exist(cur)
            assert_articles_url_unique(cur)
            

            # Batch insert articles and capture ids for newly inserted ones only
            article_rows = [(na["title"], na["url"], na["body"], na["raw"]) for (na, _) in normalized_articles]

            if args.dry_run:
                # We can’t know “new vs existing” without hitting the DB; we still do a rollback.
                pass

            execute_values(cur, ARTICLE_UPSERT_SQL, article_rows, page_size=200)
            inserted = cur.fetchall()  # rows returned are ONLY newly inserted due to DO NOTHING
            url_to_article_id = {url: aid for (aid, url) in inserted}
            inserted_articles = len(url_to_article_id)

            # Insert riders only for the newly inserted articles
            rider_rows = []
            for (na, original_article) in normalized_articles:
                aid = url_to_article_id.get(na["url"])
                if not aid:
                    continue  # existing article -> do not add riders here (incremental load)
                riders = extract_riders(original_article)
                for r in riders:
                    nr = normalize_rider(r, aid)
                    rider_rows.append(
                        (
                            nr["article_id"], nr["name"], nr["age"], nr["location"], nr["bike"], nr["key_items"],
                            nr["frame_type"], nr["frame_material"], nr["wheel_size"], nr["tyre_width"],
                            nr["electronic_shifting"], nr["raw"],
                        )
                    )

            if rider_rows:
                execute_values(cur, RIDER_INSERT_SQL, rider_rows, page_size=500)
                inserted_riders = len(rider_rows)

            if args.dry_run:
                conn.rollback()
                print(f"[dry-run] would insert ~{inserted_articles} new articles and ~{inserted_riders} riders")
                print(f"[dry-run] skipped articles with missing url: {skipped_no_url}")
                return

            conn.commit()

    print(
        f"Inserted {inserted_articles} new articles and {inserted_riders} riders. "
        f"Skipped missing-url: {skipped_no_url}."
    )


if __name__ == "__main__":
    main()
