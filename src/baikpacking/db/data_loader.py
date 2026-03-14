import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from psycopg2.extras import Json, execute_values
from pgvector.psycopg2 import register_vector

from baikpacking.db.db_connection import get_pg_connection
from baikpacking.embedding.embed import embed_texts_concurrent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SNAP_DIR = Path("data/snapshots/clean")
DEFAULT_PATTERN_JSON = "dotwatcher_bikes_cleaned_new_*.json"
DEFAULT_PATTERN_JSONL = "dotwatcher_bikes_cleaned_new_*.jsonl"

EXPECTED_EMBED_DIM = 1024
KEY_ITEMS_CHUNK_MAX_CHARS = 280

# ---------------------------------------------------------------------------
# Helpers: snapshot discovery + parsing
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r".*_(\d{8})_(\d{6})\.(json|jsonl)$", re.IGNORECASE)


def _extract_ts(path: Path) -> Optional[float]:
    """Extract YYYYMMDD_HHMMSS from filename (preferred), else None."""
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

    def sort_key(path: Path) -> Tuple[int, float]:
        ts = _extract_ts(path)
        if ts is not None:
            return (1, ts)
        return (0, path.stat().st_mtime)

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

    if isinstance(data, dict) and "articles" in data and isinstance(data["articles"], list):
        return data["articles"]
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data

    raise ValueError("Unexpected JSON structure: expected dict, list, or {'articles':[...]}.")


# ---------------------------------------------------------------------------
# Chunking helpers (based on riders.bike + riders.key_items)
# ---------------------------------------------------------------------------

def _norm_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


_SPLIT_RE = re.compile(r"(?:\n+|•|\u2022|- |\t|;|,|\|)+")


def split_key_items_to_phrases(key_items: Any) -> List[str]:
    """Turn riders.key_items into a list of short phrases."""
    if key_items is None:
        return []

    if isinstance(key_items, list):
        out: List[str] = []
        for item in key_items:
            s = _norm_str(item)
            if s:
                out.append(s)
        return out

    s = _norm_str(key_items)
    if not s:
        return []

    raw_parts = [p.strip() for p in _SPLIT_RE.split(s) if p and p.strip()]
    seen = set()
    out = []
    for part in raw_parts:
        k = part.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(part)
    return out


def pack_phrases_into_chunks(phrases: List[str], max_chars: int) -> List[str]:
    """Pack short phrases into chunk strings up to ~max_chars."""
    chunks: List[str] = []
    buf: List[str] = []
    size = 0

    for phrase in phrases:
        add_len = len(phrase) + (2 if buf else 0)
        if buf and (size + add_len) > max_chars:
            chunks.append("; ".join(buf))
            buf = [phrase]
            size = len(phrase)
        else:
            size = size + add_len if buf else len(phrase)
            buf.append(phrase)

    if buf:
        chunks.append("; ".join(buf))

    return chunks


def estimate_tokens_rough(text: str) -> int:
    """Fast token estimate (~4 chars/token)."""
    t = text.strip()
    if not t:
        return 0
    return max(1, len(t) // 4)


def build_rider_chunks_from_row(rider: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create chunk rows from riders.bike and riders.key_items.
    Returns list of dicts: chunk_kind, chunk_ix, chunk_text, chunk_tokens
    """
    out: List[Dict[str, Any]] = []

    bike = _norm_str(rider.get("bike"))
    key_items = rider.get("key_items")

    if bike:
        extra_bits: List[str] = []
        for key, label in [
            ("frame_type", "frame_type"),
            ("frame_material", "frame_material"),
            ("wheel_size", "wheel_size"),
            ("tyre_width", "tyre_width"),
            ("electronic_shifting", "electronic_shifting"),
        ]:
            value = rider.get(key)
            if value is not None and str(value).strip():
                extra_bits.append(f"{label}={str(value).strip()}")

        bike_text = f"Bike: {bike}" + (f" ({', '.join(extra_bits)})" if extra_bits else "")
        out.append(
            {
                "chunk_kind": "bike",
                "chunk_ix": 0,
                "chunk_text": bike_text,
                "chunk_tokens": estimate_tokens_rough(bike_text),
            }
        )

    phrases = split_key_items_to_phrases(key_items)
    if phrases:
        packed = pack_phrases_into_chunks(phrases, max_chars=KEY_ITEMS_CHUNK_MAX_CHARS)
        for i, text in enumerate(packed):
            chunk_text = f"Key items: {text}"
            out.append(
                {
                    "chunk_kind": "key_items",
                    "chunk_ix": i,
                    "chunk_text": chunk_text,
                    "chunk_tokens": estimate_tokens_rough(chunk_text),
                }
            )

    return out


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
    """Map an article JSON dict into DB columns (keep full payload in raw)."""
    title = article.get("title") or article.get("article_title") or "Untitled"
    url = article.get("url") or article.get("article_url")
    body = article.get("body") or article.get("content") or article.get("text")
    return {"title": title, "url": url, "body": body, "raw": Json(article)}


def normalize_rider(rider: Dict[str, Any], article_id: int) -> Dict[str, Any]:
    def g(*keys: str) -> Any:
        for key in keys:
            if key in rider and rider[key] is not None:
                return rider[key]
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
    """Extract riders list from a cleaned article record."""
    for key in ("riders", "rider_setups", "setups", "profiles", "people"):
        val = article.get(key)
        if isinstance(val, list):
            return [r for r in val if isinstance(r, dict)]
    return []


def fetch_article_ids_by_url(cur, urls: List[str]) -> Dict[str, int]:
    if not urls:
        return {}

    cur.execute(
        """
        SELECT id, url
        FROM public.articles
        WHERE url = ANY(%s);
        """,
        (urls,),
    )
    return {url: int(article_id) for article_id, url in cur.fetchall()}


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

ARTICLE_UPSERT_SQL = """
INSERT INTO articles (title, url, body, raw)
VALUES %s
ON CONFLICT (url) DO UPDATE
SET
  title = EXCLUDED.title,
  body = EXCLUDED.body,
  raw = EXCLUDED.raw;
"""

RIDER_INSERT_SQL = """
INSERT INTO riders (
  article_id, name, age, location, bike, key_items,
  frame_type, frame_material, wheel_size, tyre_width, electronic_shifting, raw
) VALUES %s;
"""

REQUIRED_TABLES_BASE = ("articles", "riders")
REQUIRED_TABLES_WITH_CHUNKS = ("articles", "riders", "rider_chunks")

FETCH_RIDERS_MISSING_CHUNKS_SQL = """
SELECT
  r.id,
  r.article_id,
  r.name,
  r.location,
  r.bike,
  r.key_items,
  r.frame_type,
  r.frame_material,
  r.wheel_size,
  r.tyre_width,
  r.electronic_shifting
FROM riders r
LEFT JOIN rider_chunks c ON c.rider_id = r.id
WHERE c.rider_id IS NULL
ORDER BY r.id;
"""

FETCH_RIDERS_FOR_CHUNKS_SQL = """
SELECT
  id,
  article_id,
  name,
  location,
  bike,
  key_items,
  frame_type,
  frame_material,
  wheel_size,
  tyre_width,
  electronic_shifting
FROM riders
ORDER BY id;
"""

TRUNCATE_RIDER_CHUNKS_SQL = "TRUNCATE TABLE rider_chunks;"

UPSERT_RIDER_CHUNKS_SQL = """
INSERT INTO rider_chunks (
  rider_id, chunk_kind, chunk_ix, chunk_text, chunk_tokens, embedding, model
)
VALUES %s
ON CONFLICT (rider_id, chunk_kind, chunk_ix)
DO UPDATE SET
  chunk_text   = EXCLUDED.chunk_text,
  chunk_tokens = EXCLUDED.chunk_tokens,
  embedding    = EXCLUDED.embedding,
  model        = EXCLUDED.model,
  updated_at   = now();
"""


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_articles_url_unique(cur) -> None:
    """Ensure uniqueness guarantee for public.articles(url)."""
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


def assert_tables_exist(cur, required: Tuple[str, ...]) -> None:
    cur.execute(
        """
        select tablename
        from pg_tables
        where schemaname='public' and tablename = any(%s)
        """,
        (list(required),),
    )
    found = {r[0] for r in cur.fetchall()}
    missing = [t for t in required if t not in found]
    if missing:
        raise RuntimeError(f"Missing tables in DB: {missing}. Run your schema/migrations first.")


# ---------------------------------------------------------------------------
# Chunk embedding routine
# ---------------------------------------------------------------------------

def fetch_riders_for_chunks(conn, only_missing: bool) -> List[Dict[str, Any]]:
    sql = FETCH_RIDERS_MISSING_CHUNKS_SQL if only_missing else FETCH_RIDERS_FOR_CHUNKS_SQL
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def truncate_rider_chunks(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(TRUNCATE_RIDER_CHUNKS_SQL)
    conn.commit()


def upsert_rider_chunks(
    conn,
    records: List[Tuple[int, str, int, str, int, List[float], str]],
    page_size: int = 500,
) -> int:
    """
    records: (rider_id, chunk_kind, chunk_ix, chunk_text, chunk_tokens, embedding, model)
    """
    if not records:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, UPSERT_RIDER_CHUNKS_SQL, records, page_size=page_size)
    conn.commit()
    return len(records)


def build_and_embed_chunks(
    conn,
    model_name: str,
    only_missing: bool,
    batch_size: int,
    dry_run: bool,
    max_workers: int = 8,
) -> Dict[str, Any]:
    register_vector(conn)

    riders = fetch_riders_for_chunks(conn, only_missing=only_missing)

    chunk_rows: List[Tuple[int, str, int, str, int]] = []
    for rider in riders:
        rider_id = int(rider["id"])
        for chunk in build_rider_chunks_from_row(rider):
            chunk_rows.append(
                (
                    rider_id,
                    str(chunk["chunk_kind"]),
                    int(chunk["chunk_ix"]),
                    str(chunk["chunk_text"]),
                    int(chunk["chunk_tokens"]),
                )
            )

    if not chunk_rows:
        return {"riders_considered": len(riders), "chunks_built": 0, "chunks_upserted": 0}

    total_upserted = 0

    for i in range(0, len(chunk_rows), batch_size):
        batch = chunk_rows[i : i + batch_size]
        texts = [row[3] for row in batch]

        vectors = embed_texts_concurrent(texts, max_workers=max_workers)

        if len(vectors) != len(texts):
            raise RuntimeError(f"Chunk embed count mismatch: {len(vectors)} != {len(texts)}")

        if vectors and len(vectors[0]) != EXPECTED_EMBED_DIM:
            raise RuntimeError(
                f"Embedding dimension mismatch: got {len(vectors[0])}, expected {EXPECTED_EMBED_DIM}. "
                "Update EXPECTED_EMBED_DIM and your DB vector(N)."
            )

        upsert_records: List[Tuple[int, str, int, str, int, List[float], str]] = []
        for row, vec in zip(batch, vectors):
            rider_id, chunk_kind, chunk_ix, chunk_text, chunk_tokens = row
            upsert_records.append(
                (rider_id, chunk_kind, chunk_ix, chunk_text, chunk_tokens, vec, model_name)
            )

        if dry_run:
            continue

        total_upserted += upsert_rider_chunks(conn, upsert_records, page_size=500)

    if dry_run:
        conn.rollback()
        return {
            "riders_considered": len(riders),
            "chunks_built": len(chunk_rows),
            "chunks_upserted": 0,
            "dry_run": True,
        }

    return {
        "riders_considered": len(riders),
        "chunks_built": len(chunk_rows),
        "chunks_upserted": total_upserted,
    }


# ---------------------------------------------------------------------------
# Main load routine (articles + riders)
# ---------------------------------------------------------------------------

def delete_riders_for_article_ids(cur, article_ids: List[int]) -> int:
    if not article_ids:
        return 0
    cur.execute(
        """
        DELETE FROM public.riders
        WHERE article_id = ANY(%s);
        """,
        (article_ids,),
    )
    return cur.rowcount


def sync_snapshot_articles_and_riders(
    conn,
    input_path: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    articles_in = _load_input(input_path)
    if not articles_in:
        return {
            "snapshot": str(input_path),
            "input_articles": 0,
            "normalized_articles": 0,
            "articles_resolved": 0,
            "article_ids": 0,
            "deleted_riders": 0,
            "inserted_riders": 0,
            "skipped_no_url": 0,
        }

    normalized_articles: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for article in articles_in:
        na = normalize_article(article)
        if not na["url"]:
            continue
        normalized_articles.append((na, article))

    skipped_no_url = len(articles_in) - len(normalized_articles)

    with conn.cursor() as cur:
        assert_tables_exist(cur, REQUIRED_TABLES_BASE)
        assert_articles_url_unique(cur)

        article_rows = [
            (na["title"], na["url"], na["body"], na["raw"])
            for na, _ in normalized_articles
        ]
        if article_rows:
            execute_values(cur, ARTICLE_UPSERT_SQL, article_rows, page_size=200)

        urls = [na["url"] for na, _ in normalized_articles if na.get("url")]
        url_to_article_id = fetch_article_ids_by_url(cur, urls)
        article_ids = sorted(set(url_to_article_id.values()))

        deleted_riders = delete_riders_for_article_ids(cur, article_ids)

        rider_rows: List[Tuple[Any, ...]] = []
        batch_seen = set()

        for na, original_article in normalized_articles:
            article_url = na["url"]
            article_id = url_to_article_id.get(article_url)
            if not article_id:
                continue

            for rider in extract_riders(original_article):
                nr = normalize_rider(rider, article_id)

                rider_name = (nr["name"] or "").strip().lower()
                dedupe_key = (article_id, rider_name)
                if dedupe_key in batch_seen:
                    continue
                batch_seen.add(dedupe_key)

                rider_rows.append(
                    (
                        nr["article_id"],
                        nr["name"],
                        nr["age"],
                        nr["location"],
                        nr["bike"],
                        nr["key_items"],
                        nr["frame_type"],
                        nr["frame_material"],
                        nr["wheel_size"],
                        nr["tyre_width"],
                        nr["electronic_shifting"],
                        nr["raw"],
                    )
                )

        inserted_riders = 0
        if rider_rows:
            execute_values(cur, RIDER_INSERT_SQL, rider_rows, page_size=500)
            inserted_riders = len(rider_rows)

    if dry_run:
        conn.rollback()
    else:
        conn.commit()

    return {
        "snapshot": str(input_path),
        "input_articles": len(articles_in),
        "normalized_articles": len(normalized_articles),
        "articles_resolved": len(url_to_article_id),
        "article_ids": len(article_ids),
        "deleted_riders": deleted_riders,
        "inserted_riders": inserted_riders,
        "skipped_no_url": skipped_no_url,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load cleaned DotWatcher snapshot and optionally build rider_chunks.")
    parser.add_argument("--input", type=str, default="", help="Optional cleaned snapshot (.json/.jsonl).")
    parser.add_argument("--snap-dir", type=str, default=str(DEFAULT_SNAP_DIR), help="Snapshots directory.")
    parser.add_argument("--dry-run", action="store_true", help="Rollback everything; print stats only.")

    parser.add_argument("--with-chunks", action="store_true", help="Build+embed rider_chunks after loading.")
    parser.add_argument("--only-missing-chunks", action="store_true", help="Only build chunks for riders without chunks.")
    parser.add_argument("--rebuild-chunks", action="store_true", help="TRUNCATE rider_chunks then rebuild.")
    parser.add_argument("--chunk-batch-size", type=int, default=128, help="Embedding batch size for chunks.")
    parser.add_argument("--chunk-embedding-model-name", type=str, default="ollama", help="Label stored in rider_chunks.model.")
    args = parser.parse_args()

    snap_dir = Path(args.snap_dir)
    input_path = Path(args.input) if args.input else find_latest_new_snapshot(snap_dir)
    print(f"Loading snapshot: {input_path}")

    with get_pg_connection(autocommit=False) as conn:
        register_vector(conn)

        stats = sync_snapshot_articles_and_riders(
            conn,
            input_path=input_path,
            dry_run=args.dry_run,
        )
        print(f"Articles+riders stats: {stats}")

        if not args.with_chunks:
            return

        with conn.cursor() as cur:
            assert_tables_exist(cur, REQUIRED_TABLES_WITH_CHUNKS)

        if args.rebuild_chunks:
            if args.dry_run:
                print("[dry-run] would TRUNCATE rider_chunks")
            else:
                truncate_rider_chunks(conn)

        only_missing = args.only_missing_chunks and not args.rebuild_chunks

        chunk_stats = build_and_embed_chunks(
            conn,
            model_name=args.chunk_embedding_model_name,
            only_missing=only_missing,
            batch_size=max(1, args.chunk_batch_size),
            dry_run=args.dry_run,
        )
        print(f"Chunks stats: {chunk_stats}")


if __name__ == "__main__":
    main()