import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

from pydantic import ValidationError
from pydantic_ai import RunContext, Tool
from psycopg2.extras import RealDictCursor  # works for psycopg2; safe if psycopg3 not used here

from baikpacking.agents.models import SimilarRider
from baikpacking.tools._trace_utils import trace_tool
from baikpacking.tools.pg_vector_search import PgVectorSearchDeps, _get_deps as _get_pg_deps

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"(19|20)\d{2}")

# Cache table-existence checks per DB URL to avoid repeated checks/warnings.
_RIDER_CHUNKS_EXISTS: Dict[str, bool] = {}


# -------------------------
# Small helpers
# -------------------------

def _infer_year_from_title(title: Optional[str]) -> Optional[int]:
    """Infer year from a title like 'Transcontinental No10 2024'."""
    if not title:
        return None
    m = _YEAR_RE.search(title)
    return int(m.group(0)) if m else None


def _extract_event_hint(query: str, event_keywords: List[str]) -> Optional[str]:
    """Extract an event keyword present in the query (lowercased contains match)."""
    q = query.lower()
    for key in event_keywords:
        if key in q:
            return key
    return None


def _connect(database_url: str):
    """
    Prefer psycopg3; fall back to psycopg2.

    Note: This file uses psycopg2 extras (RealDictCursor) for convenience.
    If you run psycopg3 here, you can remove RealDictCursor usage or implement
    row->dict conversion manually.
    """
    try:
        import psycopg  # type: ignore
        return psycopg.connect(database_url)
    except Exception:
        import psycopg2  # type: ignore
        return psycopg2.connect(database_url)


def _vector_text(vec: Sequence[float]) -> str:
    """Format vector for pgvector input (passed as bound param cast to ::vector)."""
    # Keeping this to avoid needing pgvector adapters in this tool.
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


def _clip_text(s: Any, max_chars: int = 800) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    return s if len(s) <= max_chars else (s[: max_chars - 1] + "…")


def _table_exists(conn, table_name: str) -> bool:
    """Fast existence check using to_regclass."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s) IS NOT NULL;", (f"public.{table_name}",))
            row = cur.fetchone()
            return bool(row[0]) if row is not None else False
    except Exception:
        return False


# -------------------------
# Core DB routines
# -------------------------

def _fetch_candidates(
    conn,
    qvec_param: str,
    limit: int,
    ef_search: Optional[int],
) -> List[Tuple[int, float]]:
    """Return [(rider_id, distance)] from rider_embeddings."""
    sql = """
    SELECT
      rider_id,
      (embedding <=> %s::vector)::float AS distance
    FROM rider_embeddings
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """
    with conn.cursor() as cur:
        if ef_search is not None:
            # This only matters for hnsw; harmless if extension/setting not present.
            cur.execute("SET LOCAL hnsw.ef_search = %s;", (int(ef_search),))
        cur.execute(sql, (qvec_param, qvec_param, int(limit)))
        return [(int(rid), float(dist)) for (rid, dist) in cur.fetchall()]


def _resolve_riders_cols(cur) -> dict:
    """Resolve riders column names to avoid schema drift issues."""
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'riders';
        """
    )
    cols = {r[0] for r in cur.fetchall()}

    def pick(*cands: str) -> Optional[str]:
        for c in cands:
            if c in cols:
                return c
        return None

    return {
        "event_title": pick("event_title", "event", "event_name", "race", "title"),
        "event_url": pick("event_url", "url", "event_link"),
        "article_id": pick("article_id", "dotwatcher_id"),
        "event_key": pick("event_key"),
        "frame_type": pick("frame_type"),
        "frame_material": pick("frame_material"),
        "wheel_size": pick("wheel_size"),
        "tyre_width": pick("tyre_width"),
        "electronic_shifting": pick("electronic_shifting"),
        "name": pick("name"),
        "bike": pick("bike"),
        "key_items": pick("key_items", "keyitems", "kit", "gear"),
    }


def _fetch_rider_rows(conn, rider_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Hydrate SimilarRider core fields from `riders`."""
    if not rider_ids:
        return {}

    out: Dict[int, Dict[str, Any]] = {}
    with conn.cursor() as cur:
        col = _resolve_riders_cols(cur)

        if not col["name"]:
            raise RuntimeError("riders table must contain a 'name' column (or equivalent).")

        sel_event = f"r.{col['event_title']} AS event_title" if col["event_title"] else "NULL::text AS event_title"
        sel_url = f"r.{col['event_url']} AS event_url" if col["event_url"] else "NULL::text AS event_url"
        sel_article = f"r.{col['article_id']} AS article_id" if col["article_id"] else "NULL::int AS article_id"
        sel_key = f"r.{col['event_key']} AS event_key" if col["event_key"] else "NULL::text AS event_key"
        sel_frame_type = f"r.{col['frame_type']} AS frame_type" if col["frame_type"] else "NULL::text AS frame_type"
        sel_frame_mat = f"r.{col['frame_material']} AS frame_material" if col["frame_material"] else "NULL::text AS frame_material"
        sel_wheel = f"r.{col['wheel_size']} AS wheel_size" if col["wheel_size"] else "NULL::text AS wheel_size"
        sel_tyre = f"r.{col['tyre_width']} AS tyre_width" if col["tyre_width"] else "NULL::text AS tyre_width"
        sel_elec = f"r.{col['electronic_shifting']} AS electronic_shifting" if col["electronic_shifting"] else "NULL::boolean AS electronic_shifting"
        sel_bike = f"r.{col['bike']} AS bike" if col["bike"] else "NULL::text AS bike"
        sel_key_items = f"r.{col['key_items']} AS key_items" if col["key_items"] else "NULL::text AS key_items"

        sql = f"""
        SELECT
          r.id,
          r.{col['name']} AS name,
          {sel_article},
          {sel_event},
          {sel_url},
          {sel_key},
          {sel_frame_type},
          {sel_frame_mat},
          {sel_wheel},
          {sel_tyre},
          {sel_elec},
          {sel_bike},
          {sel_key_items}
        FROM riders r
        WHERE r.id = ANY(%s::int[]);
        """

        cur.execute(sql, (rider_ids,))
        for row in cur.fetchall():
            (
                rid,
                name,
                article_id,
                event_title,
                event_url,
                event_key,
                frame_type,
                frame_material,
                wheel_size,
                tyre_width,
                electronic_shifting,
                bike,
                key_items,
            ) = row

            out[int(rid)] = {
                "rider_id": int(rid),
                "name": name,
                "article_id": int(article_id) if article_id is not None else None,
                "event_title": event_title,
                "event_url": event_url,
                "event_key": event_key,
                "frame_type": frame_type,
                "frame_material": frame_material,
                "wheel_size": wheel_size,
                "tyre_width": tyre_width,
                "electronic_shifting": electronic_shifting,
                "bike": bike,
                "key_items": key_items,
            }

    return out


def _ensure_chunks_table_known(conn, database_url: str) -> bool:
    """Memoized check for rider_chunks existence (per DB url)."""
    exists = _RIDER_CHUNKS_EXISTS.get(database_url)
    if exists is not None:
        return exists

    exists = _table_exists(conn, "rider_chunks")
    _RIDER_CHUNKS_EXISTS[database_url] = exists
    if not exists:
        logger.warning("Table public.rider_chunks does not exist; skipping chunks retrieval.")
    return exists


def _fetch_chunks(
    conn,
    database_url: str,
    rider_ids: List[int],
    max_chunks_per_rider: int,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Fetch up to max_chunks_per_rider chunks per rider.

    Returns:
      { rider_id: [ {"score": float, "text": str, "chunk_kind": str, "chunk_ix": int}, ... ] }
    """
    if not rider_ids or max_chunks_per_rider <= 0:
        return {rid: [] for rid in rider_ids}

    if not _ensure_chunks_table_known(conn, database_url):
        return {rid: [] for rid in rider_ids}

    # Use window function to cap per rider in SQL (fast + deterministic)
    sql = """
    WITH ranked AS (
      SELECT
        rider_id,
        chunk_kind,
        chunk_ix,
        chunk_text,
        chunk_tokens,
        row_number() OVER (
          PARTITION BY rider_id
          ORDER BY chunk_kind, chunk_ix
        ) AS rn
      FROM public.rider_chunks
      WHERE rider_id = ANY(%s::int[])
    )
    SELECT rider_id, chunk_kind, chunk_ix, chunk_text, chunk_tokens
    FROM ranked
    WHERE rn <= %s
    ORDER BY rider_id, chunk_kind, chunk_ix;
    """

    chunks_by_rider: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)

    try:
        with conn.cursor() as cur:
            cur.execute(sql, (rider_ids, int(max_chunks_per_rider)))
            for rider_id, chunk_kind, chunk_ix, chunk_text, _chunk_tokens in cur.fetchall():
                rid = int(rider_id)
                # Simple heuristic score: earlier chunks get slightly higher
                ix = int(chunk_ix) if chunk_ix is not None else len(chunks_by_rider[rid])
                score = 1.0 / (1.0 + ix)

                chunks_by_rider[rid].append(
                    {
                        "score": float(score),
                        "text": chunk_text,
                        "chunk_kind": chunk_kind,
                        "chunk_ix": ix,
                    }
                )
    except Exception as e:
        # Don't permanently mark the table as missing; this could be a transient SQL error.
        logger.warning("Chunks query failed (skipping chunks): %s", e)
        return {rid: [] for rid in rider_ids}

    # Ensure all requested rider ids exist as keys
    return {rid: chunks_by_rider.get(rid, []) for rid in rider_ids}


# -------------------------
# Tools
# -------------------------

@Tool
def search_similar_riders(
    ctx: RunContext,
    query: str,
    top_k_riders: int = 5,
    oversample_factor: int = 10,
    max_chunks_per_rider: int = 3,
    ef_search: Optional[int] = None,
    debug: bool = False,
) -> List[SimilarRider]:
    t0 = time.perf_counter()

    logger.info(
        "search_similar_riders called",
        extra={"top_k_riders": top_k_riders, "oversample_factor": oversample_factor},
    )

    if not query or not query.strip():
        trace_tool(ctx, "search_similar_riders", {"query": query}, {"riders": 0}, t0)
        return []

    deps: PgVectorSearchDeps = _get_pg_deps(ctx)

    qvec = deps.embed_query(query)
    if not qvec:
        logger.warning("search_similar_riders: embed_query returned empty vector.")
        trace_tool(ctx, "search_similar_riders", {"query": query}, {"riders": 0}, t0)
        return []

    qvec_param = _vector_text(qvec)
    candidate_k = max(int(top_k_riders) * int(oversample_factor), int(top_k_riders))

    conn = _connect(deps.database_url)
    try:
        with conn:
            candidates = _fetch_candidates(conn, qvec_param, candidate_k, ef_search)
            if not candidates:
                trace_tool(ctx, "search_similar_riders", {"query": query}, {"riders": 0}, t0)
                return []

            rider_ids = [rid for rid, _dist in candidates]
            dist_map = {rid: dist for rid, dist in candidates}

            rider_map = _fetch_rider_rows(conn, rider_ids)
            chunks_map = _fetch_chunks(
                conn,
                database_url=deps.database_url,
                rider_ids=rider_ids,
                max_chunks_per_rider=max_chunks_per_rider,
            )

            # Fallback: synthesize pseudo-chunks from rider fields if no real chunks returned
            if not any(chunks_map.get(rid) for rid in rider_ids):
                for rid in rider_ids:
                    base = rider_map.get(rid) or {}
                    pseudo: List[Dict[str, Any]] = []

                    bike_txt = _clip_text(base.get("bike"), 800)
                    if bike_txt:
                        pseudo.append({"score": 1.0, "text": f"Bike setup: {bike_txt}", "chunk_kind": "bike", "chunk_ix": 0})

                    key_txt = _clip_text(base.get("key_items"), 800)
                    if key_txt:
                        pseudo.append({"score": 0.95, "text": f"Key items: {key_txt}", "chunk_kind": "key_items", "chunk_ix": 0 if not pseudo else 1})

                    chunks_map[rid] = pseudo

        riders: List[SimilarRider] = []
        for rid in rider_ids:
            base = rider_map.get(rid)
            if not base:
                continue

            dist = dist_map.get(rid)
            if dist is None:
                continue

            payload = dict(base)
            payload["best_score"] = float(1.0 - dist)
            payload["chunks"] = chunks_map.get(rid, [])

            try:
                rider = SimilarRider(**payload)
            except ValidationError as e:
                logger.warning("Skipping invalid rider payload for rider_id=%s: %s", rid, e)
                continue

            if rider.year is None:
                rider.year = _infer_year_from_title(rider.event_title)

            riders.append(rider)

        if not riders:
            trace_tool(ctx, "search_similar_riders", {"query": query}, {"riders": 0}, t0)
            return []

        if riders and (debug or os.getenv("BAIKPACKING_DEBUG_RIDERS") == "1"):
            sample = riders[0].model_dump(exclude_none=True)
            if isinstance(sample.get("chunks"), list):
                sample["chunks"] = sample["chunks"][:1]
            logger.info("sample rider payload: %s", json.dumps(sample, ensure_ascii=False)[:4000])

        try:
            from baikpacking.tools.events import EVENT_KEYWORDS
        except Exception:
            EVENT_KEYWORDS = []

        event_hint = _extract_event_hint(query, EVENT_KEYWORDS)

        def sort_key(r: SimilarRider):
            title_lower = (r.event_title or "").lower()
            same_event = 1 if (event_hint and event_hint in title_lower) else 0
            year = r.year or 0
            score = r.best_score or 0.0
            return (same_event, year, score)

        final = sorted(riders, key=sort_key, reverse=True)[:top_k_riders]

        trace_tool(
            ctx,
            "search_similar_riders",
            {
                "query": query,
                "top_k_riders": top_k_riders,
                "oversample_factor": oversample_factor,
                "max_chunks_per_rider": max_chunks_per_rider,
                "ef_search": ef_search,
            },
            {
                "riders": len(final),
                "riders_with_chunks": sum(1 for r in final if r.chunks),
                "top_rider_ids": [r.rider_id for r in final[:5]],
            },
            t0,
        )
        return final
    finally:
        try:
            conn.close()
        except Exception:
            pass


@Tool
def render_grounding_riders(ctx: RunContext, riders: List[SimilarRider]) -> str:
    t0 = time.perf_counter()

    payload = []
    for r in riders:
        payload.append(
            r.model_dump(
                exclude_none=True,
                exclude={"key_items"},
            )
        )

    out = json.dumps(payload, ensure_ascii=False)
    trace_tool(ctx, "render_grounding_riders", {"riders_len": len(riders)}, {"chars": len(out)}, t0)
    return out