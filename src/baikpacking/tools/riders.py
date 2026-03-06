import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from pydantic import ValidationError
from pydantic_ai import RunContext, Tool

from baikpacking.agents.models import SimilarRider
from baikpacking.tools._trace_utils import trace_tool
from baikpacking.tools.pg_vector_search import PgVectorSearchDeps, _get_deps as _get_pg_deps

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"(19|20)\d{2}")

# Cache table-existence checks per DB URL to avoid repeated checks/warnings.
_RIDER_CHUNKS_EXISTS: Dict[str, bool] = {}

# Query embedding cache across calls.
_QUERY_EMB_CACHE: Dict[str, Sequence[float]] = {}


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
    q = (query or "").lower()
    for key in event_keywords:
        if key in q:
            return key
    return None


def _norm_q(q: str) -> str:
    return " ".join((q or "").lower().split())


def _connect(database_url: str):
    """Prefer psycopg3; fall back to psycopg2."""
    try:
        import psycopg  # type: ignore
        return psycopg.connect(database_url)
    except Exception:
        import psycopg2  # type: ignore
        return psycopg2.connect(database_url)


def _vector_text(vec: Sequence[float]) -> str:
    """Format vector for pgvector input (passed as bound param cast to ::vector)."""
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


def _table_exists(conn, table_name: str) -> bool:
    """Fast existence check using to_regclass."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s) IS NOT NULL;", (f"public.{table_name}",))
            row = cur.fetchone()
            return bool(row[0]) if row is not None else False
    except Exception:
        return False


def _ensure_chunks_table_known(conn, database_url: str) -> bool:
    """Memoized check for rider_chunks existence (per DB url)."""
    exists = _RIDER_CHUNKS_EXISTS.get(database_url)
    if exists is not None:
        return exists

    exists = _table_exists(conn, "rider_chunks")
    _RIDER_CHUNKS_EXISTS[database_url] = exists
    if not exists:
        logger.warning("Table public.rider_chunks does not exist; skipping chunk retrieval.")
    return exists


# -------------------------
# Chunk-first retrieval SQL
# -------------------------

FETCH_TOP_RIDERS_BY_CHUNKS_SQL = """
WITH chunk_hits AS (
  SELECT
    rc.rider_id,
    rc.chunk_kind,
    rc.chunk_ix,
    rc.chunk_text,
    (rc.embedding <=> %(qvec)s::vector)::float AS distance
  FROM public.rider_chunks rc
  WHERE rc.embedding IS NOT NULL
  ORDER BY rc.embedding <=> %(qvec)s::vector
  LIMIT %(top_k_chunks)s
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY rider_id ORDER BY distance ASC) AS rn
  FROM chunk_hits
),
rider_rank AS (
  SELECT
    rider_id,
    MIN(distance) AS best_distance,
    AVG(distance) AS avg_distance,
    COUNT(*) AS n_hits
  FROM chunk_hits
  GROUP BY rider_id
),
top_riders AS (
  SELECT rider_id, best_distance, avg_distance, n_hits
  FROM rider_rank
  ORDER BY best_distance ASC
  LIMIT %(top_k_riders)s
)
SELECT
  tr.rider_id,
  tr.best_distance,
  tr.avg_distance,
  tr.n_hits,
  r.chunk_kind,
  r.chunk_ix,
  r.chunk_text,
  r.distance
FROM top_riders tr
JOIN ranked r ON r.rider_id = tr.rider_id AND r.rn <= %(max_chunks_per_rider)s
ORDER BY tr.best_distance ASC, r.distance ASC;
"""


def _fetch_top_riders_by_chunks(
    conn,
    qvec_param: str,
    top_k_riders: int,
    top_k_chunks: int,
    max_chunks_per_rider: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Returns:
      {
        rider_id: {
          rider_id, best_distance, avg_distance, n_hits,
          chunks: [ {score, text, chunk_kind, chunk_ix}, ... ]
        }
      }
    """
    rows_by_rider: Dict[int, Dict[str, Any]] = {}

    with conn.cursor() as cur:
        cur.execute(
            FETCH_TOP_RIDERS_BY_CHUNKS_SQL,
            {
                "qvec": qvec_param,
                "top_k_chunks": int(top_k_chunks),
                "top_k_riders": int(top_k_riders),
                "max_chunks_per_rider": int(max_chunks_per_rider),
            },
        )

        for (
            rider_id,
            best_distance,
            avg_distance,
            n_hits,
            chunk_kind,
            chunk_ix,
            chunk_text,
            distance,
        ) in cur.fetchall():
            rid = int(rider_id)
            rec = rows_by_rider.get(rid)
            if rec is None:
                rec = {
                    "rider_id": rid,
                    "best_distance": float(best_distance),
                    "avg_distance": float(avg_distance),
                    "n_hits": int(n_hits),
                    "_order": len(rows_by_rider),
                    "chunks": [],
                }
                rows_by_rider[rid] = rec

            rec["chunks"].append(
                {
                    "score": float(1.0 / (1.0 + float(distance))),
                    "text": chunk_text,
                    "chunk_kind": chunk_kind,
                    "chunk_ix": int(chunk_ix),
                }
            )

    return rows_by_rider


# -------------------------
# Rider row hydration
# -------------------------

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
        FROM public.riders r
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


# -------------------------
# Plain implementations
# -------------------------

def run_search_similar_riders(
    *,
    query: str,
    top_k_riders: int = 5,
    oversample_factor: int = 10,   # kept for backward compat, not used in chunk-first path
    max_chunks_per_rider: int = 3,
    ef_search: Optional[int] = None,  # kept for backward compat, not used here
    top_k_chunks: Optional[int] = None,
    debug: bool = False,
    deps: PgVectorSearchDeps,
    trace_ctx: Optional[RunContext] = None,
) -> List[SimilarRider]:
    """
    Plain Python implementation.

    Safe to call from deterministic orchestration code.
    """
    del oversample_factor, ef_search, debug

    t0 = time.perf_counter()
    logger.info("search_similar_riders called", extra={"top_k_riders": top_k_riders})

    if not query or not query.strip():
        if trace_ctx is not None:
            trace_tool(trace_ctx, "search_similar_riders", {"query": query}, {"riders": 0}, t0)
        return []

    qkey = _norm_q(query)
    qvec = _QUERY_EMB_CACHE.get(qkey)
    if qvec is None:
        qvec = deps.embed_query(query)
        if qvec:
            _QUERY_EMB_CACHE[qkey] = qvec

    if not qvec:
        logger.warning("search_similar_riders: embed_query returned empty vector.")
        if trace_ctx is not None:
            trace_tool(trace_ctx, "search_similar_riders", {"query": query}, {"riders": 0}, t0)
        return []

    conn = _connect(deps.database_url)
    try:
        with conn:
            if not _ensure_chunks_table_known(conn, deps.database_url):
                if trace_ctx is not None:
                    trace_tool(trace_ctx, "search_similar_riders", {"query": query}, {"riders": 0}, t0)
                return []

            qvec_param = _vector_text(qvec)

            resolved_top_k_chunks = (
                int(top_k_chunks)
                if top_k_chunks is not None
                else max(50, int(top_k_riders) * 50)
            )

            chunk_rank = _fetch_top_riders_by_chunks(
                conn,
                qvec_param=qvec_param,
                top_k_riders=int(top_k_riders),
                top_k_chunks=resolved_top_k_chunks,
                max_chunks_per_rider=int(max_chunks_per_rider),
            )

            rider_ids = sorted(chunk_rank.keys(), key=lambda rid: chunk_rank[rid]["best_distance"])
            if not rider_ids:
                if trace_ctx is not None:
                    trace_tool(trace_ctx, "search_similar_riders", {"query": query}, {"riders": 0}, t0)
                return []

            rider_map = _fetch_rider_rows(conn, rider_ids)

        riders: List[SimilarRider] = []
        for rid in rider_ids:
            base = rider_map.get(rid)
            if not base:
                continue

            best_distance = float(chunk_rank[rid]["best_distance"])
            best_score = float(1.0 / (1.0 + best_distance))

            payload = dict(base)
            payload["best_score"] = best_score
            payload["chunks"] = [
                {
                    "score": float(c["score"]),
                    "text": c["text"],
                    "chunk_index": int(c["chunk_ix"]),
                }
                for c in chunk_rank[rid]["chunks"]
            ]

            try:
                rider = SimilarRider(**payload)
            except ValidationError as e:
                logger.warning("Skipping invalid rider payload for rider_id=%s: %s", rid, e)
                continue

            if getattr(rider, "year", None) is None:
                rider.year = _infer_year_from_title(getattr(rider, "event_title", None))

            riders.append(rider)

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

        final = sorted(riders, key=sort_key, reverse=True)[: int(top_k_riders)]

        if trace_ctx is not None:
            trace_tool(
                trace_ctx,
                "search_similar_riders",
                {
                    "query": query,
                    "top_k_riders": top_k_riders,
                    "max_chunks_per_rider": max_chunks_per_rider,
                    "top_k_chunks": resolved_top_k_chunks,
                },
                {
                    "riders": len(final),
                    "riders_with_chunks": sum(1 for r in final if getattr(r, "chunks", None)),
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


def run_render_grounding_riders(
    *,
    riders: List[SimilarRider],
    trace_ctx: Optional[RunContext] = None,
) -> str:
    """
    Plain Python implementation.
    Keep this payload reasonably small to reduce LLM latency.
    """
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

    if trace_ctx is not None:
        trace_tool(trace_ctx, "render_grounding_riders", {"riders_len": len(riders)}, {"chars": len(out)}, t0)

    return out


# -------------------------
# Tool wrappers
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
    top_k_chunks: Optional[int] = None,
) -> List[SimilarRider]:
    """
    Agent-exposed wrapper around the plain implementation.
    """
    deps: PgVectorSearchDeps = _get_pg_deps(ctx)
    return run_search_similar_riders(
        query=query,
        top_k_riders=top_k_riders,
        oversample_factor=oversample_factor,
        max_chunks_per_rider=max_chunks_per_rider,
        ef_search=ef_search,
        top_k_chunks=top_k_chunks,
        debug=debug,
        deps=deps,
        trace_ctx=ctx,
    )


@Tool
def render_grounding_riders(ctx: RunContext, riders: List[SimilarRider]) -> str:
    """
    Agent-exposed wrapper around the plain implementation.
    """
    return run_render_grounding_riders(
        riders=riders,
        trace_ctx=ctx,
    )