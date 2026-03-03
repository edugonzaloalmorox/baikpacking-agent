import logging
import os
import time
from typing import Any, Callable, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field
from pydantic_ai import RunContext, Tool

from baikpacking.tools._trace_utils import trace_tool
from baikpacking.tools.call_trace import CallTrace

logger = logging.getLogger(__name__)


class PgVectorHit(BaseModel):
    """Coarse pgvector search hit for routing (NOT grounding)."""
    rider_id: int
    distance: float = Field(..., description="Cosine distance (lower is better).")
    score: Optional[float] = Field(
        None, description="Optional similarity score derived from distance (higher is better)."
    )
    event_title: Optional[str] = None
    article_id: Optional[int] = None
    event_url: Optional[str] = None


class PgVectorSearchDeps(BaseModel):
    """
    Inject via ctx.deps when running the agent.

    embed_query: (text) -> List[float] with same dim as embedding column.
    database_url: Postgres DSN.
    call_trace: optional CallTrace collector for debugging/visualization.
    """
    embed_query: Callable[[str], List[float]]
    database_url: str
    call_trace: Optional[CallTrace] = None


def _get_deps(ctx: RunContext) -> PgVectorSearchDeps:
    deps: Any = getattr(ctx, "deps", None)
    if deps is None:
        raise RuntimeError(
            "pgvector tools require ctx.deps=PgVectorSearchDeps(embed_query=..., database_url=..., call_trace=...)."
        )

    if isinstance(deps, PgVectorSearchDeps):
        return deps

    if isinstance(deps, Mapping):
        embed_query = deps.get("embed_query")
        database_url = deps.get("database_url") or deps.get("db_dsn")
        call_trace = deps.get("call_trace")
    else:
        embed_query = getattr(deps, "embed_query", None)
        database_url = getattr(deps, "database_url", None) or getattr(deps, "db_dsn", None)
        call_trace = getattr(deps, "call_trace", None)

    if not callable(embed_query) or not isinstance(database_url, str) or not database_url:
        raise RuntimeError("ctx.deps must provide .embed_query and .database_url (or .db_dsn).")

    return PgVectorSearchDeps(embed_query=embed_query, database_url=database_url, call_trace=call_trace)


def _connect(database_url: str):
    """Prefer psycopg3; fall back to psycopg2."""
    try:
        import psycopg  #
        return psycopg.connect(database_url)
    except Exception:
        import psycopg2  # type: ignore
        return psycopg2.connect(database_url)


def _vector_text(vec: Sequence[float]) -> str:
    """Format a vector into pgvector text input form (cast to ::vector in SQL)."""
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


def _resolve_riders_cols(cur) -> dict:
    """Returns a mapping of logical -> physical column names that exist in riders."""
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'riders';
        """
    )
    cols = {r[0] for r in cur.fetchall()}

    def pick(*candidates: str) -> Optional[str]:
        for c in candidates:
            if c in cols:
                return c
        return None

    return {
        "event_title": pick("event_title", "event", "event_name", "race", "title"),
        "event_url": pick("event_url", "url", "event_link"),
        "article_id": pick("article_id", "dotwatcher_id"),
    }


@Tool
def pgvector_search_riders(
    ctx: RunContext,
    query: str,
    limit: int = 30,
    ef_search: Optional[int] = None,
) -> List[PgVectorHit]:
    t0 = time.perf_counter()
    deps = _get_deps(ctx)

    if not query or not query.strip():
        trace_tool(ctx, "pgvector_search_riders", {"query": query, "limit": limit}, {"hits": 0}, t0)
        return []

    qvec = deps.embed_query(query)
    if not qvec:
        logger.warning("pgvector_search_riders: embed_query returned empty vector.")
        trace_tool(ctx, "pgvector_search_riders", {"query": query, "limit": limit}, {"hits": 0}, t0)
        return []

    qvec_param = _vector_text(qvec)

    conn = _connect(deps.database_url or os.getenv("DATABASE_URL", ""))
    try:
        with conn:
            with conn.cursor() as cur:
                if ef_search is not None:
                    cur.execute("SET LOCAL hnsw.ef_search = %s;", (int(ef_search),))

                colmap = _resolve_riders_cols(cur)
                event_col = colmap["event_title"]
                url_col = colmap["event_url"]
                article_col = colmap["article_id"]

                select_event = f"r.{event_col} AS event_title" if event_col else "NULL::text AS event_title"
                select_url = f"r.{url_col} AS event_url" if url_col else "NULL::text AS event_url"
                select_article = f"r.{article_col} AS article_id" if article_col else "NULL::int AS article_id"

                sql = f"""
                SELECT
                  re.rider_id AS rider_id,
                  (re.embedding <=> %s::vector)::float AS distance,
                  (1 - (re.embedding <=> %s::vector))::float AS score,
                  {select_event},
                  {select_article},
                  {select_url}
                FROM rider_embeddings re
                LEFT JOIN riders r ON r.id = re.rider_id
                ORDER BY re.embedding <=> %s::vector
                LIMIT %s;
                """

                cur.execute(sql, (qvec_param, qvec_param, qvec_param, int(limit)))
                rows = cur.fetchall()

        hits: List[PgVectorHit] = []
        for rider_id, distance, score, event_title, article_id, event_url in rows:
            hits.append(
                PgVectorHit(
                    rider_id=int(rider_id),
                    distance=float(distance),
                    score=float(score) if score is not None else None,
                    event_title=event_title,
                    article_id=article_id,
                    event_url=event_url,
                )
            )

        trace_tool(
            ctx,
            "pgvector_search_riders",
            {"query": query, "limit": limit, "ef_search": ef_search},
            {"hits": len(hits), "top_rider_ids": [h.rider_id for h in hits[:10]]},
            t0,
        )
        return hits
    finally:
        try:
            conn.close()
        except Exception:
            pass