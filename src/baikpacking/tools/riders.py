import json
import logging
import re
import time
import logfire 
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import ValidationError
from pydantic_ai import RunContext, Tool

from baikpacking.agents.models import SimilarRider
from baikpacking.tools._trace_utils import trace_tool
from baikpacking.tools.pg_vector_search import PgVectorSearchDeps, _get_deps as _get_pg_deps

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"(19|20)\d{2}")

_RIDER_CHUNKS_EXISTS: Dict[str, bool] = {}
_QUERY_EMB_CACHE: Dict[str, Sequence[float]] = {}

_SIMILAR_EVENT_MAP: Dict[str, List[str]] = {
    "transiberica": ["transcontinental race", "transpyrenees", "pan celtic race", "madrid to barcelona"],
    "atlas mountain race": ["gb duro", "silk road mountain race", "badlands"],
    "tour divide": ["highland trail 550", "silk road mountain race"],
    "gb duro": ["atlas mountain race", "highland trail 550"],
    "silk road mountain race": ["atlas mountain race", "tour divide"],
}

_MIN_EXACT_EVENT_RIDERS = 4
_MAX_SIMILAR_EVENTS = 4
_SIMILAR_EVENT_SCORE_WEIGHT = 0.72


# -------------------------------------------------------------------
# Event normalization / matching
# -------------------------------------------------------------------

def _strip_accents(text: str) -> str:
    """
    Convert accented characters to their ASCII base form.

    Examples:
    - TransIbérica -> TransIberica
    - Pyrénées -> Pyrenees
    - Liège -> Liege
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_event_text(text: Optional[str]) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    text = _strip_accents(text)
    text = text.lower()

    # Remove years.
    text = _YEAR_RE.sub(" ", text)

    # Normalize some separators / punctuation to spaces.
    text = re.sub(r"[’'`´]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)

    return " ".join(text.split())

def _event_titles_match(a: Optional[str], b: Optional[str]) -> bool:
    na = _normalize_event_text(a)
    nb = _normalize_event_text(b)
    if not na or not nb:
        return False

    if na == nb:
        return True

    ta = set(na.split())
    tb = set(nb.split())

    if not ta or not tb:
        return False

    # exact token-set equality
    if ta == tb:
        return True

    # one title contains the other's tokens
    if tb.issubset(ta):
        return True
    if ta.issubset(tb):
        return True

    # fallback phrase containment
    if nb in na or na in nb:
        return True

    return False

def _is_exact_event_title(title: Optional[str], event_hint: Optional[str]) -> bool:
    nt = _normalize_event_text(title)
    ne = _normalize_event_text(event_hint)

    if not nt or not ne:
        return False

    if nt == ne:
        return True

    title_tokens = nt.split()
    event_tokens = ne.split()

    if not title_tokens or not event_tokens:
        return False

    # event phrase appears contiguously anywhere in title tokens
    for i in range(0, len(title_tokens) - len(event_tokens) + 1):
        if title_tokens[i:i + len(event_tokens)] == event_tokens:
            return True

    return False


def _token_set(text: Optional[str]) -> set[str]:
    norm = _normalize_event_text(text)
    return set(norm.split()) if norm else set()


def _token_overlap_score(a: Optional[str], b: Optional[str]) -> float:
    sa = _token_set(a)
    sb = _token_set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _norm_q(q: str) -> str:
    return " ".join((q or "").lower().split())


def _infer_surface_bias(query: str, requested_event_hint: Optional[str] = None) -> str:
    q = _normalize_event_text(query)

    road_terms = [
        "road race",
        "road ultra",
        "endurance road",
        "road bikepacking",
        "all road",
        "transiberica",
    ]
    gravel_terms = [
        "gravel",
        "gravel race",
        "off road",
        "singletrack",
        "mtb",
        "mountain bike",
    ]

    road_hits = sum(1 for t in road_terms if t in q)
    gravel_hits = sum(1 for t in gravel_terms if t in q)

    if requested_event_hint and _normalize_event_text(requested_event_hint) == "transiberica":
        road_hits += 3

    if road_hits > gravel_hits:
        return "road"
    if gravel_hits > road_hits:
        return "gravel"
    return "neutral"


def _surface_penalty_or_boost(r: SimilarRider, surface_bias: str) -> float:
    """
    Multiplicative factor on rider score.
    >1 boosts, <1 penalizes.
    """
    if surface_bias == "neutral":
        return 1.0

    text = _join_rider_source_text(r).lower()

    road_markers = [
        "road bike",
        "endurance bike",
        "endurace",
        "domane",
        "emonda",
        "700c",
        "28mm",
        "30mm",
        "32mm",
        "slick",
    ]
    gravel_markers = [
        "gravel",
        "grx",
        "650b",
        "hardtail",
        "mtb",
        "mountain bike",
        "29er",
        "27.5",
        "suspension",
    ]

    has_road = any(x in text for x in road_markers)
    has_gravel = any(x in text for x in gravel_markers)

    if surface_bias == "road":
        if has_road and not has_gravel:
            return 1.15
        if has_gravel and not has_road:
            return 0.72
        if has_gravel and has_road:
            return 0.92
        return 1.0

    if surface_bias == "gravel":
        if has_gravel and not has_road:
            return 1.15
        if has_road and not has_gravel:
            return 0.80
        if has_gravel and has_road:
            return 0.95
        return 1.0

    return 1.0


def _is_exact_event_rider(r: SimilarRider, requested_event_hint: Optional[str]) -> bool:
    return bool(
        requested_event_hint
        and _is_exact_event_title(getattr(r, "event_title", None), requested_event_hint)
    )


def _get_cached_embedding(embed_fn, text: str) -> Optional[Sequence[float]]:
    key = _norm_q(text)
    vec = _QUERY_EMB_CACHE.get(key)
    if vec is not None:
        return vec
    vec = embed_fn(text)
    if vec:
        _QUERY_EMB_CACHE[key] = vec
    return vec


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = 0.0
    na = 0.0
    nb = 0.0
    for xa, xb in zip(a, b):
        fa = float(xa)
        fb = float(xb)
        dot += fa * fb
        na += fa * fa
        nb += fb * fb

    if na <= 0.0 or nb <= 0.0:
        return 0.0

    return dot / ((na ** 0.5) * (nb ** 0.5))


def _event_metadata_text(article: Dict[str, Any]) -> str:
    parts = [
        article.get("title"),
        article.get("url"),
    ]
    return " | ".join(str(x).strip() for x in parts if isinstance(x, str) and x.strip())


def _connect(database_url: str):
    """Prefer psycopg3; fall back to psycopg2."""
    try:
        import psycopg  # type: ignore
        return psycopg.connect(database_url)
    except Exception:
        import psycopg2  # type: ignore
        return psycopg2.connect(database_url)


def _vector_text(vec: Sequence[float]) -> str:
    """Format vector for pgvector input."""
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


def _split_key_items(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [p.strip() for p in re.split(r"[|,]", value) if p.strip()]
    text = str(value).strip()
    return [text] if text else []


def _ensure_chunks_table_known(conn, database_url: str) -> bool:
    """Memoized check for rider_chunks existence (per DB URL)."""
    exists = _RIDER_CHUNKS_EXISTS.get(database_url)
    if exists is not None:
        return exists

    exists = _table_exists(conn, "rider_chunks")
    _RIDER_CHUNKS_EXISTS[database_url] = exists
    if not exists:
        logger.warning("Table public.rider_chunks does not exist; skipping chunk retrieval.")
    return exists


def _fetch_event_articles(conn) -> List[Dict[str, Any]]:
    """
    Return article rows that actually have riders.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT a.id, a.title, a.url
            FROM public.articles a
            JOIN public.riders r
              ON r.article_id = a.id
            WHERE a.title IS NOT NULL
              AND btrim(a.title) <> ''
            ORDER BY a.id;
            """
        )
        rows = cur.fetchall()

    return [
        {"article_id": int(article_id), "title": title, "url": url}
        for article_id, title, url in rows
        if title
    ]


def _extract_event_hint_from_query(query: str, known_event_names: List[str]) -> Optional[str]:
    """
    Prefer a DB-driven event hint over a static keyword list.
    """
    nq = _normalize_event_text(query)
    if not nq:
        return None

    ordered = sorted(set(known_event_names), key=len, reverse=True)

    for name in ordered:
        nn = _normalize_event_text(name)
        if nn and nn in nq:
            return name

    return None


def _find_matching_articles(conn, event_hint: str) -> List[Dict[str, Any]]:
    if not event_hint:
        return []

    matches: List[Dict[str, Any]] = []
    articles = _fetch_event_articles(conn)

    for article in articles:
        if _is_exact_event_title(article["title"], event_hint):
            matches.append(article)

    return matches


def _find_similar_articles(conn, event_hint: str) -> List[Dict[str, Any]]:
    if not event_hint:
        return []

    normalized_hint = _normalize_event_text(event_hint)
    similar_targets = _SIMILAR_EVENT_MAP.get(normalized_hint, [])
    if not similar_targets:
        return []

    matches: List[Dict[str, Any]] = []
    articles = _fetch_event_articles(conn)

    for article in articles:
        title = article["title"]
        if any(_event_titles_match(title, candidate) for candidate in similar_targets):
            matches.append(article)

    seen = set()
    out = []
    for article in matches:
        aid = article["article_id"]
        if aid in seen:
            continue
        seen.add(aid)
        out.append(article)

    return out

def _classify_event_titles(conn, event_hint: str) -> Dict[str, List[str]]:
    articles = _fetch_event_articles(conn)

    exact_titles = []
    related_titles = []

    for article in articles:
        title = article["title"]
        if _is_exact_event_title(title, event_hint):
            exact_titles.append(title)
        elif _event_titles_match(title, event_hint):
            related_titles.append(title)

    return {
        "exact_titles": sorted(set(exact_titles)),
        "related_titles": sorted(set(related_titles)),
    }


def _count_distinct_riders_for_articles(conn, article_ids: List[int]) -> int:
    if not article_ids:
        return 0

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(DISTINCT r.id)
            FROM public.riders r
            WHERE r.article_id = ANY(%s::int[]);
            """,
            (article_ids,),
        )
        row = cur.fetchone()

    return int(row[0]) if row and row[0] is not None else 0


def _rank_similar_event_articles(
    *,
    articles: List[Dict[str, Any]],
    requested_event_hint: str,
    query: str,
    embed_query_fn,
    max_similar_events: int = _MAX_SIMILAR_EVENTS,
) -> List[Dict[str, Any]]:
    """
    Rank fallback events using event-level metadata.

    Signals:
    - curated similar-event prior
    - normalized title/token overlap
    - embedding similarity between query and event metadata text
    """
    if not requested_event_hint:
        return []

    normalized_hint = _normalize_event_text(requested_event_hint)
    curated_targets = {
        _normalize_event_text(x)
        for x in _SIMILAR_EVENT_MAP.get(normalized_hint, [])
        if _normalize_event_text(x)
    }

    query_vec = _get_cached_embedding(embed_query_fn, query)
    ranked: List[Tuple[float, Dict[str, Any]]] = []

    for article in articles:
        title = article.get("title")
        if not title:
            continue

        if _event_titles_match(title, requested_event_hint):
            continue

        lexical = _token_overlap_score(title, requested_event_hint)

        curated = 0.0
        if curated_targets and any(_event_titles_match(title, target) for target in curated_targets):
            curated = 1.0

        meta_text = _event_metadata_text(article)
        meta_vec = _get_cached_embedding(embed_query_fn, meta_text) if query_vec else None
        semantic = _cosine_similarity(query_vec, meta_vec) if (query_vec and meta_vec) else 0.0

        score = (0.50 * curated) + (0.30 * semantic) + (0.20 * lexical)
        if score <= 0.0:
            continue

        ranked.append((score, article))

    ranked.sort(key=lambda x: x[0], reverse=True)

    out: List[Dict[str, Any]] = []
    seen_titles: set[str] = set()

    for score, article in ranked:
        title = article["title"]
        nt = _normalize_event_text(title)
        if nt in seen_titles:
            continue
        seen_titles.add(nt)

        item = dict(article)
        item["similarity_score"] = float(score)
        out.append(item)

        if len(out) >= int(max_similar_events):
            break

    return out


def _ground_event_scope(
    conn,
    *,
    query: str,
    event_hint: Optional[str],
    embed_query_fn,
    min_exact_riders: int = _MIN_EXACT_EVENT_RIDERS,
    max_similar_events: int = _MAX_SIMILAR_EVENTS,
) -> Dict[str, Any]:
    """
    Two-step event grounding.

    1. Resolve the requested event from DB article titles.
    2. Use exact-event articles first.
    3. Only add similar-event fallback when exact event is missing or under-covered.
    """
    articles = _fetch_event_articles(conn)
    known_event_names = [a["title"] for a in articles if a.get("title")]

    requested_event_hint = _extract_event_hint_from_query(query, known_event_names) or event_hint

    exact_articles: List[Dict[str, Any]] = []
    similar_articles: List[Dict[str, Any]] = []

    if requested_event_hint:
        exact_articles = _find_matching_articles(conn, requested_event_hint)

        if not exact_articles and event_hint and requested_event_hint != event_hint:
            exact_articles = _find_matching_articles(conn, event_hint)
            if exact_articles:
                requested_event_hint = event_hint

    exact_article_ids = [a["article_id"] for a in exact_articles]
    exact_rider_count = _count_distinct_riders_for_articles(conn, exact_article_ids)

    needs_similar_fallback = bool(requested_event_hint) and (
        not exact_articles or exact_rider_count < int(min_exact_riders)
    )

    if needs_similar_fallback:
        similar_articles = _rank_similar_event_articles(
            articles=articles,
            requested_event_hint=requested_event_hint,
            query=query,
            embed_query_fn=embed_query_fn,
            max_similar_events=max_similar_events,
        )

    if exact_articles and exact_rider_count >= int(min_exact_riders):
        retrieval_scope = "exact_event"
    elif exact_articles:
        retrieval_scope = "exact_plus_similar"
    elif similar_articles:
        retrieval_scope = "similar_events"
    else:
        retrieval_scope = "global"

    return {
        "requested_event_hint": requested_event_hint,
        "exact_articles": exact_articles,
        "similar_articles": similar_articles,
        "exact_event_titles": [a["title"] for a in exact_articles if a.get("title")],
        "similar_event_titles": [a["title"] for a in similar_articles if a.get("title")],
        "exact_rider_count": exact_rider_count,
        "needs_similar_fallback": needs_similar_fallback,
        "retrieval_scope": retrieval_scope,
    }


# -------------------------------------------------------------------
# Text enrichment helpers
# -------------------------------------------------------------------

def _join_rider_source_text(r: SimilarRider) -> str:
    parts: List[str] = []

    primary_fields = [
        r.bike,
        " | ".join(r.key_items) if r.key_items else None,
        r.bike_type,
        r.wheels,
        r.tyres,
        r.drivetrain,
        r.bags,
        r.sleep_system,
        r.wheel_size,
        r.tyre_width,
        r.frame_type,
        r.frame_material,
    ]

    for value in primary_fields:
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    for chunk in r.chunks or []:
        text = getattr(chunk, "text", None) or ""
        if text:
            parts.append(text)

    return " | ".join(parts)


def _extract_setup_hints(text: str) -> Dict[str, Optional[str]]:
    source = (text or "").strip()
    tl = source.lower()

    out: Dict[str, Optional[str]] = {
        "bike_type": None,
        "wheels": None,
        "tyres": None,
        "drivetrain": None,
        "bags": None,
        "sleep_system": None,
    }

    if "hardtail" in tl:
        out["bike_type"] = "hardtail"
    elif "mountain bike" in tl or " mtb" in tl:
        out["bike_type"] = "mtb"
    elif "gravel" in tl:
        out["bike_type"] = "gravel bike"
    elif "road bike" in tl or "endurace" in tl or "domane" in tl or "emonda" in tl:
        out["bike_type"] = "road / endurance bike"

    wheel_patterns = [
        r"\benve [^|,;]+wheels?\b",
        r"\bninth wave [^|,;]+wheels?\b",
        r"\bzipp [^|,;]+wheels?\b",
        r"\b650b\b",
        r"\b700c\b",
        r"\b29(?:er)?\b",
        r"\b27\.5\b",
    ]
    for pat in wheel_patterns:
        m = re.search(pat, tl)
        if m:
            out["wheels"] = source[m.start():m.end()]
            break

    tyre_patterns = [
        r"\bschwalbe [^|,;]+\b",
        r"\btufo [^|,;]+\b",
        r"\brene herse [^|,;]+\b",
        r"\bconti(?:nental)? [^|,;]+\b",
        r"\b\d{2}\s?mm\b",
        r"\b650x\d+[a-z]?c?\b",
        r"\b700x\d+[a-z]?c?\b",
    ]
    for pat in tyre_patterns:
        m = re.search(pat, tl)
        if m:
            out["tyres"] = source[m.start():m.end()]
            break

    drivetrain_patterns = [
        r"\bshimano [^|,;]+",
        r"\bsram [^|,;]+",
        r"\bgrx [^|,;]+",
        r"\bxtr [^|,;]+",
        r"\b105 rd-r\d+[^\|,;]*",
        r"\b\d{1,2}\s?x\s?\d{2}(?:-\d{2})?\b",
        r"\b\d{2}-\d{2}\/\d{2}-\d{2}\b",
    ]
    for pat in drivetrain_patterns:
        m = re.search(pat, tl)
        if m:
            out["drivetrain"] = source[m.start():m.end()]
            break

    bag_patterns = [
        r"\bapidura [^|,;]*bags?\b",
        r"\brestrap [^|,;]*bags?\b",
        r"\btailfin [^|,;]*bags?\b",
        r"\bortlieb [^|,;]*bags?\b",
        r"\brevelate [^|,;]*bags?\b",
        r"\bframe bag\b",
        r"\bsaddle bag\b",
        r"\bsaddlebag\b",
        r"\bseat pack\b",
        r"\btop tube bag\b",
        r"\bhandlebar bag\b",
        r"\bdrybag\b",
        r"\bcargo pack\b",
    ]
    for pat in bag_patterns:
        m = re.search(pat, tl)
        if m:
            out["bags"] = source[m.start():m.end()]
            break

    sleep_patterns = [
        r"\byeti sleeping bag\b",
        r"\bsleeping bag\b",
        r"\bair mattress\b",
        r"\bmattress\b",
        r"\bquilt\b",
        r"\btent\b",
        r"\bbivy\b",
        r"\bbivvy\b",
        r"\bbivi\b",
        r"\bmat\b",
    ]
    for pat in sleep_patterns:
        m = re.search(pat, tl)
        if m:
            out["sleep_system"] = source[m.start():m.end()]
            break

    return out


def _enrich_rider_from_text(r: SimilarRider) -> SimilarRider:
    hints = _extract_setup_hints(_join_rider_source_text(r))

    if not r.bike_type and hints["bike_type"]:
        r.bike_type = hints["bike_type"]
    if not r.wheels and hints["wheels"]:
        r.wheels = hints["wheels"]
    if not r.tyres and hints["tyres"]:
        r.tyres = hints["tyres"]
    if not r.drivetrain and hints["drivetrain"]:
        r.drivetrain = hints["drivetrain"]
    if not r.bags and hints["bags"]:
        r.bags = hints["bags"]
    if not r.sleep_system and hints["sleep_system"]:
        r.sleep_system = hints["sleep_system"]

    return r


# -------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------

def _infer_year_from_title(title: Optional[str]) -> Optional[int]:
    """Infer year from a title like 'Transcontinental No10 2024'."""
    if not title:
        return None
    m = _YEAR_RE.search(title)
    return int(m.group(0)) if m else None


def _extract_event_hint(query: str, event_keywords: List[str]) -> Optional[str]:
    nq = _normalize_event_text(query)
    if not nq:
        return None

    for key in event_keywords:
        nk = _normalize_event_text(key)
        if nk and nk in nq:
            return key

    return None

def _fetch_all_articles(conn) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT a.id, a.title, a.url
            FROM public.articles a
            WHERE a.title IS NOT NULL
              AND btrim(a.title) <> ''
            ORDER BY a.id;
            """
        )
        rows = cur.fetchall()

    return [
        {"article_id": int(article_id), "title": title, "url": url}
        for article_id, title, url in rows
        if title
    ]


def _debug_exact_event_title_presence(conn, event_hint: str) -> Dict[str, Any]:
    all_articles = _fetch_all_articles(conn)
    rider_articles = _fetch_event_articles(conn)

    all_exact = [a["title"] for a in all_articles if _is_exact_event_title(a["title"], event_hint)]
    rider_exact = [a["title"] for a in rider_articles if _is_exact_event_title(a["title"], event_hint)]

    return {
        "event_hint": event_hint,
        "all_articles_exact_titles": sorted(set(all_exact)),
        "rider_backed_exact_titles": sorted(set(rider_exact)),
        "all_articles_exact_count": len(set(all_exact)),
        "rider_backed_exact_count": len(set(rider_exact)),
    }


# -------------------------------------------------------------------
# Chunk retrieval SQL
# -------------------------------------------------------------------

FETCH_TOP_RIDERS_BY_CHUNKS_FOR_EVENT_TITLES_SQL = """
WITH event_riders AS (
  SELECT r.id
  FROM public.riders r
  JOIN public.articles a
    ON a.id = r.article_id
  WHERE a.title = ANY(%(event_titles)s::text[])
),
chunk_hits AS (
  SELECT
    rc.rider_id,
    rc.chunk_kind,
    rc.chunk_ix,
    rc.chunk_text,
    (rc.embedding <=> %(qvec)s::vector)::float AS distance
  FROM public.rider_chunks rc
  JOIN event_riders er
    ON er.id = rc.rider_id
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
JOIN ranked r
  ON r.rider_id = tr.rider_id
 AND r.rn <= %(max_chunks_per_rider)s
ORDER BY tr.best_distance ASC, r.distance ASC;
"""

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

FETCH_TOP_CHUNKS_FOR_RIDER_IDS_SQL = """
WITH chunk_hits AS (
  SELECT
    rc.rider_id,
    rc.chunk_kind,
    rc.chunk_ix,
    rc.chunk_text,
    (rc.embedding <=> %(qvec)s::vector)::float AS distance
  FROM public.rider_chunks rc
  WHERE rc.embedding IS NOT NULL
    AND rc.rider_id = ANY(%(rider_ids)s::int[])
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
)
SELECT
  rr.rider_id,
  rr.best_distance,
  rr.avg_distance,
  rr.n_hits,
  r.chunk_kind,
  r.chunk_ix,
  r.chunk_text,
  r.distance
FROM rider_rank rr
JOIN ranked r
  ON r.rider_id = rr.rider_id
 AND r.rn <= %(max_chunks_per_rider)s
ORDER BY rr.best_distance ASC, r.distance ASC;
"""


def _rows_to_chunk_rank(rows: List[tuple]) -> Dict[int, Dict[str, Any]]:
    rows_by_rider: Dict[int, Dict[str, Any]] = {}

    for (
        rider_id,
        best_distance,
        avg_distance,
        n_hits,
        chunk_kind,
        chunk_ix,
        chunk_text,
        distance,
    ) in rows:
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


def _fetch_top_chunks_for_rider_ids(
    conn,
    qvec_param: str,
    rider_ids: List[int],
    max_chunks_per_rider: int,
) -> Dict[int, Dict[str, Any]]:
    if not rider_ids:
        return {}

    with conn.cursor() as cur:
        cur.execute(
            FETCH_TOP_CHUNKS_FOR_RIDER_IDS_SQL,
            {
                "qvec": qvec_param,
                "rider_ids": rider_ids,
                "max_chunks_per_rider": int(max_chunks_per_rider),
            },
        )
        return _rows_to_chunk_rank(cur.fetchall())



def _fetch_top_riders_by_chunks_for_event_titles(
    conn,
    qvec_param: str,
    event_titles: List[str],
    top_k_riders: int,
    top_k_chunks: int,
    max_chunks_per_rider: int,
) -> Dict[int, Dict[str, Any]]:
    if not event_titles:
        return {}

    with conn.cursor() as cur:
        cur.execute(
            FETCH_TOP_RIDERS_BY_CHUNKS_FOR_EVENT_TITLES_SQL,
            {
                "qvec": qvec_param,
                "event_titles": event_titles,
                "top_k_chunks": int(top_k_chunks),
                "top_k_riders": int(top_k_riders),
                "max_chunks_per_rider": int(max_chunks_per_rider),
            },
        )
        return _rows_to_chunk_rank(cur.fetchall())


def _fetch_top_riders_by_chunks(
    conn,
    qvec_param: str,
    top_k_riders: int,
    top_k_chunks: int,
    max_chunks_per_rider: int,
) -> Dict[int, Dict[str, Any]]:
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
        return _rows_to_chunk_rank(cur.fetchall())


def _merge_chunk_ranks(
    exact_chunk_rank: Dict[int, Dict[str, Any]],
    similar_chunk_rank: Dict[int, Dict[str, Any]],
    similar_weight: float = _SIMILAR_EVENT_SCORE_WEIGHT,
) -> Dict[int, Dict[str, Any]]:
    """
    Merge exact-event and fallback similar-event candidates.

    Similar-event riders are downweighted so that exact-event riders keep priority.
    """
    out: Dict[int, Dict[str, Any]] = {}

    for rider_id, rec in exact_chunk_rank.items():
        clone = dict(rec)
        clone["source_scope"] = "exact_event"
        clone["weighted_best_distance"] = float(rec["best_distance"])
        clone["weighted_best_score"] = float(1.0 / (1.0 + float(rec["best_distance"])))
        out[rider_id] = clone

    for rider_id, rec in similar_chunk_rank.items():
        if rider_id in out:
            continue

        base_score = float(1.0 / (1.0 + float(rec["best_distance"])))
        weighted_score = base_score * float(similar_weight)

        clone = dict(rec)
        clone["source_scope"] = "similar_events"
        clone["weighted_best_score"] = weighted_score
        clone["weighted_best_distance"] = float((1.0 / weighted_score) - 1.0) if weighted_score > 0 else 999999.0
        out[rider_id] = clone

    return out


def _has_enough_evidence(riders: List[SimilarRider], query_component: Optional[str]) -> bool:
    if not riders:
        return False

    if query_component and query_component != "full_setup":
        return len(riders) >= 2

    covered = 0
    for rider in riders:
        n = sum(
            1
            for value in [
                rider.bike_type,
                rider.wheels,
                rider.tyres,
                rider.drivetrain,
                rider.bags,
                rider.sleep_system,
                rider.bike,
            ]
            if isinstance(value, str) and value.strip()
        )
        if n >= 2:
            covered += 1

    return covered >= 2


# -------------------------------------------------------------------
# Rider row hydration
# -------------------------------------------------------------------

def _resolve_riders_cols(cur) -> Dict[str, Optional[str]]:
    """
    Resolve riders column names only for fields that truly belong to riders.
    """
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
        "name": pick("name"),
        "article_id": pick("article_id", "dotwatcher_id"),
        "frame_type": pick("frame_type"),
        "frame_material": pick("frame_material"),
        "wheel_size": pick("wheel_size"),
        "tyre_width": pick("tyre_width"),
        "electronic_shifting": pick("electronic_shifting"),
        "bike": pick("bike"),
        "bike_type": pick("bike_type", "bike_platform"),
        "wheels": pick("wheels", "wheelset"),
        "tyres": pick("tyres", "tires", "tire_setup"),
        "drivetrain": pick("drivetrain", "groupset", "gearing"),
        "bags": pick("bags", "bag_setup", "luggage"),
        "sleep_system": pick("sleep_system", "sleep", "sleep_setup"),
        "key_items": pick("key_items", "keyitems", "kit", "gear"),
    }


def _resolve_articles_cols(cur) -> Dict[str, Optional[str]]:
    """
    Resolve article-side columns that provide event metadata.
    """
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'articles';
        """
    )
    cols = {r[0] for r in cur.fetchall()}

    def pick(*cands: str) -> Optional[str]:
        for c in cands:
            if c in cols:
                return c
        return None

    return {
        "id": pick("id"),
        "title": pick("title"),
        "url": pick("url"),
    }


def _fetch_rider_ids_for_event_titles(conn, event_titles: List[str]) -> List[int]:
    if not event_titles:
        return []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT r.id
            FROM public.riders r
            JOIN public.articles a
              ON a.id = r.article_id
            WHERE a.title = ANY(%s::text[])
            ORDER BY r.id;
            """,
            (event_titles,),
        )
        return [int(row[0]) for row in cur.fetchall()]




def _fetch_rider_rows(conn, rider_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Hydrate SimilarRider core fields from public.riders joined to public.articles.

    event_title <- articles.title
    event_url   <- articles.url
    """
    if not rider_ids:
        return {}

    def _select_expr(table_alias: str, column_name: Optional[str], alias: str, null_sql: str) -> str:
        if column_name:
            return f"{table_alias}.{column_name} AS {alias}"
        return f"{null_sql} AS {alias}"

    rider_field_specs = [
        ("article_id", "article_id", "NULL::int"),
        ("frame_type", "frame_type", "NULL::text"),
        ("frame_material", "frame_material", "NULL::text"),
        ("wheel_size", "wheel_size", "NULL::text"),
        ("tyre_width", "tyre_width", "NULL::text"),
        ("electronic_shifting", "electronic_shifting", "NULL::boolean"),
        ("bike", "bike", "NULL::text"),
        ("bike_type", "bike_type", "NULL::text"),
        ("wheels", "wheels", "NULL::text"),
        ("tyres", "tyres", "NULL::text"),
        ("drivetrain", "drivetrain", "NULL::text"),
        ("bags", "bags", "NULL::text"),
        ("sleep_system", "sleep_system", "NULL::text"),
        ("key_items", "key_items", "NULL::text"),
    ]

    out: Dict[int, Dict[str, Any]] = {}

    with conn.cursor() as cur:
        rider_cols = _resolve_riders_cols(cur)
        article_cols = _resolve_articles_cols(cur)

        if not rider_cols["name"]:
            raise RuntimeError("riders table must contain a 'name' column (or equivalent).")

        select_parts = [
            "r.id AS rider_id",
            f"r.{rider_cols['name']} AS name",
            *[
                _select_expr("r", rider_cols.get(source_key), alias, null_sql)
                for source_key, alias, null_sql in rider_field_specs
            ],
            _select_expr("a", article_cols.get("title"), "event_title", "NULL::text"),
            _select_expr("a", article_cols.get("url"), "event_url", "NULL::text"),
            "NULL::text AS event_key",
        ]

        sql = f"""
        SELECT
          {", ".join(select_parts)}
        FROM public.riders r
        LEFT JOIN public.articles a
          ON a.id = r.article_id
        WHERE r.id = ANY(%s::int[]);
        """

        cur.execute(sql, (rider_ids,))
        aliases = [
            "rider_id",
            "name",
            *[alias for _, alias, _ in rider_field_specs],
            "event_title",
            "event_url",
            "event_key",
        ]

        for row in cur.fetchall():
            data = dict(zip(aliases, row))
            rid = int(data["rider_id"])

            article_id = data.get("article_id")
            data["article_id"] = int(article_id) if article_id is not None else None
            data["rider_id"] = rid
            data["key_items"] = _split_key_items(data.get("key_items"))

            out[rid] = data

    return out



# -------------------------------------------------------------------
# Plain implementations
# -------------------------------------------------------------------

def run_search_similar_riders(
    *,
    query: str,
    query_component: Optional[str] = None,
    component_terms: Optional[List[str]] = None,
    top_k_riders: int = 5,
    oversample_factor: int = 10,
    max_chunks_per_rider: int = 3,
    ef_search: Optional[int] = None,
    top_k_chunks: Optional[int] = None,
    debug: bool = False,
    deps: PgVectorSearchDeps,
    trace_ctx: Optional[RunContext] = None,
) -> List[SimilarRider]:
    """
    Safe to call from deterministic orchestration code.

    Retrieval strategy:
    1. Ground the query to an event.
    2. Prefer exact-event riders.
    3. Add similar-event fallback only when exact coverage is missing or thin.
    4. Avoid global fallback when a named event was requested but no scoped matches exist.
    """
    del oversample_factor, ef_search, debug

    
    def _event_subtype_from_query(query: str) -> Optional[str]:
        q = f" {_normalize_event_text(query)} "
        if " road " in q:
            return "road"
        if " gravel " in q:
            return "gravel"
        if " trail " in q:
            return "trail"
        return None


    def _event_subtype_bonus(
        event_title: Optional[str],
        requested_event_hint: Optional[str],
        query: str,
    ) -> float:
            title = f" {_normalize_event_text(event_title)} "
            hint = f" {_normalize_event_text(requested_event_hint)} "
            subtype = _event_subtype_from_query(query)

            if not title.strip():
                return 0.0

            bonus = 0.0

            # Family-level boost
            if " granguanche " in title and " granguanche " in hint:
                bonus += 0.08

            # Requested subtype boost
            if subtype == "road":
                if " road " in title:
                    bonus += 0.16
                elif " gravel " in title:
                    bonus -= 0.04
                elif " trail " in title:
                    bonus -= 0.08
            elif subtype == "gravel":
                if " gravel " in title:
                    bonus += 0.16
                elif " trail " in title:
                    bonus += 0.04
                elif " road " in title:
                    bonus -= 0.05
            elif subtype == "trail":
                if " trail " in title:
                    bonus += 0.16
                elif " gravel " in title:
                    bonus += 0.04
                elif " road " in title:
                    bonus -= 0.08

            return bonus
    
    
    def _is_known_event_title(title: Optional[str]) -> int:
        if not title:
            return 0
        t = title.strip().lower()
        return 0 if not t or t == "unknown event" else 1

    def _setup_field_coverage(r: SimilarRider) -> int:
        fields = [
            getattr(r, "bike", None),
            getattr(r, "bike_type", None),
            getattr(r, "wheels", None),
            getattr(r, "tyres", None),
            getattr(r, "drivetrain", None),
            getattr(r, "bags", None),
            getattr(r, "sleep_system", None),
            getattr(r, "wheel_size", None),
            getattr(r, "tyre_width", None),
        ]
        coverage = sum(1 for v in fields if isinstance(v, str) and v.strip())
        if getattr(r, "key_items", None):
            coverage += 1
        return coverage

    def _chunk_text_len(r: SimilarRider) -> int:
        return sum(len(getattr(c, "text", "") or "") for c in (getattr(r, "chunks", None) or []))

    def _component_hit(r: SimilarRider, terms: List[str]) -> int:
        if not terms:
            return 0
        searchable = _join_rider_source_text(r).lower()
        return 1 if any(term in searchable for term in terms) else 0
    
    

    def _effective_component_terms(
        query_component: Optional[str],
        component_terms: Optional[List[str]],
    ) -> List[str]:
        if not query_component or query_component == "full_setup":
            return []
        return [t.strip().lower() for t in (component_terms or []) if isinstance(t, str) and t.strip()]

    def _build_riders_from_chunk_rank(
        rider_ids: List[int],
        rider_map: Dict[int, Dict[str, Any]],
        chunk_rank: Dict[int, Dict[str, Any]],
    ) -> List[SimilarRider]:
        riders: List[SimilarRider] = []
        invalid_payloads = 0

        for rid in rider_ids:
            base = rider_map.get(rid)
            if not base:
                continue

            rec = chunk_rank.get(rid)
            if not rec:
                continue

            best_score = float(
                rec.get("weighted_best_score", 1.0 / (1.0 + float(rec["best_distance"])))
            )

            payload = dict(base)
            payload["best_score"] = best_score
            payload["chunks"] = [
                {
                    "score": float(c["score"]),
                    "text": c["text"],
                    "chunk_index": int(c["chunk_ix"]),
                }
                for c in rec.get("chunks", [])
            ]

            try:
                rider = SimilarRider(**payload)
                rider = _enrich_rider_from_text(rider)
            except ValidationError as e:
                invalid_payloads += 1
                logger.warning("Skipping invalid rider payload for rider_id=%s: %s", rid, e)
                continue

            if getattr(rider, "year", None) is None:
                rider.year = _infer_year_from_title(getattr(rider, "event_title", None))

            setattr(rider, "_source_scope", rec.get("source_scope", "global"))
            riders.append(rider)

        logfire.info(
            "built similar riders",
            rider_count=len(riders),
            invalid_payloads=invalid_payloads,
        )
        return riders

    t0 = time.perf_counter()
    effective_component_terms = _effective_component_terms(query_component, component_terms)

    with logfire.span(
        "tool.search_similar_riders",
        query=query,
        query_component=query_component,
        component_terms=effective_component_terms,
        top_k_riders=top_k_riders,
        max_chunks_per_rider=max_chunks_per_rider,
        top_k_chunks=top_k_chunks,
    ):
        if not query or not query.strip():
            if trace_ctx is not None:
                trace_tool(
                    trace_ctx,
                    "search_similar_riders",
                    {
                        "query": query,
                        "query_component": query_component,
                        "component_terms": effective_component_terms,
                    },
                    {"riders": 0},
                    t0,
                )
            return []

        qvec = _get_cached_embedding(deps.embed_query, query)
        if not qvec:
            if trace_ctx is not None:
                trace_tool(
                    trace_ctx,
                    "search_similar_riders",
                    {
                        "query": query,
                        "query_component": query_component,
                        "component_terms": effective_component_terms,
                    },
                    {"riders": 0},
                    t0,
                )
            return []

        try:
            from baikpacking.tools.events import EVENT_KEYWORDS
        except Exception:
            EVENT_KEYWORDS = []

        event_hint = _extract_event_hint(query, EVENT_KEYWORDS)
        conn = _connect(deps.database_url)

        try:
            with conn:
                if not _ensure_chunks_table_known(conn, deps.database_url):
                    if trace_ctx is not None:
                        trace_tool(
                            trace_ctx,
                            "search_similar_riders",
                            {
                                "query": query,
                                "query_component": query_component,
                                "component_terms": effective_component_terms,
                            },
                            {"riders": 0},
                            t0,
                        )
                    return []

                qvec_param = _vector_text(qvec)
                resolved_top_k_chunks = (
                    int(top_k_chunks)
                    if top_k_chunks is not None
                    else max(50, int(top_k_riders) * 50)
                )

                grounded = _ground_event_scope(
                    conn,
                    query=query,
                    event_hint=event_hint,
                    embed_query_fn=deps.embed_query,
                    min_exact_riders=_MIN_EXACT_EVENT_RIDERS,
                    max_similar_events=_MAX_SIMILAR_EVENTS,
                )

                requested_event_hint = grounded["requested_event_hint"]
                exact_event_titles = grounded["exact_event_titles"]
                similar_event_titles = grounded["similar_event_titles"]
                exact_rider_count = grounded["exact_rider_count"]
                retrieval_scope = grounded["retrieval_scope"]

                logfire.info(
                    "event grounding result",
                    query=query,
                    event_hint=event_hint,
                    requested_event_hint=requested_event_hint,
                    exact_event_titles=exact_event_titles,
                    similar_event_titles=similar_event_titles,
                    exact_rider_count=exact_rider_count,
                    retrieval_scope=retrieval_scope,
                )

                exact_rider_ids: List[int] = []
                exact_chunk_rank: Dict[int, Dict[str, Any]] = {}
                similar_chunk_rank: Dict[int, Dict[str, Any]] = {}
                
                if requested_event_hint:
                    exact_debug = _debug_exact_event_title_presence(conn, requested_event_hint)
                    logfire.info("exact title presence debug", **exact_debug)

                if exact_event_titles:
                    exact_rider_ids = _fetch_rider_ids_for_event_titles(conn, exact_event_titles)
                    exact_chunk_rank = _fetch_top_chunks_for_rider_ids(
                        conn,
                        qvec_param=qvec_param,
                        rider_ids=exact_rider_ids,
                        max_chunks_per_rider=int(max_chunks_per_rider),
                    )

                if similar_event_titles and (not exact_chunk_rank or exact_rider_count < _MIN_EXACT_EVENT_RIDERS):
                    similar_chunk_rank = _fetch_top_riders_by_chunks_for_event_titles(
                        conn,
                        qvec_param=qvec_param,
                        event_titles=similar_event_titles,
                        top_k_riders=max(int(top_k_riders), _MIN_EXACT_EVENT_RIDERS),
                        top_k_chunks=resolved_top_k_chunks,
                        max_chunks_per_rider=int(max_chunks_per_rider),
                    )

                logfire.info(
                    "exact vs similar retrieval counts",
                    requested_event_hint=requested_event_hint,
                    exact_event_titles=exact_event_titles,
                    similar_event_titles=similar_event_titles,
                    exact_rider_count=exact_rider_count,
                    exact_candidate_rider_ids=len(exact_rider_ids),
                    exact_retrieved_rider_count=len(exact_chunk_rank),
                    similar_retrieved_rider_count=len(similar_chunk_rank),
                )

                merged_chunk_rank = _merge_chunk_ranks(
                    exact_chunk_rank=exact_chunk_rank,
                    similar_chunk_rank=similar_chunk_rank,
                    similar_weight=_SIMILAR_EVENT_SCORE_WEIGHT,
                )

                if not merged_chunk_rank:
                    if requested_event_hint:
                        retrieval_scope = "no_exact_or_similar_matches"
                    else:
                        merged_chunk_rank = _fetch_top_riders_by_chunks(
                            conn,
                            qvec_param=qvec_param,
                            top_k_riders=int(top_k_riders),
                            top_k_chunks=resolved_top_k_chunks,
                            max_chunks_per_rider=int(max_chunks_per_rider),
                        )
                        retrieval_scope = "global"

                rider_ids = sorted(
                    merged_chunk_rank.keys(),
                    key=lambda rid: (
                        0 if merged_chunk_rank[rid].get("source_scope") == "exact_event" else 1,
                        merged_chunk_rank[rid].get(
                            "weighted_best_distance",
                            merged_chunk_rank[rid]["best_distance"],
                        ),
                    ),
                )

                rider_map = _fetch_rider_rows(conn, rider_ids) if rider_ids else {}

            riders = _build_riders_from_chunk_rank(
                rider_ids=rider_ids,
                rider_map=rider_map,
                chunk_rank=merged_chunk_rank,
            )

            surface_bias = _infer_surface_bias(query, requested_event_hint=requested_event_hint)

            for rider in riders:
                base_score = rider.best_score or 0.0
                base_score *= _surface_penalty_or_boost(rider, surface_bias)
                base_score += _event_subtype_bonus(
                    getattr(rider, "event_title", None),
                    requested_event_hint,
                    query,
                )
                rider.best_score = base_score

            def sort_key(r: SimilarRider):
                same_event = 1 if _is_exact_event_rider(r, requested_event_hint) else 0
                source_scope = 1 if getattr(r, "_source_scope", "") == "exact_event" else 0
                component_hit = _component_hit(r, effective_component_terms)
                known_event = _is_known_event_title(r.event_title)
                coverage = _setup_field_coverage(r)
                score = r.best_score or 0.0
                year = r.year or 0
                chunk_len = _chunk_text_len(r)

                if query_component and query_component != "full_setup":
                    return (
                        same_event,
                        source_scope,
                        component_hit,
                        known_event,
                        score,
                        coverage,
                        year,
                        chunk_len,
                    )

                return (
                    same_event,
                    source_scope,
                    known_event,
                    score,
                    coverage,
                    year,
                    chunk_len,
                )

            exact_riders = [r for r in riders if _is_exact_event_rider(r, requested_event_hint)]
            fallback_riders = [r for r in riders if not _is_exact_event_rider(r, requested_event_hint)]

            exact_riders = sorted(exact_riders, key=sort_key, reverse=True)
            fallback_riders = sorted(fallback_riders, key=sort_key, reverse=True)

            final: List[SimilarRider] = exact_riders[: int(top_k_riders)]
            remaining = int(top_k_riders) - len(final)
            if remaining > 0:
                final.extend(fallback_riders[:remaining])

            log_payload = {
                "returned_riders": len(final),
                "requested_event_hint": requested_event_hint,
                "retrieval_scope": retrieval_scope,
                "exact_event_titles": exact_event_titles,
                "similar_event_titles": similar_event_titles,
                "exact_rider_count": exact_rider_count,
                "used_exact_event_scope": retrieval_scope in {"exact_event", "exact_plus_similar"},
                "used_similar_event_scope": retrieval_scope in {"exact_plus_similar", "similar_events"},
                "used_global_fallback": retrieval_scope == "global",
                "top_rider_ids": [r.rider_id for r in final[:5]],
                "top_rider_names": [r.name for r in final[:5]],
                "top_event_titles": [r.event_title for r in final[:5]],
                "surface_bias": surface_bias,
                "exact_event_rider_names": [r.name for r in exact_riders[:5]],
                "fallback_rider_names": [r.name for r in fallback_riders[:5]],
                "exact_event_count_in_final": sum(
                    1 for r in final if _is_exact_event_rider(r, requested_event_hint)
                ),
                "exact_candidate_rider_ids": len(exact_rider_ids),
                "exact_retrieved_rider_count": len(exact_chunk_rank),
                "similar_retrieved_rider_count": len(similar_chunk_rank),
                "avg_setup_field_coverage": (
                    sum(_setup_field_coverage(r) for r in final) / len(final)
                    if final else 0.0
                ),
            }

            if effective_component_terms:
                log_payload["component_hit_count"] = sum(
                    _component_hit(r, effective_component_terms) for r in final
                )
            else:
                log_payload["bike_type_count"] = sum(1 for r in final if getattr(r, "bike_type", None))
                log_payload["wheels_count"] = sum(1 for r in final if getattr(r, "wheels", None))
                log_payload["tyres_count"] = sum(1 for r in final if getattr(r, "tyres", None))
                log_payload["drivetrain_count"] = sum(1 for r in final if getattr(r, "drivetrain", None))
                log_payload["bags_count"] = sum(1 for r in final if getattr(r, "bags", None))
                log_payload["sleep_system_count"] = sum(1 for r in final if getattr(r, "sleep_system", None))

            logfire.info("search_similar_riders completed", **log_payload)

            if trace_ctx is not None:
                trace_tool(
                    trace_ctx,
                    "search_similar_riders",
                    {
                        "query": query,
                        "query_component": query_component,
                        "component_terms": effective_component_terms,
                        "top_k_riders": top_k_riders,
                        "max_chunks_per_rider": max_chunks_per_rider,
                        "top_k_chunks": resolved_top_k_chunks,
                    },
                    log_payload,
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
    return run_render_grounding_riders(
        riders=riders,
        trace_ctx=ctx,
    )