import hashlib
import logging
import unicodedata
from typing import Optional, List, Dict, Any, Iterable

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from .config import Settings
from .embed import embed_texts
from baikpacking.tools.events import EVENT_ALIASES

try:
    from baikpacking.retrieval.rank import RerankerConfig, rerank_hits
except Exception:
    RerankerConfig = None 
    rerank_hits = None  

logger = logging.getLogger(__name__)
settings = Settings()

_HTTP_TIMEOUT_S = 30


# ---------------------------------------------------------------------------
# IDs
# ---------------------------------------------------------------------------


def stable_point_id(*, rider_id: int, chunk_index: int, event_title: str) -> int:
    """
    Generate a stable 64-bit integer point ID for Qdrant.

    This avoids overwriting or mixing points across re-index runs and makes upserts idempotent.
    """
    key = f"{rider_id}:{chunk_index}:{event_title}".encode("utf-8")
    return int(hashlib.blake2b(key, digest_size=8).hexdigest(), 16)


# ---------------------------------------------------------------------------
# Qdrant client + collection management
# ---------------------------------------------------------------------------


def get_qdrant_client() -> QdrantClient:
    """
    Build a QdrantClient from settings.
    """
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=60.0,
    )


def ensure_collection(vector_size: int, client: Optional[QdrantClient] = None) -> QdrantClient:
    """
    Ensure the target collection exists with the correct vector size and
    required payload indexes for filtering.

    Creates (idempotently):
      - collection (if missing)
      - payload index on event_key (KEYWORD) 
      - payload index on rider_id (INTEGER)   
    """
    client = client or get_qdrant_client()
    name = settings.qdrant_collection

    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection '%s' (vector_size=%s)", name, vector_size)

    def _ensure_payload_index(field_name: str, schema: rest.PayloadSchemaType) -> None:
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=schema,
            )
            logger.info("Created payload index on '%s.%s' (%s)", name, field_name, schema)
        except Exception as exc:
            # Qdrant may raise if index already exists; we tolerate that.
            # But keep a log line for visibility.
            logger.debug("Payload index '%s.%s' not created (likely exists): %s", name, field_name, exc)

    _ensure_payload_index("event_key", rest.PayloadSchemaType.KEYWORD)
    _ensure_payload_index("rider_id", rest.PayloadSchemaType.INTEGER)

    return client


def _validate_chunks(chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        raise ValueError("chunks is empty")

    required = {"rider_id", "chunk_index", "vector", "text"}
    missing_any = required - set(chunks[0].keys())
    if missing_any:
        raise ValueError(f"Chunk missing required keys: {sorted(missing_any)}")

    vec_size = len(chunks[0]["vector"])
    if vec_size <= 0:
        raise ValueError("Vector size is invalid (<=0)")

    for i, ch in enumerate(chunks[:200]):  # sanity-check first N only
        if "vector" not in ch or not isinstance(ch["vector"], list):
            raise ValueError(f"Chunk[{i}] has invalid vector")
        if len(ch["vector"]) != vec_size:
            raise ValueError(f"Chunk[{i}] vector size mismatch ({len(ch['vector'])} != {vec_size})")
        if ch.get("rider_id") is None:
            raise ValueError(f"Chunk[{i}] missing rider_id")
        if ch.get("chunk_index") is None:
            raise ValueError(f"Chunk[{i}] missing chunk_index")


def upsert_chunks_to_qdrant(chunks: List[Dict[str, Any]], batch_size: int = 500) -> None:
    """
    Upsert rider chunks into Qdrant in batches.

    Each chunk dict is expected to have:
      - rider_id (int)
      - chunk_index (int)
      - text (str)
      - vector (List[float])
      - optional metadata fields: name, event_title, event_url, frame_type, tyre_width, event_key, ...
    """
    if not chunks:
        logger.info("No chunks to upsert into Qdrant.")
        return

    _validate_chunks(chunks)

    client = get_qdrant_client()
    vec_size = len(chunks[0]["vector"])
    ensure_collection(vec_size, client=client)

    collection_name = settings.qdrant_collection
    total = len(chunks)
    logger.info("Upserting %d chunks into Qdrant collection '%s' (batch_size=%d)", total, collection_name, batch_size)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = chunks[start:end]

        points: List[rest.PointStruct] = []
        for chunk in batch:
            rider_id = int(chunk["rider_id"])
            chunk_index = int(chunk["chunk_index"])
            event_title = str(chunk.get("event_title") or "")

            pid = stable_point_id(rider_id=rider_id, chunk_index=chunk_index, event_title=event_title)
            payload = {k: v for k, v in chunk.items() if k != "vector"}

            # Optional debug: ensure event_key exists if event_title exists
            if payload.get("event_title") and not payload.get("event_key"):
                logger.debug("Chunk missing event_key for event_title=%r rider_id=%s", payload.get("event_title"), rider_id)

            points.append(rest.PointStruct(id=pid, vector=chunk["vector"], payload=payload))

        client.upsert(collection_name=collection_name, points=points, wait=True)
        logger.info("Upserted %d/%d chunks", end, total)

    logger.info("Finished upserting %d chunks into '%s'.", total, collection_name)


# ---------------------------------------------------------------------------
# Search helpers (semantic + event-aware)
# ---------------------------------------------------------------------------


_session = requests.Session()


def _qdrant_search_http(
    *,
    query_vector: List[float],
    top_k: int,
    event_key: Optional[str],
) -> List[Dict[str, Any]]:
    base_url = settings.qdrant_url.rstrip("/")
    collection_name = settings.qdrant_collection
    url = f"{base_url}/collections/{collection_name}/points/search"

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if settings.qdrant_api_key:
        headers["api-key"] = settings.qdrant_api_key

    body: Dict[str, Any] = {
        "vector": query_vector,
        "limit": top_k,
        "with_payload": True,
        "with_vectors": False,
    }

    if event_key:
        body["filter"] = {
            "must": [{"key": "event_key", "match": {"value": event_key}}],
        }

    resp = _session.post(url, json=body, headers=headers, timeout=_HTTP_TIMEOUT_S)
    if resp.status_code != 200:
        raise RuntimeError(f"Qdrant search failed: {resp.status_code} {resp.text}")

    data = resp.json()
    results = data.get("result") or []

    return [
        {"id": r.get("id"), "score": r.get("score"), "payload": r.get("payload") or {}}
        for r in results
    ]


def search_riders(query: str, top_k: int = 5, event_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Semantic search over rider chunks in Qdrant.

    - Embeds the query with the embedding model
    - Searches Qdrant for nearest chunks
    - Optionally filters to a single event via payload.event_key
    """
    vectors = embed_texts([query])
    if not vectors:
        logger.warning("embed_texts returned no vectors for query")
        return []

    query_vector = vectors[0]
    return _qdrant_search_http(query_vector=query_vector, top_k=top_k, event_key=event_key)


def group_hits_by_rider(
    hits: List[Dict[str, Any]],
    top_k_riders: int = 5,
    max_chunks_per_rider: int = 3,
) -> List[Dict[str, Any]]:
    """
    Group raw Qdrant hits (chunk-level) by rider_id and return a ranked list of riders.
    """
    by_rider: Dict[Any, Dict[str, Any]] = {}

    for hit in hits:
        payload = hit.get("payload") or {}
        rider_id = payload.get("rider_id")
        if rider_id is None:
            continue

        score = float(hit.get("score") or 0.0)

        agg = by_rider.get(rider_id)
        if agg is None:
            agg = {
                "rider_id": rider_id,
                "name": payload.get("name"),
                "event_title": payload.get("event_title"),
                "event_url": payload.get("event_url"),
                "frame_type": payload.get("frame_type"),
                "frame_material": payload.get("frame_material"),
                "wheel_size": payload.get("wheel_size"),
                "tyre_width": payload.get("tyre_width"),
                "electronic_shifting": payload.get("electronic_shifting"),
                "event_key": payload.get("event_key"),
                "best_score": score,
                "chunks": [],
            }
            by_rider[rider_id] = agg

        if score > agg["best_score"]:
            agg["best_score"] = score
            agg["name"] = payload.get("name")
            agg["event_title"] = payload.get("event_title")
            agg["event_url"] = payload.get("event_url")
            agg["frame_type"] = payload.get("frame_type")
            agg["frame_material"] = payload.get("frame_material")
            agg["wheel_size"] = payload.get("wheel_size")
            agg["tyre_width"] = payload.get("tyre_width")
            agg["electronic_shifting"] = payload.get("electronic_shifting")
            agg["event_key"] = payload.get("event_key")

        agg["chunks"].append(
            {
                "score": score,
                "text": payload.get("text", ""),
                "chunk_index": payload.get("chunk_index", 0),
            }
        )

    riders = list(by_rider.values())

    for r in riders:
        r["chunks"].sort(key=lambda c: c["score"], reverse=True)
        r["chunks"] = r["chunks"][:max_chunks_per_rider]

    riders.sort(key=lambda r: r["best_score"], reverse=True)
    return riders[:top_k_riders]


# ---------------------------------------------------------------------------
# Event detection / normalization
# ---------------------------------------------------------------------------


def normalize_text_for_match(s: str) -> str:
    """
    Lowercase, remove accents and keep only alphanumerics + spaces.
    Helps match 'TransIbérica' with 'transiberica'.
    """
    if not s:
        return ""

    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return "".join(ch for ch in s if ch.isalnum() or ch.isspace())


_NORMALIZED_EVENT_ALIASES: Dict[str, List[str]] = {
    key: [normalize_text_for_match(a) for a in aliases]
    for key, aliases in EVENT_ALIASES.items()
}


def detect_event_key(text: str) -> Optional[str]:
    """
    Detect a canonical event_key from arbitrary text using EVENT_ALIASES.
    Used both at embedding time (from event_title) and query time (from user query).
    """
    norm = normalize_text_for_match(text)
    if not norm:
        return None

    for key, aliases in _NORMALIZED_EVENT_ALIASES.items():
        for alias in aliases:
            if alias and alias in norm:
                return key
    return None


# ---------------------------------------------------------------------------
# High-level grouped search (used by the recommender agent)
# ---------------------------------------------------------------------------

class _RerankHit:
    """Minimal adapter for rerank_hits(): doc_id/score/payload."""
    def __init__(self, doc_id: int, score: float, payload: Dict[str, Any]):
        self.doc_id = doc_id
        self.score = score
        self.payload = payload


def _rerank_grouped_riders(
    query: str,
    riders: List[Dict[str, Any]],
    cfg: "RerankerConfig",
) -> List[Dict[str, Any]]:
    # Build one hit per rider using best_score + best chunk text as payload["text"]
    hits: List[_RerankHit] = []
    by_id: Dict[int, Dict[str, Any]] = {}

    for r in riders:
        rid = int(r["rider_id"])
        by_id[rid] = r

        chunks = r.get("chunks") or []
        best_text = (chunks[0].get("text") if chunks else "") or ""

        payload = {
            "text": best_text,  # reranker expects payload["text"]
            "event_key": r.get("event_key"),
            "frame_type": r.get("frame_type"),
            "tyre_width": r.get("tyre_width"),
            "electronic_shifting": r.get("electronic_shifting"),
        }
        hits.append(_RerankHit(doc_id=rid, score=float(r.get("best_score") or 0.0), payload=payload))

    ranked_hits = rerank_hits(query, hits, cfg)  # type: ignore[misc]
    return [by_id[int(h.doc_id)] for h in ranked_hits if int(h.doc_id) in by_id]



def search_riders_grouped(
    query: str,
    top_k_riders: int = 10,
    oversample_factor: int = 10,
    max_chunks_per_rider: int = 3,
    *,
    apply_rerank: bool = True,
    rerank_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    High-level helper:

    1) Detect event key in the query.
    2) If found, search within that event (payload.event_key filter).
    3) If event-filter yields 0 hits, fall back to global search (and log).
    4) Group results by rider_id and return top_k_riders.
    """
    limit = top_k_riders * oversample_factor
    event_key = detect_event_key(query)

    raw_hits = search_riders(query, top_k=limit, event_key=event_key)

    if event_key and not raw_hits:
        logger.warning("0 hits for event_key=%s; falling back to global search", event_key)
        raw_hits = search_riders(query, top_k=limit, event_key=None)

    riders = group_hits_by_rider(
        raw_hits,
        top_k_riders=limit,
        max_chunks_per_rider=max_chunks_per_rider,
    )
    
    if apply_rerank and rerank_hits is not None and RerankerConfig is not None:
        cfg = RerankerConfig(**(rerank_config or {}))
        riders = _rerank_grouped_riders(query, riders, cfg)

    return riders[:top_k_riders]
 
