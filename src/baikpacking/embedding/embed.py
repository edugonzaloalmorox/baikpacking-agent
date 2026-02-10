import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import requests

from .config import Settings

settings = Settings()

_SESSION = requests.Session()
_TIMEOUT_S = 30


def _post_ollama(url: str, payload: dict) -> dict:
    last_exc: Exception | None = None
    for _ in range(3):
        try:
            resp = _SESSION.post(url, json=payload, timeout=_TIMEOUT_S)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"Ollama request failed after 3 attempts: {last_exc}")


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using Ollama embeddings endpoint.
    Returns a list of vectors (list[float]).
    """
    if not texts:
        return []

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    url = f"{ollama_host.rstrip('/')}/api/embeddings"

    vectors: List[List[float]] = []
    for text in texts:
        payload = {"model": settings.embedding_model, "prompt": text}
        data = _post_ollama(url, payload)
        emb = data.get("embedding")
        if not emb:
            raise RuntimeError(f"Unexpected response from Ollama: {data}")
        vectors.append(emb)

    return vectors


def build_rider_embedding_text(r: Mapping[str, Any]) -> str:
    """
    Deterministic rider text builder.
    Works even if some columns are missing in the DB row.
    """
    def g(*keys: str) -> str:
        for k in keys:
            v = r.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
        return ""

    parts: List[str] = []

    # identifiers / context first
    name = g("name", "rider_name")
    if name:
        parts.append(f"Rider: {name}")

    event = g("event_title", "event", "event_name")
    if event:
        parts.append(f"Event: {event}")

    # setup details (keep order stable)
    for label, keys in [
        ("Location", ("location", "country")),
        ("Bike", ("bike", "bicycle")),
        ("Frame type", ("frame_type",)),
        ("Frame material", ("frame_material",)),
        ("Wheel size", ("wheel_size",)),
        ("Tyre width", ("tyre_width", "tire_width")),
        ("Key items", ("key_items", "notes", "setup_notes")),
    ]:
        val = g(*keys)
        if val:
            parts.append(f"{label}: {val}")

    return " | ".join(parts)


def embed_riders_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_dim: int = 1024,
) -> List[Tuple[int, List[float]]]:
    """
    Given DB rider rows, embed 1 vector per rider.

    Returns: [(rider_id, vector), ...]
    """
    texts: List[str] = []
    ids: List[int] = []

    for r in rows:
        rider_id = r.get("id") or r.get("rider_id")
        if rider_id is None:
            raise ValueError("Row missing rider id (expected 'id' or 'rider_id').")
        ids.append(int(rider_id))
        texts.append(build_rider_embedding_text(r))

    vectors = embed_texts(texts)

    if len(vectors) != len(ids):
        raise RuntimeError(f"Embedding count mismatch: {len(vectors)} != {len(ids)}")

    if vectors and len(vectors[0]) != expected_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: got {len(vectors[0])}, expected {expected_dim}"
        )

    return list(zip(ids, vectors))