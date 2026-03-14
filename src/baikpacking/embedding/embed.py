import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .config import Settings

import requests

from .config import Settings

settings = Settings()
_TIMEOUT_S = 30


def _ollama_embeddings_url() -> str:
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return f"{ollama_host.rstrip('/')}/api/embeddings"


def _post_ollama(url: str, payload: dict) -> dict:
    last_exc: Exception | None = None
    backoffs = [0.2, 0.5, 1.0]

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=_TIMEOUT_S)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            if attempt < len(backoffs):
                time.sleep(backoffs[attempt])

    raise RuntimeError(f"Ollama request failed after 3 attempts: {last_exc}")


def _extract_embedding(data: dict) -> List[float]:
    emb = data.get("embedding")
    if not emb or not isinstance(emb, list):
        raise RuntimeError(f"Unexpected response from Ollama: {data}")
    return emb


def _check_dim(vectors: List[List[float]], expected_dim: Optional[int]) -> None:
    if expected_dim is None:
        return
    if vectors and len(vectors[0]) != expected_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: got {len(vectors[0])}, expected {expected_dim}"
        )


def embed_texts(
    texts: List[str],
    *,
    model: Optional[str] = None,
    concurrent: bool = False,
    max_workers: int = 8,
    expected_dim: Optional[int] = None,
) -> List[List[float]]:
    if not texts:
        return []

    for i, text in enumerate(texts):
        if not text or not text.strip():
            raise ValueError(f"Empty text at index {i}")

    chosen_model = model or settings.embedding_model

    if concurrent:
        return embed_texts_concurrent(
            texts,
            model=chosen_model,
            max_workers=max_workers,
            expected_dim=EXPECTED_EMBED_DIM,
        )

    url = _ollama_embeddings_url()
    vectors: List[List[float]] = []
    for text in texts:
        payload = {"model": chosen_model, "prompt": text}
        data = _post_ollama(url, payload)
        vectors.append(_extract_embedding(data))

    _check_dim(vectors, expected_dim)
    return vectors


def embed_text(
    text: str,
    *,
    model: Optional[str] = None,
    expected_dim: Optional[int] = None,
) -> List[float]:
    vecs = embed_texts([text], model=model, expected_dim=expected_dim)
    return vecs[0] if vecs else []


def embed_texts_concurrent(
    texts: List[str],
    *,
    model: Optional[str] = None,
    max_workers: int = 8,
    expected_dim: Optional[int] = None,
) -> List[List[float]]:
    if not texts:
        return []

    for i, text in enumerate(texts):
        if not text or not text.strip():
            raise ValueError(f"Empty text at index {i}")

    url = _ollama_embeddings_url()
    chosen_model = model or settings.embedding_model
    vectors: List[Optional[List[float]]] = [None] * len(texts)

    def _one(i: int, text: str) -> tuple[int, List[float]]:
        payload = {"model": chosen_model, "prompt": text}
        try:
            data = _post_ollama(url, payload)
            return i, _extract_embedding(data)
        except Exception as e:
            preview = text[:200].replace("\n", " ")
            raise RuntimeError(
                f"Embedding failed for index={i}, text_preview={preview!r}: {e}"
            ) from e

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_one, i, t) for i, t in enumerate(texts)]
        for fut in as_completed(futures):
            i, emb = fut.result()
            vectors[i] = emb

    out: List[List[float]] = []
    for v in vectors:
        if v is None:
            raise RuntimeError("Internal error: missing embedding result for one or more inputs.")
        out.append(v)

    _check_dim(out, expected_dim)
    return out


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

    name = g("name", "rider_name")
    if name:
        parts.append(f"Rider: {name}")

    event = g("event_title", "event", "event_name")
    if event:
        parts.append(f"Event: {event}")

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
    *,
    concurrent: bool = False,
    max_workers: int = 8,
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

    vectors = embed_texts(
        texts,
        concurrent=concurrent,
        max_workers=max_workers,
        expected_dim=expected_dim,
    )

    if len(vectors) != len(ids):
        raise RuntimeError(f"Embedding count mismatch: {len(vectors)} != {len(ids)}")

    return list(zip(ids, vectors))