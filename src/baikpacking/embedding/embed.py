import os
import json
import unicodedata
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

import requests

from .config import Settings
from baikpacking.tools.events import EVENT_ALIASES

settings = Settings()

_SESSION = requests.Session()
_TIMEOUT_S = 30


def _normalize_text_for_match(s: str) -> str:
    if not s:
        return ""
    s = s.lower()

    # treat typical URL/title separators as spaces
    for sep in ("-", "_", "/", "."):
        s = s.replace(sep, " ")

    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

    return "".join(ch for ch in s if ch.isalnum() or ch.isspace())



_NORMALIZED_EVENT_ALIASES: Dict[str, List[str]] = {
    key: [_normalize_text_for_match(a) for a in aliases]
    for key, aliases in EVENT_ALIASES.items()
}


def infer_event_key_from_title(title: str) -> Optional[str]:
    norm = _normalize_text_for_match(title)
    if not norm:
        return None
    for key, aliases in _NORMALIZED_EVENT_ALIASES.items():
        for alias in aliases:
            if alias and alias in norm:
                return key
    return None


def _stable_rider_id(event_url: str, event_title: str, rider_name: str) -> int:
    key = f"{event_url}|{event_title}|{rider_name}".encode("utf-8")
    return int(hashlib.blake2b(key, digest_size=8).hexdigest(), 16)


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


def build_embedding_text(rider: Dict[str, Any]) -> str:
    parts = [
        rider.get("name", ""),
        rider.get("age", ""),
        rider.get("location", ""),
        rider.get("bike", ""),
        rider.get("frame_type", ""),
        rider.get("frame_material", ""),
        rider.get("wheel_size", ""),
        rider.get("tyre_width", ""),
        rider.get("key_items", ""),
    ]
    if rider.get("event_title"):
        parts.append(f"Event: {rider['event_title']}")
    if rider.get("event_key"):
        parts.append(f"EventKey: {rider['event_key']}")
    return " | ".join(p for p in parts if p)


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks


def embed_rider_chunks(
    rider: Dict[str, Any],
    event_title: str,
    event_url: str,
    event_key: Optional[str],
    max_chars: int = 800,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    # ensure build_embedding_text sees event metadata
    rider = dict(rider)
    rider["event_title"] = event_title
    rider["event_url"] = event_url
    rider["event_key"] = event_key

    text = build_embedding_text(rider)
    chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
    if not chunks:
        return []

    vectors = embed_texts(chunks)
    if len(vectors) != len(chunks):
        raise RuntimeError(f"Embedding count mismatch: {len(vectors)} != {len(chunks)}")

    payload_base = {
        "rider_id": rider.get("rider_id"),
        "name": rider.get("name"),
        "event_title": event_title,
        "event_url": event_url,
        "frame_type": rider.get("frame_type"),
        "frame_material": rider.get("frame_material"),
        "wheel_size": rider.get("wheel_size"),
        "tyre_width": rider.get("tyre_width"),
        "electronic_shifting": rider.get("electronic_shifting"),
        "event_key": event_key,
    }

    out: List[Dict[str, Any]] = []
    for i, (chunk_text_value, vec) in enumerate(zip(chunks, vectors)):
        out.append({**payload_base, "chunk_index": i, "text": chunk_text_value, "vector": vec})
    return out


def embed_riders_from_json(
    json_path: Union[Path, str] = "data/dotwatcher_bikes_cleaned.json",
    max_chars: int = 800,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    path = Path(json_path)
    articles = json.loads(path.read_text(encoding="utf-8"))

    all_chunks: List[Dict[str, Any]] = []

    for article in articles:
        event_title = (article.get("title") or "").strip()
        event_url = (article.get("url") or "").strip()

        event_key = infer_event_key_from_title(event_title) or infer_event_key_from_title(event_url)


        riders = article.get("riders") or []
        for rider in riders:
            r: Dict[str, Any] = dict(rider)
            r["rider_id"] = _stable_rider_id(event_url, event_title, str(r.get("name") or ""))

            chunks = embed_rider_chunks(
                r,
                event_title=event_title,
                event_url=event_url,
                event_key=event_key,
                max_chars=max_chars,
                overlap=overlap,
            )
            all_chunks.extend(chunks)

    return all_chunks
