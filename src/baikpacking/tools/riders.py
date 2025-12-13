import json
import logging
import os
import re
from typing import List, Optional

from pydantic_ai import RunContext, Tool
from pydantic import ValidationError

from baikpacking.embedding.qdrant_utils import search_riders_grouped
from baikpacking.agents.models import SimilarRider

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"(19|20)\d{2}")


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


@Tool
def search_similar_riders(
    ctx: RunContext,
    query: str,
    top_k_riders: int = 5,
    debug: bool = False,
) -> List[SimilarRider]:
    """
    Retrieve similar rider setups from Qdrant and return a clean, grounded list.

    Key properties:
    - Oversamples then re-ranks.
    - Never crashes when debug=True.
    - Never mutates 'year' to a fake value like 0; keep None if unknown.
    """
    logger.info("search_similar_riders called", extra={"top_k_riders": top_k_riders})

    raw = search_riders_grouped(
        query=query,
        top_k_riders=top_k_riders * 2,
        oversample_factor=5,
        max_chunks_per_rider=3,
    )

    logger.info("search_riders_grouped returned %d raw riders", len(raw))

    if raw and (debug or os.getenv("BAIKPACKING_DEBUG_RIDERS") == "1"):
        sample = dict(raw[0])
        if "chunks" in sample and isinstance(sample["chunks"], list):
            sample["chunks"] = sample["chunks"][:1]
        logger.info("sample raw rider payload: %s", json.dumps(sample, ensure_ascii=False)[:4000])

    riders: List[SimilarRider] = []
    for r in raw:
        try:
            rider = SimilarRider(**r)
        except ValidationError as e:
            # This prevents “one bad record breaks the run”
            logger.warning("Skipping invalid rider payload: %s", e)
            continue

        if rider.year is None:
            rider.year = _infer_year_from_title(rider.event_title)

        riders.append(rider)

    if not riders:
        logger.warning("No valid riders parsed from Qdrant result.")
        return []

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

    riders_sorted = sorted(riders, key=sort_key, reverse=True)
    final = riders_sorted[:top_k_riders]

    logger.info(
        "search_similar_riders returning %d riders (names=%s)",
        len(final),
        [r.name for r in final if r.name][:3],
    )
    return final

@Tool
def render_grounding_riders(riders: List[SimilarRider]) -> str:
    """
    Render riders as a JSON string to be copied verbatim into SetupRecommendation.similar_riders.
    This avoids the LLM dropping fields like name/article_id.
    """

    payload = [r.model_dump(exclude_none=True) for r in riders]
    return json.dumps(payload, ensure_ascii=False)