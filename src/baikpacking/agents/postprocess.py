import re
from collections import Counter
from typing import Optional

from baikpacking.agents.models import SetupRecommendation


_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _infer_year_from_title(title: Optional[str]) -> Optional[int]:
    if not title:
        return None
    match = _YEAR_RE.search(title)
    return int(match.group(0)) if match else None


def _infer_event_from_riders(rec: SetupRecommendation) -> Optional[str]:
    titles = [
        rider.event_title
        for rider in (rec.similar_riders or [])
        if isinstance(rider.event_title, str) and rider.event_title.strip()
    ]
    return Counter(titles).most_common(1)[0][0] if titles else None


def _postprocess_recommendation(rec: SetupRecommendation) -> SetupRecommendation:
    for rider in rec.similar_riders:
        for idx, chunk in enumerate(getattr(rider, "chunks", []) or []):
            if chunk.chunk_index is None:
                chunk.chunk_index = idx
        if getattr(rider, "year", None) is None:
            rider.year = _infer_year_from_title(getattr(rider, "event_title", None))

    if not rec.event:
        rec.event = _infer_event_from_riders(rec) or (
            rec.similar_riders[0].event_title if rec.similar_riders else None
        )

    event_lower = (rec.event or "").lower()
    rec.similar_riders.sort(
        key=lambda rider: (
            event_lower in (rider.event_title or "").lower(),
            rider.best_score or 0,
            rider.year or 0,
        ),
        reverse=True,
    )
    return rec
