from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from baikpacking.agents.models import QueryIntent
from baikpacking.agents.orchestration_models import (
    EvidenceSummary,
    EventResolutionResult,
    RetrievalExecutionResult,
)


def _rider_component_hit_count(riders: List[Any], component_terms: List[str]) -> int:
    if not riders or not component_terms:
        return 0

    terms = [t.lower() for t in component_terms if t.strip()]
    hits = 0

    for r in riders:
        parts = []

        for value in [
            getattr(r, "bike_type", None),
            getattr(r, "wheels", None),
            getattr(r, "tyres", None),
            getattr(r, "drivetrain", None),
            getattr(r, "bags", None),
            getattr(r, "sleep_system", None),
        ]:
            if isinstance(value, str) and value.strip():
                parts.append(value)

        for item in getattr(r, "key_items", None) or []:
            if isinstance(item, str) and item.strip():
                parts.append(item)

        for c in getattr(r, "chunks", None) or []:
            text = getattr(c, "text", None) or getattr(c, "content", None) or ""
            if text:
                parts.append(text)

        searchable = " ".join(parts).lower()

        if any(term in searchable for term in terms):
            hits += 1

    return hits


def _collect_text_support(riders: Sequence[Any], field_name: str) -> int:
    count = 0
    field_terms = {
        "bike_type": ["bike type", "hardtail", "gravel", "mtb", "road", "mountain bike", "frame"],
        "tyres": ["tyre", "tire", "tubeless", "casing", "mm", "650b", "700c"],
        "drivetrain": ["drivetrain", "groupset", "cassette", "chainring", "gear", "grx", "shimano", "sram"],
        "bags": ["bag", "bags", "frame bag", "seat pack", "handlebar bag", "apidura", "tailfin", "ortlieb"],
        "sleep_system": ["sleep", "sleeping bag", "bivy", "bivvy", "quilt", "mat", "tent", "pad"],
        "wheels": ["wheel", "wheels", "wheelset", "rim", "hubs", "650b", "700c", "29er"],
    }[field_name]

    for rider in riders:
        parts = []
        for value in [
            getattr(rider, field_name, None),
            getattr(rider, "bike", None),
            getattr(rider, "key_items", None),
        ]:
            if isinstance(value, list):
                parts.extend([str(v) for v in value if isinstance(v, str) and v.strip()])
            elif isinstance(value, str) and value.strip():
                parts.append(value)
        for chunk in getattr(rider, "chunks", None) or []:
            text = getattr(chunk, "text", None) or getattr(chunk, "content", None) or ""
            if text:
                parts.append(text)
        searchable = " ".join(parts).lower()
        if any(term in searchable for term in field_terms):
            count += 1
    return count


def _classify_support(n: int) -> str:
    if n <= 0:
        return "none"
    if n == 1:
        return "single"
    if n <= 3:
        return "weak_pattern"
    return "pattern"


def _evidence_strength(rider_count: int, support_counts: Dict[str, int], component_hit_count: int) -> str:
    max_field = max(support_counts.values()) if support_counts else 0
    if rider_count == 0:
        return "none"
    if component_hit_count == 0 and max_field <= 1:
        return "weak"
    if component_hit_count >= 2 or max_field >= 4:
        return "strong"
    if component_hit_count >= 1 or max_field >= 2:
        return "moderate"
    return "weak"


def _consistency_label(rider_count: int, support_counts: Dict[str, int]) -> str:
    if rider_count <= 1:
        return "sparse"
    values = [v for v in support_counts.values() if v > 0]
    if not values:
        return "sparse"
    if len(set(values)) == 1:
        return "consistent"
    if max(values) - min(values) <= 1:
        return "mostly_consistent"
    return "mixed"


def summarize_evidence(
    riders: List[Any],
    intent: QueryIntent,
    event_resolution: EventResolutionResult,
    retrieval_result: RetrievalExecutionResult,
) -> EvidenceSummary:
    rider_count = len(riders or [])
    component_hit_count = _rider_component_hit_count(riders or [], intent.component_terms or [])

    support_counts = {
        field: _collect_text_support(riders or [], field)
        for field in ["bike_type", "tyres", "drivetrain", "bags", "sleep_system", "wheels"]
    }

    field_support = {field: _classify_support(count) for field, count in support_counts.items()}

    return EvidenceSummary(
        rider_count=rider_count,
        component_hit_count=component_hit_count,
        field_support=field_support,
        evidence_strength=_evidence_strength(rider_count, support_counts, component_hit_count),
        consistency=_consistency_label(rider_count, support_counts),
    )
