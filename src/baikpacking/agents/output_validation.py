from typing import Any, Dict, List, Optional

from baikpacking.agents.models import SetupRecommendation


_COMPONENT_FILL_RULES: Dict[str, Dict[str, Any]] = {
    "tyres": {
        "target_field": "tyres",
        "structured_field": "tyres",
        "chunk_keywords": ["tyre", "tyres", "tire", "tires", "tubeless", "casing", "mm"],
    },
    "wheels": {
        "target_field": "wheels",
        "structured_field": "wheels",
        "chunk_keywords": ["wheel", "wheels", "wheelset", "rim", "rims", "hub", "hubs", "650b", "700c", "29er", "27.5"],
    },
    "drivetrain": {
        "target_field": "drivetrain",
        "structured_field": "drivetrain",
        "chunk_keywords": ["drivetrain", "groupset", "cassette", "chainring", "gearing", "gear ratio", "sram", "shimano", "grx", "1x", "2x"],
    },
    "bags": {
        "target_field": "bags",
        "structured_field": "bags",
        "chunk_keywords": ["bag", "bags", "frame bag", "seat pack", "saddle bag", "top tube bag", "handlebar bag", "apidura", "tailfin", "ortlieb", "restrap"],
    },
    "sleep_system": {
        "target_field": "sleep_system",
        "structured_field": "sleep_system",
        "chunk_keywords": ["sleep", "sleeping bag", "bivy", "bivvy", "quilt", "mat", "pad", "tent"],
    },
    "bike_type": {
        "target_field": "bike_type",
        "structured_field": "bike_type",
        "chunk_keywords": ["gravel bike", "road bike", "endurance bike", "mountain bike", "mtb", "hardtail", "full suspension", "bike type", "frame"],
    },
}


def _first_nonempty(values: List[Optional[str]]) -> Optional[str]:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _fill_requested_component_from_riders(
    rec: SetupRecommendation,
    riders: List[Any],
    query_component: str,
) -> SetupRecommendation:
    rule = _COMPONENT_FILL_RULES.get(query_component)
    if not rule:
        return rec

    rs = rec.recommended_setup
    target_field = rule["target_field"]
    structured_field = rule["structured_field"]
    chunk_keywords = [k.lower() for k in rule["chunk_keywords"]]

    current_value = getattr(rs, target_field, None)
    if isinstance(current_value, str) and current_value.strip():
        return rec

    structured_candidates: List[str] = []
    chunk_candidates: List[str] = []

    for r in riders:
        structured_value = getattr(r, structured_field, None)
        if isinstance(structured_value, str) and structured_value.strip():
            structured_candidates.append(structured_value.strip())

        for c in getattr(r, "chunks", None) or []:
            text = (getattr(c, "text", None) or "").strip()
            if not text:
                continue

            tl = text.lower()
            if any(keyword in tl for keyword in chunk_keywords):
                chunk_candidates.append(text)

    best = _first_nonempty(structured_candidates) or _first_nonempty(chunk_candidates)
    if best:
        setattr(rs, target_field, best)

    return rec
