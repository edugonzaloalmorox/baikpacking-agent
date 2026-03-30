import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from baikpacking.agents.models import SetupRecommendation
from baikpacking.agents.orchestration_models import RecommendationPolicy


_TYRE_WIDTH_RE = re.compile(r"\b\d{2,3}(?:\.\d+)?\s?mm\b", re.IGNORECASE)
_FIELD_NAMES = ["bike_type", "wheels", "tyres", "drivetrain", "bags", "sleep_system"]


_COMPONENT_FILL_RULES: Dict[str, Dict[str, Any]] = {
    "tyres": {
        "target_field": "tyres",
        "structured_fields": ["tyres", "tyre_width"],
        "positive_terms": ["tyre", "tyres", "tire", "tires", "tubeless", "casing", "tread", "slick", "semi-slick", "width"],
        "negative_terms": [
            "bike", "bikepacking", "wheel", "wheels", "wheelset", "hub", "rim",
            "drivetrain", "cassette", "chainring", "bag", "sleep", "navigation",
            "water", "repair", "kit", "sealant", "pump", "patch",
        ],
    },
    "wheels": {
        "target_field": "wheels",
        "structured_fields": ["wheels", "wheel_size"],
        "positive_terms": ["wheel", "wheels", "wheelset", "rim", "rims", "hub", "hubs", "650b", "700c", "29er", "27.5"],
        "negative_terms": [
            "bike", "bikepacking", "drivetrain", "cassette", "chainring", "derailleur",
            "tyre", "tyres", "tire", "tires", "bag", "sleep", "navigation", "water",
        ],
    },
    "drivetrain": {
        "target_field": "drivetrain",
        "structured_fields": ["drivetrain"],
        "positive_terms": ["drivetrain", "groupset", "cassette", "chainring", "gearing", "gear ratio", "sram", "shimano", "grx", "1x", "2x", "derailleur"],
        "negative_terms": ["bike", "bikepacking", "wheel", "wheels", "wheelset", "rim", "tyre", "bag", "sleep", "navigation", "water"],
    },
    "bags": {
        "target_field": "bags",
        "structured_fields": ["bags"],
        "positive_terms": ["bag", "bags", "frame bag", "seat pack", "saddle bag", "top tube bag", "top-tube bag", "handlebar bag", "handlebar roll", "apidura", "tailfin", "ortlieb", "restrap", "cargo"],
        "negative_terms": ["bike", "bikepacking", "wheel", "wheels", "wheelset", "rim", "drivetrain", "cassette", "chainring", "sleep", "navigation", "water"],
    },
    "sleep_system": {
        "target_field": "sleep_system",
        "structured_fields": ["sleep_system"],
        "positive_terms": ["sleep", "sleeping bag", "bivy", "bivvy", "quilt", "mat", "pad", "tent", "sleep system"],
        "negative_terms": ["bike", "bikepacking", "wheel", "wheels", "wheelset", "bag", "bags", "navigation", "water", "drivetrain", "cassette", "chainring"],
    },
    "bike_type": {
        "target_field": "bike_type",
        "structured_fields": ["bike_type", "bike"],
        "positive_terms": [
            "gravel bike", "road bike", "endurance bike", "mountain bike", "mtb",
            "hardtail", "full suspension", "bike type", "bike platform", "platform",
            "drop-bar", "drop bar",
        ],
        "negative_terms": ["bag", "bags", "sleep", "sleep system", "wheel", "wheels", "tyre", "tyres", "drivetrain", "cassette", "chainring", "navigation", "water"],
    },
}


def _first_nonempty(values: List[Optional[str]]) -> Optional[str]:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _clean_text(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    cleaned = " ".join(text.split()).strip(" \t\r\n")
    return cleaned or None


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        cleaned = _clean_text(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _support_label(count: int) -> str:
    if count <= 0:
        return "none"
    if count == 1:
        return "single"
    if count <= 3:
        return "weak_pattern"
    return "pattern"


def _merge_text(existing: Optional[str], addition: Optional[str]) -> Optional[str]:
    if not addition:
        return existing
    addition = addition.strip()
    if not addition:
        return existing
    if not existing or not existing.strip():
        return addition
    if addition in existing:
        return existing
    sep = "\n" if "\n" in existing else " "
    return f"{existing.rstrip()}{sep}{addition}"


def _text_supports_field(text: str, rule: Dict[str, Any]) -> bool:
    tl = text.lower()
    positive_hits = sum(1 for term in rule["positive_terms"] if term in tl)
    if positive_hits == 0:
        if rule["target_field"] == "tyres" and _TYRE_WIDTH_RE.search(tl):
            return not any(term in tl for term in rule["negative_terms"])
        return False
    if any(term in tl for term in rule["negative_terms"]):
        return False
    return True


def _strip_field_label(text: str, field_name: str) -> str:
    tl = text.lower()
    prefixes = {
        "tyres": ["tyres:", "tyre:", "tire:", "tires:"],
        "wheels": ["wheels:", "wheel:", "wheelset:"],
        "drivetrain": ["drivetrain:", "groupset:", "gearing:"],
        "bags": ["bags:", "bag:", "frame bag:", "seat pack:", "saddle bag:", "handlebar bag:", "top tube bag:"],
        "sleep_system": ["sleep system:", "sleep:", "sleeping bag:", "bivy:", "bivvy:", "quilt:", "mat:", "pad:", "tent:"],
        "bike_type": ["bike type:", "bike:", "platform:"],
    }.get(field_name, [])

    for prefix in prefixes:
        if tl.startswith(prefix):
            return text[len(prefix):].strip()
    return text.strip()


def _compact_field_value(field_name: str, text: str) -> str:
    rule = _COMPONENT_FILL_RULES[field_name]
    compact = text.strip()
    separators = [" with ", " plus ", " and "]

    for separator in separators:
        if separator not in compact.lower():
            continue
        head = compact.split(separator, 1)[0].strip(" \t\r\n-–—:;,")
        if head and _text_supports_field(head, rule):
            compact = head
            break

    return compact


def _normalize_candidate(field_name: str, text: str) -> str:
    cleaned = " ".join(text.split()).strip(" \t\r\n")
    cleaned = _strip_field_label(cleaned, field_name)
    cleaned = cleaned.strip(" \t\r\n-–—:;,.")
    cleaned = _compact_field_value(field_name, cleaned)
    return cleaned


def _candidate_texts_for_rider(
    rider: Any,
    field_name: str,
    rule: Dict[str, Any],
    include_key_items: bool = True,
    include_chunks: bool = True,
) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []

    if field_name == "bike_type":
        direct = _clean_text(getattr(rider, "bike_type", None) or getattr(rider, "bike", None))
        if direct:
            candidates.append(("structured", direct))
    else:
        for source_field in rule.get("structured_fields", []):
            value = _clean_text(getattr(rider, source_field, None))
            if value:
                candidates.append(("structured", value))

    if include_key_items:
        for item in getattr(rider, "key_items", None) or []:
            value = _clean_text(item)
            if value:
                candidates.append(("key_item", value))

    if include_chunks:
        for chunk in getattr(rider, "chunks", None) or []:
            text = _clean_text(getattr(chunk, "text", None) or getattr(chunk, "content", None))
            if text:
                candidates.append(("chunk", text[:160]))

    return candidates


def _rider_field_candidates(
    rider: Any,
    field_name: str,
    rule: Dict[str, Any],
    include_key_items: bool = True,
    include_chunks: bool = True,
) -> List[str]:
    normalized_candidates: List[str] = []

    for _, text in _candidate_texts_for_rider(
        rider,
        field_name,
        rule,
        include_key_items=include_key_items,
        include_chunks=include_chunks,
    ):
        if _text_supports_field(text, rule):
            normalized_candidates.append(_normalize_candidate(field_name, text))

    return _dedupe_preserve_order(normalized_candidates)


def _rider_field_examples(
    rider: Any,
    field_name: str,
    rule: Dict[str, Any],
    include_key_items: bool = True,
    include_chunks: bool = True,
) -> List[str]:
    examples: List[str] = []

    for source, text in _candidate_texts_for_rider(
        rider,
        field_name,
        rule,
        include_key_items=include_key_items,
        include_chunks=include_chunks,
    ):
        if source == "structured" and _text_supports_field(text, rule):
            examples.append(text)
        elif source in {"key_item", "chunk"} and _text_supports_field(text, rule):
            examples.append(text)

    return _dedupe_preserve_order(examples)


def _collect_field_evidence(
    riders: List[Any],
    field_name: str,
    include_key_items: bool = True,
    include_chunks: bool = True,
) -> Tuple[int, List[str]]:
    rule = _COMPONENT_FILL_RULES[field_name]
    support_count = 0
    candidates: List[str] = []

    for rider in riders or []:
        rider_candidates = _rider_field_candidates(
            rider,
            field_name,
            rule,
            include_key_items=include_key_items,
            include_chunks=include_chunks,
        )
        if rider_candidates:
            support_count += 1
            candidates.extend(rider_candidates)

    return support_count, _dedupe_preserve_order(candidates)


def _collect_field_examples(
    riders: List[Any],
    field_name: str,
    include_key_items: bool = True,
    include_chunks: bool = True,
) -> List[str]:
    rule = _COMPONENT_FILL_RULES[field_name]
    examples: List[str] = []
    for rider in riders or []:
        examples.extend(
            _rider_field_examples(
                rider,
                field_name,
                rule,
                include_key_items=include_key_items,
                include_chunks=include_chunks,
            )
        )
    return _dedupe_preserve_order(examples)


def _best_field_value(
    field_name: str,
    current_value: Optional[str],
    candidates: List[str],
) -> Optional[str]:
    rule = _COMPONENT_FILL_RULES[field_name]
    cleaned_current = _normalize_candidate(field_name, current_value) if isinstance(current_value, str) else None
    if cleaned_current and _text_supports_field(cleaned_current, rule):
        candidates = [cleaned_current] + candidates

    if not candidates:
        return cleaned_current if cleaned_current and _text_supports_field(cleaned_current, rule) else None

    counts = Counter(candidates)
    first_index: Dict[str, int] = {}
    for idx, value in enumerate(candidates):
        first_index.setdefault(value, idx)

    best_value = None
    best_score = None
    for value, count in counts.items():
        score = (count, -first_index[value], -len(value), value)
        if best_score is None or score > best_score:
            best_score = score
            best_value = value
    return best_value


def _prune_unrequested_fields(setup: Any, query_component: str) -> None:
    if query_component == "full_setup":
        return
    target_field = _COMPONENT_FILL_RULES.get(query_component, {}).get("target_field")
    if not target_field:
        return
    for field_name in _FIELD_NAMES:
        if field_name != target_field and hasattr(setup, field_name):
            setattr(setup, field_name, None)
    for extra_field in ["lighting", "navigation", "water_capacity"]:
        if hasattr(setup, extra_field):
            setattr(setup, extra_field, None)


def _policy_allows_examples(policy: Optional[RecommendationPolicy]) -> bool:
    if policy is None:
        return True
    return policy.allow_specific_specs


def _policy_allows_additional_text(policy: Optional[RecommendationPolicy]) -> bool:
    if policy is None:
        return True
    return policy.allow_specific_specs


def _append_component_synthesis(field_name: str, support_count: int, examples: List[str]) -> Optional[str]:
    if not examples:
        return None

    shown = examples[:3]
    if support_count <= 1 or len(shown) == 1:
        return f"Grounded example for {field_name}: {shown[0]}."

    if len(set(shown)) == 2:
        return f"Riders cluster around {field_name} values like {shown[0]} and {shown[1]}."

    return f"Riders show mixed but grounded {field_name} examples such as {', '.join(shown)}."


def _fill_requested_component_from_riders(
    rec: SetupRecommendation,
    riders: List[Any],
    query_component: str,
    policy: Optional[RecommendationPolicy] = None,
) -> SetupRecommendation:
    rs = rec.recommended_setup
    allow_examples = _policy_allows_examples(policy)
    allow_additional_text = _policy_allows_additional_text(policy)

    if query_component == "full_setup":
        evidence_parts: List[str] = []
        for field_name in _FIELD_NAMES:
            support_count, examples = _collect_field_evidence(
                riders,
                field_name,
                include_key_items=allow_examples,
                include_chunks=allow_examples,
            )
            field_examples = _collect_field_examples(
                riders,
                field_name,
                include_key_items=allow_examples,
                include_chunks=allow_examples,
            )
            current_value = getattr(rs, field_name, None)
            best = _best_field_value(field_name, current_value, examples)
            if best and best != current_value:
                setattr(rs, field_name, best)
            if allow_examples and support_count > 0 and field_examples:
                evidence_parts.append(f"{field_name} ({_support_label(support_count)}): {', '.join(field_examples[:2])}")

        if allow_additional_text and evidence_parts:
            rec.recommended_setup.notes = _merge_text(
                rec.recommended_setup.notes,
                "Grounded examples: " + "; ".join(evidence_parts),
            )
            rec.summary = _merge_text(
                rec.summary,
                "Grounded rider patterns support the filled setup fields.",
            ) or rec.summary
            rec.reasoning = _merge_text(
                rec.reasoning,
                "Examples come from rider structured fields first, with chunk text used only when it matches the same component.",
            ) or rec.reasoning
        return rec

    rule = _COMPONENT_FILL_RULES.get(query_component)
    if not rule:
        return rec

    target_field = rule["target_field"]
    current_value = getattr(rs, target_field, None)
    support_count, examples = _collect_field_evidence(
        riders,
        query_component,
        include_key_items=allow_examples,
        include_chunks=allow_examples,
    )
    field_examples = _collect_field_examples(
        riders,
        query_component,
        include_key_items=allow_examples,
        include_chunks=allow_examples,
    )
    best = _best_field_value(target_field, current_value, examples)
    if best and best != current_value:
        setattr(rs, target_field, best)

    _prune_unrequested_fields(rs, query_component)

    if allow_additional_text and field_examples:
        rec.recommended_setup.notes = _merge_text(
            rec.recommended_setup.notes,
            f"Grounded examples for {query_component}: " + "; ".join(field_examples[:3]),
        )
        rec.reasoning = _merge_text(
            rec.reasoning,
            _append_component_synthesis(query_component, support_count, examples),
        ) or rec.reasoning

    return rec
