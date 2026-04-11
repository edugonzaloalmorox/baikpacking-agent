"""Deterministic live evaluation records for real recommendation queries."""


import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

DEFAULT_LIVE_RUNS_PATH = Path(__file__).resolve().parents[3] / "data/eval/live_runs.jsonl"
_REQUIRED_SETUP_FIELDS = [
    "bike_type",
    "wheels",
    "tyres",
    "drivetrain",
    "bags",
    "sleep_system",
]
_COMPONENT_TERMS = {
    "tyres": [
        "tyre",
        "tyres",
        "tire",
        "tires",
    ],
    "bags": [
        "bag",
        "bags",
        "frame bag",
        "seat pack",
        "saddle bag",
        "handlebar bag",
        "top tube bag",
    ],
    "drivetrain": [
        "drivetrain",
        "groupset",
        "cassette",
        "chainring",
        "gearing",
        "gear",
    ],
    "sleep_system": [
        "sleep",
        "sleep system",
        "sleeping bag",
        "bivy",
        "bivvy",
        "quilt",
        "mat",
        "pad",
        "tent",
    ],
    "wheels": [
        "wheel",
        "wheels",
        "wheelset",
        "rim",
        "rims",
        "hub",
        "hubs",
        "650b",
        "700c",
        "29er",
        "27.5",
    ],
    "bike_type": [
        "bike type",
        "gravel",
        "hardtail",
        "mtb",
        "mountain bike",
        "road bike",
        "endurance",
        "drop bar",
        "drop-bar",
    ],
}
_COMPONENT_QUERY_PATTERNS = {
    component: re.compile("|".join(re.escape(term) for term in terms), re.IGNORECASE)
    for component, terms in _COMPONENT_TERMS.items()
}
_FALLBACK_HONESTY_PHRASES = [
    "based on similar events",
    "similar event",
    "similar events",
    "for an event like",
    "similar riders",
    "fallback",
    "using similar events",
]
_EXACT_GROUNDING_CLAIMS = [
    r"\bfor\s+{expected}\b",
    r"\briders?\s+(?:in|for|from)\s+{expected}\b",
    r"\b{expected}\s+riders?\b",
    r"\bgrounded in\s+{expected}\b",
]


class LiveRunRecord(BaseModel):
    """Structured record for one live recommender query."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    timestamp: str
    query: str
    status: str
    error: Optional[str] = None
    failure_kind: Optional[str] = None
    guard_type: str = ""
    guard_reason: str = ""
    resolved_event_name: str = ""
    event_match_type: str = ""
    retrieval_source: str = ""
    retrieval_mode: str = ""
    policy_mode: str = ""
    query_component: str = ""
    recommended_setup: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    reasoning: str = ""
    rider_count: int = 0
    component_hit_count: int = 0
    evidence_strength: str = ""
    evidence_consistency: str = ""
    setup_complete: bool = False
    setup_is_partial: bool = False
    missing_fields: list[str] = Field(default_factory=list)
    component_relevance: dict[str, Any] = Field(default_factory=dict)
    retrieval_policy_issues: list[str] = Field(default_factory=list)
    quality_issue_codes: list[str] = Field(default_factory=list)
    raw_trace_steps: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: Optional[float] = None
    request_meta: dict[str, Any] = Field(default_factory=dict)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.split()).strip().lower()


def _safe_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _safe_json(dump())
        except Exception:
            logger.debug("model_dump failed for %s", type(value).__name__, exc_info=True)
            return None
    return None


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None if value is None else int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _trace_calls(trace: Any) -> list[dict[str, Any]]:
    if trace is None:
        return []
    calls = getattr(trace, "calls", None)
    if isinstance(calls, list):
        return calls
    entries = getattr(trace, "entries", None)
    if isinstance(entries, list):
        return entries
    if isinstance(trace, list):
        return trace
    return []


def _find_trace_step(trace: Any, tool_name: str) -> dict[str, Any] | None:
    for step in reversed(_trace_calls(trace)):
        if isinstance(step, dict) and step.get("tool") == tool_name:
            return step
    return None


def _trace_result_dict(trace: Any, tool_name: str) -> dict[str, Any]:
    step = _find_trace_step(trace, tool_name)
    if not step:
        return {}
    result = step.get("result")
    return result if isinstance(result, dict) else {}


def classify_failure_kind(error: str | Mapping[str, Any] | None) -> str | None:
    """Classify a failed request using the same coarse taxonomy as scenario eval."""
    if not error:
        return None

    if isinstance(error, Mapping):
        error_type = _normalize_text(error.get("error_type"))
        failure_stage = _normalize_text(error.get("failure_stage"))
        message = _normalize_text(error.get("message"))
        schema_errors = error.get("schema_errors")

        if error_type in {"output_schema_failure", "schema_validation"}:
            return "output_schema_failure"
        if failure_stage in {"writer_output_validation", "output_schema_validation", "writer_repair_pass"}:
            return "output_schema_failure"
        if isinstance(schema_errors, list) and schema_errors:
            return "output_schema_failure"

        normalized = " ".join(part for part in [error_type, failure_stage, message] if part)
        if (
            "validationerror" in normalized
            or "unexpectedmodelbehavior" in normalized
            or "output validation" in normalized
            or "schema" in normalized
            or "setuprecommendation" in normalized
        ):
            return "output_schema_failure"
        return "runtime_failure"

    normalized = str(error).lower()
    if (
        "validationerror" in normalized
        or "unexpectedmodelbehavior" in normalized
        or "output validation" in normalized
        or "schema" in normalized
        or "setuprecommendation" in normalized
    ):
        return "output_schema_failure"
    return "runtime_failure"


def _missing_setup_fields(recommended_setup: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for field_name in _REQUIRED_SETUP_FIELDS:
        value = recommended_setup.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field_name)
    return missing


def _combined_response_text(response: Mapping[str, Any]) -> str:
    parts: list[str] = []
    recommendation = response.get("recommendation")
    if isinstance(recommendation, Mapping):
        for field in ("summary", "reasoning"):
            value = recommendation.get(field)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        setup = recommendation.get("recommended_setup")
        if isinstance(setup, Mapping):
            for value in setup.values():
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
    return " ".join(parts).lower()


def _query_components(query: str, query_component: str) -> list[str]:
    components = [component for component, pattern in _COMPONENT_QUERY_PATTERNS.items() if pattern.search(query)]
    query_component = _normalize_text(query_component)
    if query_component and query_component != "full_setup" and query_component not in components:
        components.append(query_component)
    return components


def _component_addressed(response: Mapping[str, Any], component: str) -> bool:
    recommendation = response.get("recommendation")
    if not isinstance(recommendation, Mapping):
        return False

    setup = recommendation.get("recommended_setup")
    if isinstance(setup, Mapping):
        field_value = setup.get(component)
        if isinstance(field_value, str) and field_value.strip():
            return True

    text = _combined_response_text(response)
    return any(term in text for term in _COMPONENT_TERMS.get(component, []))


def _component_relevance(response: Mapping[str, Any], query: str, query_component: str) -> dict[str, Any]:
    signals = _query_components(query, query_component)
    addressed = [component for component in signals if _component_addressed(response, component)]
    missing = [component for component in signals if component not in addressed]
    issues = [f"missing_component:{component}" for component in missing]
    return {
        "query_components": signals,
        "addressed_components": addressed,
        "missing_components": missing,
        "passed": not missing,
        "issues": issues,
    }


def _retrieval_mode(retrieval_source: str, event_match_type: str, exact_event_hit_count: int | None) -> str:
    if retrieval_source == "exact_event":
        return "exact_only"
    if retrieval_source == "similar_event":
        if _normalize_text(event_match_type) in {"exact", "alias", "trusted_exact"}:
            return "exact_then_similar"
        if exact_event_hit_count and exact_event_hit_count > 0:
            return "exact_then_similar"
        return "similar_only"
    return "generic_fallback"


def _fallback_honesty_issues(*, expected_event: str, retrieval_source: str, summary: str, reasoning: str) -> list[str]:
    if retrieval_source not in {"similar_event", "unknown_global"}:
        return []

    text = f"{summary} {reasoning}".lower()
    expected = _normalize_text(expected_event)
    if not expected:
        return []

    has_fallback_framing = any(phrase in text for phrase in _FALLBACK_HONESTY_PHRASES)
    exact_grounding_claims = [
        pattern.format(expected=re.escape(expected))
        for pattern in _EXACT_GROUNDING_CLAIMS
    ]
    has_exact_claim = any(re.search(pattern, text) for pattern in exact_grounding_claims)

    if has_exact_claim and not has_fallback_framing:
        return [
            "similar_event_fallback_not_disclosed",
            "exact_grounding_claim_with_similar_event_retrieval",
        ]
    return []


def _retrieval_policy_issues(
    response: Mapping[str, Any],
    *,
    exact_event_hit_count: int | None,
) -> list[str]:
    issues: list[str] = []
    policy = response.get("policy")
    resolved_event = response.get("resolved_event")
    recommendation = response.get("recommendation")
    if not isinstance(policy, Mapping):
        policy = {}
    if not isinstance(resolved_event, Mapping):
        resolved_event = {}
    if not isinstance(recommendation, Mapping):
        recommendation = {}

    policy_mode = _normalize_text(policy.get("mode"))
    notes = policy.get("notes")
    retrieval_source = ""
    matched_event_name = ""
    event_match_type = _normalize_text(resolved_event.get("match_type"))
    if isinstance(notes, list):
        for note in notes:
            text = _normalize_text(note)
            if text in {"exact_event_retrieval", "similar_event_retrieval", "unknown_event"}:
                retrieval_source = {
                    "exact_event_retrieval": "exact_event",
                    "similar_event_retrieval": "similar_event",
                    "unknown_event": "unknown_global",
                }[text]
            elif text.startswith("matched_event="):
                matched_event_name = text.split("=", 1)[1].strip()
            elif text.startswith("event_match_type="):
                event_match_type = text.split("=", 1)[1].strip()
            elif text == "strong_evidence":
                pass
            elif text.endswith("_evidence"):
                pass

    if not retrieval_source:
        if event_match_type in {"exact", "alias", "trusted_exact"}:
            retrieval_source = "exact_event"
        elif event_match_type in {"similar", "fuzzy_candidate", "weak_candidate"}:
            retrieval_source = "similar_event"
        else:
            retrieval_source = "unknown_global"

    if not matched_event_name:
        matched_event_name = _normalize_text(
            resolved_event.get("canonical_name")
            or resolved_event.get("display_name")
            or recommendation.get("event")
        )

    if retrieval_source == "unknown_global" and policy_mode == "strict_grounded":
        issues.append("unknown_global_with_strict_grounded_policy")

    if _normalize_text(resolved_event.get("match_type")) in {"exact", "trusted_exact"} and retrieval_source != "exact_event":
        issues.append("exact_match_not_grounded_exactly")

    if retrieval_source == "exact_event" and exact_event_hit_count == 0:
        issues.append("exact_event_source_with_zero_exact_hits")

    if retrieval_source in {"similar_event", "unknown_global"} and policy_mode == "strict_grounded":
        issues.append("fallback_policy_too_strict")

    summary = str(recommendation.get("summary") or "")
    reasoning = str(recommendation.get("reasoning") or "")
    issues.extend(
        _fallback_honesty_issues(
            expected_event=matched_event_name or str(resolved_event.get("display_name") or recommendation.get("event") or ""),
            retrieval_source=retrieval_source,
            summary=summary,
            reasoning=reasoning,
        )
    )

    if retrieval_source == "exact_event" and policy_mode == "generic_fallback":
        issues.append("exact_event_retrieval_with_generic_fallback_policy")

    return issues


def build_live_run_record(
    *,
    run_id: str | None = None,
    query: str,
    status: str,
    error: str | Mapping[str, Any] | None,
    response: Mapping[str, Any] | None,
    trace: Any,
    latency_ms: float | None,
    request_meta: Mapping[str, Any] | None = None,
) -> LiveRunRecord:
    """Build a deterministic live-eval record from a real recommendation run."""
    record_run_id = run_id or uuid.uuid4().hex
    response = response or {}
    request_meta = dict(request_meta or {})
    recommendation = response.get("recommendation") if isinstance(response, Mapping) else {}
    resolved_event = response.get("resolved_event") if isinstance(response, Mapping) else {}
    intent = response.get("intent") if isinstance(response, Mapping) else {}
    evidence = response.get("evidence") if isinstance(response, Mapping) else {}
    policy = response.get("policy") if isinstance(response, Mapping) else {}
    guard = response.get("guard") if isinstance(response, Mapping) else {}
    if not isinstance(recommendation, Mapping):
        recommendation = {}
    if not isinstance(resolved_event, Mapping):
        resolved_event = {}
    if not isinstance(intent, Mapping):
        intent = {}
    if not isinstance(evidence, Mapping):
        evidence = {}
    if not isinstance(policy, Mapping):
        policy = {}
    if not isinstance(guard, Mapping):
        guard = {}

    event_step = _trace_result_dict(trace, "event_resolution")
    intent_step = _trace_result_dict(trace, "intent_classification")
    guard_step = _trace_result_dict(trace, "guard_decision")

    policy_step = _find_trace_step(trace, "policy_selection")
    policy_args = policy_step.get("args", {}) if isinstance(policy_step, dict) and isinstance(policy_step.get("args"), dict) else {}
    policy_result = policy_step.get("result", {}) if isinstance(policy_step, dict) and isinstance(policy_step.get("result"), dict) else {}

    retrieval_source = _normalize_text(policy_args.get("retrieval_source"))
    event_match_type = _normalize_text(policy_args.get("event_match_type") or resolved_event.get("match_type"))
    matched_event_name = _normalize_text(
        policy_args.get("matched_event_name")
        or resolved_event.get("canonical_name")
        or resolved_event.get("display_name")
        or recommendation.get("event")
    )
    exact_event_hit_count = _to_int(policy_args.get("exact_event_hit_count"))
    guard_type = _normalize_text(guard.get("guard_type") or guard_step.get("guard_type"))
    guard_reason = str(guard.get("reason") or guard_step.get("reason") or "")

    if status == "skipped":
        return LiveRunRecord(
            run_id=record_run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
            status=status,
            error=None if error is None else str(error),
            failure_kind="guard_blocked",
            guard_type=guard_type,
            guard_reason=guard_reason,
            resolved_event_name=str(
                event_step.get("display_name")
                or event_step.get("canonical_name")
                or resolved_event.get("display_name")
                or recommendation.get("event")
                or ""
            ),
            event_match_type=_normalize_text(
                event_step.get("match_type")
                or resolved_event.get("match_type")
                or "unknown"
            ),
            retrieval_source="",
            retrieval_mode="",
            policy_mode="",
            query_component=_normalize_text(intent_step.get("component") or intent.get("component")),
            recommended_setup={},
            summary="",
            reasoning="",
            rider_count=0,
            component_hit_count=0,
            evidence_strength="",
            evidence_consistency="",
            setup_complete=False,
            setup_is_partial=False,
            missing_fields=[],
            component_relevance={},
            retrieval_policy_issues=[],
            quality_issue_codes=[],
            raw_trace_steps=_safe_json(_trace_calls(trace)) or [],
            latency_ms=latency_ms,
            request_meta=_safe_json(request_meta) or {},
        )

    if not retrieval_source:
        if event_match_type in {"exact", "alias", "trusted_exact"}:
            retrieval_source = "exact_event"
        elif event_match_type in {"similar", "fuzzy_candidate", "weak_candidate"}:
            retrieval_source = "similar_event"
        else:
            retrieval_source = "unknown_global"

    live_retrieval_mode = _retrieval_mode(retrieval_source, event_match_type, exact_event_hit_count)
    policy_mode = _normalize_text(policy.get("mode") or policy_result.get("mode"))

    recommended_setup = recommendation.get("recommended_setup") if isinstance(recommendation, Mapping) else {}
    if not isinstance(recommended_setup, Mapping):
        recommended_setup = {}

    missing_fields = _missing_setup_fields(recommended_setup)
    setup_complete = not missing_fields
    component_relevance = _component_relevance(response, query, _normalize_text(intent.get("component")))
    retrieval_policy_issues = _retrieval_policy_issues(
        response,
        exact_event_hit_count=exact_event_hit_count,
    )
    quality_issue_codes = [
        *component_relevance.get("issues", []),
        *retrieval_policy_issues,
    ]

    raw_trace_steps = _trace_calls(trace)
    failure_kind = classify_failure_kind(error) if status == "failure" else None

    return LiveRunRecord(
        run_id=record_run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        query=query,
        status=status,
        error=None if error is None else str(error),
        failure_kind=failure_kind,
        guard_type=guard_type,
        guard_reason=guard_reason,
        resolved_event_name=str(resolved_event.get("display_name") or recommendation.get("event") or ""),
        event_match_type=event_match_type,
        retrieval_source=retrieval_source,
        retrieval_mode=live_retrieval_mode,
        policy_mode=policy_mode,
        query_component=_normalize_text(intent.get("component")),
        recommended_setup=_safe_json(recommended_setup) or {},
        summary=str(recommendation.get("summary") or ""),
        reasoning=str(recommendation.get("reasoning") or ""),
        rider_count=_to_int(evidence.get("rider_count")) or 0,
        component_hit_count=_to_int(evidence.get("component_hit_count")) or 0,
        evidence_strength=str(evidence.get("evidence_strength") or ""),
        evidence_consistency=str(evidence.get("consistency") or ""),
        setup_complete=setup_complete,
        setup_is_partial=not setup_complete,
        missing_fields=missing_fields,
        component_relevance=component_relevance,
        retrieval_policy_issues=retrieval_policy_issues,
        quality_issue_codes=quality_issue_codes,
        raw_trace_steps=_safe_json(raw_trace_steps) or [],
        latency_ms=latency_ms,
        request_meta=_safe_json(request_meta) or {},
    )


def append_live_run(record: LiveRunRecord, path: str | Path | None = None) -> None:
    """Append one live-run record to the JSONL sink."""
    run_path = Path(path or DEFAULT_LIVE_RUNS_PATH)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    with run_path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json())
        handle.write("\n")
