import argparse
import json
import logging
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from baikpacking.agents.live_evaluation import classify_failure_kind
from baikpacking.agents.recommender_agent import recommend_setup_with_trace

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/eval/manual_scenarios.yaml")
DEFAULT_OUTPUT = Path("data/eval/scenario_runs.jsonl")
ScenarioEventMatchType = Literal["exact", "alias", "similar", "unknown"]
ScenarioRetrievalMode = Literal["exact_only", "exact_then_similar", "similar_only", "generic_fallback"]
ScenarioPolicyMode = Literal["strict_grounded", "pattern_based", "generic_fallback"]
ScenarioGuardedness = Literal["low", "medium", "high"]

_REQUIRED_SETUP_FIELDS = [
    "bike_type",
    "wheels",
    "tyres",
    "drivetrain",
    "bags",
    "sleep_system",
]
_STORAGE_RE = re.compile(
    r"\b(bag|bags|pack|pouch|frame|saddle|top tube|handlebar|feed|cargo)\b",
    re.IGNORECASE,
)
_DRIVETRAIN_RE = re.compile(
    r"(\b\d{1,2}\s*[-/]\s*\d{1,2}\b|\b\d{1,2}t\b|\b(?:cassette|chainring|groupset|gear|gearing|shimano|sram|grx|ultegra|deore|eagle)\b|\b[123]x\b)",
    re.IGNORECASE,
)
_TYRE_SIGNAL_RE = re.compile(
    r"(\bmm\b|\b\d{1,3}\.\d+\b|\b\d{2,3}(?:\.\d+)?\s?mm\b|\b\d{3}x\d{2,3}c\b|\b\d\.\d{1,2}\b|\b(?:tubeless|casing|tread|slick)\b)",
    re.IGNORECASE,
)
_LIGHTING_SIGNAL_RE = re.compile(
    r"(\blight\b|\blights\b|\bheadlight\b|\bfront light\b|\brear light\b|\blumen\b|\blux\b|\bdynamo\b|\bdynamo hub\b|\bexposure\b|\bsupernova\b|\blezyne\b|\bk-lite\b|\bsinewave\b|\bbeacon\b)",
    re.IGNORECASE,
)
_NAVIGATION_SIGNAL_RE = re.compile(
    r"(\bgarmin\b|\bwahoo\b|\belemnt\b|\betrex\b|\bgps\b|\bnavigation\b|\bnav\b|\broute\b|\bmaps\b|\bkomoot\b|\bosmand\b|\bbike computer\b|\bphone as backup\b)",
    re.IGNORECASE,
)
_GENERAL_NAVIGATION_TERMS_RE = re.compile(
    r"(\bgarmin\b|\bwahoo\b|\belemnt\b|\betrex\b|\bgps\b|\bnavigation\b|\bnav\b|\broute\b|\bmaps\b|\bkomoot\b|\bosmand\b|\bbike computer\b|\bphone\b)",
    re.IGNORECASE,
)
_GENERAL_LIGHTING_TERMS_RE = re.compile(
    r"(\blight\b|\blights\b|\bheadlight\b|\bfront light\b|\brear light\b|\blumen\b|\blux\b|\bdynamo\b|\bdynamo hub\b|\bexposure\b|\bsupernova\b|\blezyne\b|\bk-lite\b|\bsinewave\b|\bbeacon\b)",
    re.IGNORECASE,
)
_G_GENERIC_EVENT_RE = re.compile(
    r"^(unknown(?: event)?|generic|any|unspecified|n/?a|na|full setup|setup)$",
    re.IGNORECASE,
)
_POLICY_TO_GUARDEDNESS = {
    "strict_grounded": "low",
    "pattern_based": "medium",
    "generic_fallback": "high",
}


class ManualEvalScenario(BaseModel):
    """Typed contract for manual evaluation scenarios.

    The YAML contract stays hand-editable while the loader validates the
    scenario shape. Event match type, retrieval mode, policy mode, and
    guardedness are constrained values; `must_mention` and `must_not_do`
    remain explicit string lists for easy manual authoring.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    question: str
    expected_event: str
    expected_event_match_type: ScenarioEventMatchType
    expected_intent: str
    expected_component: str
    expected_policy_mode: ScenarioPolicyMode
    expected_retrieval_mode: ScenarioRetrievalMode
    expected_guardedness: ScenarioGuardedness
    must_mention: list[str] = Field(default_factory=list)
    must_not_do: list[str] = Field(default_factory=list)
    group: str | None = None
    why: str | None = None
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _alias_question_field(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "question" not in normalized and "query" in normalized:
            normalized["question"] = normalized["query"]
        return normalized


def _setup_logging() -> None:
    """Configure basic logging for CLI execution."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

def _is_strict_event_scenario(run: Mapping[str, Any]) -> bool:
    """Return True when strict exact-event alignment checks should count for this row."""
    expected_event = _normalize_event_name(run.get("expected_event"))
    if not expected_event or _G_GENERIC_EVENT_RE.match(expected_event):
        return False

    mode = _normalize_text(run.get("event_grounding_mode"))
    if not mode:
        mode = _scenario_event_grounding_mode(run)

    return mode == "exact_grounding_required"


def load_scenarios(path: str | Path) -> list[dict[str, Any]]:
    """Load manual evaluation scenarios from a YAML file."""
    scenario_path = Path(path)
    with scenario_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not payload:
        return []

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping at {scenario_path}, got {type(payload).__name__}")

    scenarios = payload.get("scenarios", [])
    if scenarios is None:
        return []
    if not isinstance(scenarios, list):
        raise ValueError(f"Expected 'scenarios' to be a list in {scenario_path}")

    out: list[dict[str, Any]] = []
    for idx, scenario in enumerate(scenarios, start=1):
        if not isinstance(scenario, dict):
            logger.warning("Skipping non-mapping scenario at index %s", idx)
            continue
        try:
            model = ManualEvalScenario.model_validate(scenario)
        except ValidationError as exc:
            raise ValueError(f"Invalid scenario at index {idx} in {scenario_path}: {exc}") from exc

        scenario_dict = model.model_dump()
        scenario_dict["query"] = model.question
        out.append(scenario_dict)
    return out


def _trace_calls(trace: Any) -> list[dict[str, Any]]:
    """Return a JSON-serializable list of trace call records."""
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


def extract_trace_step(trace: Any, tool_name: str) -> dict[str, Any] | None:
    """Extract the last trace step for a tool name, if present."""
    for step in reversed(_trace_calls(trace)):
        if isinstance(step, dict) and step.get("tool") == tool_name:
            return step
    return None


def _get_scenario_value(scenario: Mapping[str, Any], key: str) -> Any:
    """Return a scenario value or None when the field is absent."""
    value = scenario.get(key)
    if value is None and key == "query":
        value = scenario.get("question")
    if value is None and key == "question":
        value = scenario.get("query")
    return value if value is not None else None


def _missing_setup_fields(recommendation: Any) -> list[str] | None:
    """Return missing canonical setup fields, or None if the recommendation is unavailable."""
    if recommendation is None:
        return None

    setup = getattr(recommendation, "recommended_setup", None)
    if setup is None:
        return None

    missing: list[str] = []
    for field_name in _REQUIRED_SETUP_FIELDS:
        value = getattr(setup, field_name, None)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field_name)
    return missing


def _content_apply_row(run: Mapping[str, Any]) -> bool:
    """Return True when deterministic content assertions should be applied."""
    expected_intent = str(run.get("expected_intent") or "").strip().lower()
    intent_component = str(run.get("intent_component") or "").strip().lower()
    return expected_intent == "full_setup" or intent_component == "full_setup"


def _normalize_text(value: Any) -> str:
    """Normalize values to lowercase text for simple rule checks."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.split()).strip().lower()


def _normalize_event_name(value: Any) -> str:
    """Normalize event names for soft equality checks."""
    text = _normalize_text(value)
    text = re.sub(r"\b(19|20)\d{2}\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _get_setup_field(run: Mapping[str, Any], field_name: str) -> str:
    """Read a recommended_setup field from either a dict or a model-dumped row."""
    setup = run.get("recommended_setup") or {}
    if isinstance(setup, Mapping):
        return _normalize_text(setup.get(field_name))
    return _normalize_text(getattr(setup, field_name, None))


def _to_int(value: Any) -> int | None:
    """Best-effort integer coercion for trace fields."""
    if isinstance(value, bool) or value is None:
        return None if value is None else int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _has_any_signal(value: str, pattern: re.Pattern[str]) -> bool:
    """Return True when a field contains at least one field-appropriate signal."""
    if not value:
        return False
    return bool(pattern.search(value))


def _scenario_event_grounding_mode(run: Mapping[str, Any]) -> str:
    """Infer whether a scenario requires exact grounding or allows similar-event fallback."""
    match_type = _normalize_text(run.get("expected_event_match_type"))
    retrieval_mode = _normalize_text(run.get("expected_retrieval_mode"))

    if match_type in {"exact", "alias", "trusted_exact"}:
        return "exact_grounding_required"
    if retrieval_mode == "exact_only":
        return "exact_grounding_required"
    if match_type in {"similar", "unknown"} or retrieval_mode in {"exact_then_similar", "similar_only", "generic_fallback"}:
        return "similar_event_fallback_allowed"

    scenario_id = _normalize_text(run.get("scenario_id") or run.get("id"))
    group = _normalize_text(run.get("group"))
    expected_behavior = _normalize_text(run.get("expected_behavior"))
    why = _normalize_text(run.get("why"))
    notes = _normalize_text(run.get("notes"))
    blob = " ".join(part for part in [scenario_id, group, expected_behavior, why, notes] if part)

    alias_hints = [
        "alias",
        "variation",
        "fuzzy",
        "semantic alias",
        "short alias",
        "spell",
        "nickname",
    ]
    if "alias" in scenario_id or "alias" in group or any(hint in blob for hint in alias_hints):
        return "similar_event_fallback_allowed"
    return "exact_grounding_required"


def _scenario_actual_guardedness(run: Mapping[str, Any]) -> str | None:
    """Map the selected policy mode to a coarse guardedness level."""
    policy_mode = _normalize_text(run.get("policy_mode"))
    return _POLICY_TO_GUARDEDNESS.get(policy_mode)


def _scenario_terms(value: Any) -> list[str]:
    """Normalize a scenario term list to non-empty strings."""
    if not isinstance(value, list):
        return []
    terms: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            terms.append(text)
    return terms


def _failure_kind(error: str | Mapping[str, Any] | None) -> str | None:
    """Classify a failed eval row by the source of the failure."""
    return classify_failure_kind(error)


def _fallback_honesty_issues(
    *,
    expected_event: str,
    retrieval_source: str,
    summary: str,
    reasoning: str,
) -> list[str]:
    """Flag exact-grounding language when retrieval came from a fallback source."""
    if retrieval_source not in {"similar_event", "unknown_global"}:
        return []

    text = f"{summary} {reasoning}".lower()
    expected = _normalize_event_name(expected_event)
    if not expected:
        return []

    fallback_phrases = [
        "based on similar events",
        "similar event",
        "similar events",
        "for an event like",
        "similar ultra-endurance events suggest",
        "similar riders",
        "fallback",
        "using similar events",
    ]
    exact_grounding_claims = [
        rf"\bfor\s+{re.escape(expected)}\b",
        rf"\briders?\s+(?:in|for|from)\s+{re.escape(expected)}\b",
        rf"\b{re.escape(expected)}\s+riders?\b",
        rf"\bgrounded in\s+{re.escape(expected)}\b",
    ]

    has_fallback_framing = any(phrase in text for phrase in fallback_phrases)
    has_exact_claim = any(re.search(pattern, text) for pattern in exact_grounding_claims)

    if has_exact_claim and not has_fallback_framing:
        return [
            "similar_event_fallback_not_disclosed",
            "exact_grounding_claim_with_similar_event_retrieval",
        ]
    return []

def _row_text_blob(run: Mapping[str, Any]) -> str:
    """Build a normalized text blob for simple phrase checks."""
    parts: list[str] = []

    for key in ("summary", "reasoning"):
        value = run.get(key)
        if value:
            parts.append(_normalize_text(value))

    setup = run.get("recommended_setup")
    if isinstance(setup, Mapping):
        for value in setup.values():
            if value:
                parts.append(_normalize_text(value))
    elif setup is not None:
        parts.append(_normalize_text(setup))

    return " ".join(parts)

def evaluate_content_assertions(run: dict[str, Any]) -> dict[str, Any]:
    """Apply lightweight deterministic content checks to a scenario run row."""
    if not _content_apply_row(run):
        return {
            "content_assertions_passed": True,
            "content_assertion_issues": [],
            "content_assertion_issue_count": 0,
        }

    issues: list[str] = []

    bike_type = _get_setup_field(run, "bike_type")
    tyres = _get_setup_field(run, "tyres")
    bags = _get_setup_field(run, "bags")
    drivetrain = _get_setup_field(run, "drivetrain")

    if not bike_type:
        issues.append("missing_bike_type")
    if not tyres:
        issues.append("missing_tyres")
    if not bags:
        issues.append("missing_bags")

    mention_blob = _row_text_blob(run)
    for term in _scenario_terms(run.get("must_mention")):
        if _normalize_text(term) not in mention_blob:
            issues.append(f"missing_required_phrase:{term}")
    for term in _scenario_terms(run.get("must_not_do")):
        if _normalize_text(term) and _normalize_text(term) in mention_blob:
            issues.append(f"forbidden_phrase:{term}")

    if drivetrain and _STORAGE_RE.search(drivetrain):
        issues.append("drivetrain_contains_storage_terms")
    if drivetrain and _GENERAL_NAVIGATION_TERMS_RE.search(drivetrain):
        issues.append("drivetrain_contains_navigation_terms")
    if drivetrain and _GENERAL_LIGHTING_TERMS_RE.search(drivetrain):
        issues.append("drivetrain_contains_lighting_terms")
    if drivetrain and not _DRIVETRAIN_RE.search(drivetrain):
        issues.append("drivetrain_missing_drivetrain_signals")

    if bags and _DRIVETRAIN_RE.search(bags):
        issues.append("bags_contains_drivetrain_terms")

    if tyres and _STORAGE_RE.search(tyres):
        issues.append("tyres_contains_storage_terms")

    if not tyres or not _TYRE_SIGNAL_RE.search(tyres):
        issues.append("tyres_missing_tyre_signals")

    lighting = _get_setup_field(run, "lighting")
    if lighting:
        if _GENERAL_NAVIGATION_TERMS_RE.search(lighting):
            issues.append("lighting_contains_navigation_terms")
        if not _has_any_signal(lighting, _LIGHTING_SIGNAL_RE):
            issues.append("lighting_missing_lighting_signals")

    navigation = _get_setup_field(run, "navigation")
    if navigation:
        if _GENERAL_LIGHTING_TERMS_RE.search(navigation):
            issues.append("navigation_contains_lighting_terms")
        if not _has_any_signal(navigation, _NAVIGATION_SIGNAL_RE):
            issues.append("navigation_missing_navigation_signals")

    return {
        "content_assertions_passed": not issues,
        "content_assertion_issues": issues,
        "content_assertion_issue_count": len(issues),
    }


def evaluate_event_alignment_assertions(run: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic event-alignment checks to a scenario run row."""
    expected_event = _normalize_event_name(run.get("expected_event"))
    if not expected_event or _G_GENERIC_EVENT_RE.match(expected_event):
        return {
            "event_alignment_assertions_passed": True,
            "event_alignment_issues": [],
            "event_alignment_issue_count": 0,
        }

    issues: list[str] = []
    resolved_event = _normalize_event_name(run.get("resolved_event_name") or run.get("event_name"))
    matched_event = _normalize_event_name(run.get("matched_event_name"))
    exact_event_hit_count = run.get("exact_event_hit_count")
    exact_event_hit_count = _to_int(exact_event_hit_count)
    retrieval_source = _normalize_text(run.get("retrieval_source"))
    event_match_type = _normalize_text(run.get("event_match_type"))
    expected_event_match_type = _normalize_text(run.get("expected_event_match_type"))
    expected_retrieval_mode = _normalize_text(run.get("expected_retrieval_mode"))
    expected_policy_mode = _normalize_text(run.get("expected_policy_mode"))
    expected_guardedness = _normalize_text(run.get("expected_guardedness"))
    scenario_mode = _scenario_event_grounding_mode(run)

    if (
        scenario_mode == "similar_event_fallback_allowed"
        or expected_event_match_type in {"similar", "unknown"}
        or expected_retrieval_mode in {"similar_only", "generic_fallback"}
    ):
        if expected_event and resolved_event and expected_event != resolved_event:
            issues.append("matched_event_differs_from_expected_event")
        issues.extend(
            _fallback_honesty_issues(
                expected_event=expected_event,
                retrieval_source=retrieval_source,
                summary=_normalize_text(run.get("summary")),
                reasoning=_normalize_text(run.get("reasoning")),
            )
        )
    else:
        if exact_event_hit_count in {0, None}:
            issues.append("expected_exact_event_but_no_exact_hits")
        if retrieval_source == "similar_event":
            issues.append("expected_exact_event_but_used_similar_event")
        if event_match_type in {"exact", "trusted_exact"} and exact_event_hit_count in {0, None}:
            issues.append("event_match_type_exact_but_zero_exact_hits")
        if expected_event and matched_event and expected_event != matched_event:
            issues.append("matched_event_differs_from_expected_event")

    actual_policy_mode = _normalize_text(run.get("policy_mode"))
    if expected_policy_mode and actual_policy_mode and expected_policy_mode != actual_policy_mode:
        issues.append("policy_mode_does_not_match_expected_policy_mode")

    actual_guardedness = _scenario_actual_guardedness(run)
    if expected_guardedness and actual_guardedness and expected_guardedness != actual_guardedness:
        issues.append("guardedness_does_not_match_expected_guardedness")

    if expected_retrieval_mode == "exact_only":
        if not (retrieval_source == "exact_event" and exact_event_hit_count and exact_event_hit_count > 0):
            issues.append("retrieval_mode_does_not_match_expected_retrieval_mode")
    elif expected_retrieval_mode == "exact_then_similar":
        if not (
            (retrieval_source == "exact_event" and exact_event_hit_count and exact_event_hit_count > 0)
            or (retrieval_source == "similar_event" and exact_event_hit_count == 0)
        ):
            issues.append("retrieval_mode_does_not_match_expected_retrieval_mode")
    elif expected_retrieval_mode == "similar_only":
        if not (retrieval_source == "similar_event" and exact_event_hit_count == 0):
            issues.append("retrieval_mode_does_not_match_expected_retrieval_mode")
    elif expected_retrieval_mode == "generic_fallback":
        if retrieval_source != "unknown_global":
            issues.append("retrieval_mode_does_not_match_expected_retrieval_mode")

    return {
        "event_alignment_assertions_passed": not issues,
        "event_alignment_issues": issues,
        "event_alignment_issue_count": len(issues),
    }


def _jsonable(value: Any) -> Any:
    """Best-effort JSON-safe conversion. Returns None when the object is not serializable."""
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _jsonable(dump())
        except Exception:
            logger.debug("model_dump failed for %s", type(value).__name__, exc_info=True)
            return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return None


def extract_run_row(
    scenario: Mapping[str, Any],
    recommendation: Any,
    trace: Any,
    *,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    """Convert one scenario execution into a JSON-serializable row."""
    intent_step = extract_trace_step(trace, "intent_classification")
    search_step = extract_trace_step(trace, "search_similar_riders")
    evidence_step = extract_trace_step(trace, "evidence_summary")
    component_check_step = extract_trace_step(trace, "component_evidence_check")
    policy_step = extract_trace_step(trace, "policy_selection")
    writer_validation_step = extract_trace_step(trace, "writer_validation")
    writer_summary_step = extract_trace_step(trace, "writer_stage_summary")

    normalized_error = error or ""
    failure_kind = _failure_kind(error) if status == "failure" else None

    row: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "error": error,
        "failure_kind": failure_kind,
        "failure_stage": None,
        "failure_message": error,
        "output_schema_failure": failure_kind == "output_schema_failure",
        "runtime_failure": failure_kind == "runtime_failure",
        "schema_error_count": 0,
        "schema_error_paths": [],
        "schema_error_messages": [],
        "scenario_contract_valid": True,
        "scenario_id": _get_scenario_value(scenario, "id"),
        "group": _get_scenario_value(scenario, "group"),
        "event_grounding_mode": _scenario_event_grounding_mode(scenario),
        "question": _get_scenario_value(scenario, "question"),
        "query": _get_scenario_value(scenario, "query"),
        "why": _get_scenario_value(scenario, "why"),
        "expected_event": _get_scenario_value(scenario, "expected_event"),
        "expected_event_match_type": _get_scenario_value(scenario, "expected_event_match_type"),
        "expected_intent": _get_scenario_value(scenario, "expected_intent"),
        "expected_component": _get_scenario_value(scenario, "expected_component"),
        "expected_policy_mode": _get_scenario_value(scenario, "expected_policy_mode"),
        "expected_retrieval_mode": _get_scenario_value(scenario, "expected_retrieval_mode"),
        "expected_guardedness": _get_scenario_value(scenario, "expected_guardedness"),
        "must_mention": _get_scenario_value(scenario, "must_mention"),
        "must_not_do": _get_scenario_value(scenario, "must_not_do"),
        "expected_behavior": _get_scenario_value(scenario, "expected_behavior"),
        "notes": _get_scenario_value(scenario, "notes"),
        "requested_event_name": _get_scenario_value(scenario, "expected_event"),
        "resolved_event_name": getattr(recommendation, "event", None) if recommendation is not None else None,
        "event_name": getattr(recommendation, "event", None) if recommendation is not None else None,
        "summary": getattr(recommendation, "summary", None) if recommendation is not None else None,
        "reasoning": getattr(recommendation, "reasoning", None) if recommendation is not None else None,
        "missing_fields": _missing_setup_fields(recommendation),
        "recommended_setup": _jsonable(getattr(recommendation, "recommended_setup", None))
        if recommendation is not None
        else None,
        "grounding_riders": _jsonable(getattr(recommendation, "similar_riders", None))
        if recommendation is not None
        else None,
        "intent_component": None,
        "intent_confidence": None,
        "retrieval_query": None,
        "retrieved_rider_count": None,
        "component_hit_count": None,
        "evidence_strength": None,
        "consistency": None,
        "field_support": None,
        "policy_mode": None,
        "event_match_type": None,
        "matched_event_name": None,
        "retrieval_source": None,
        "retrieval_mode": None,
        "knowledge_base_exact_match": None,
        "matched_reference_event": None,
        "exact_event_hit_count": None,
        "writer_call_count": None,
        "writer_first_pass_ok": None,
        "writer_validation_failed": None,
        "writer_second_pass_triggered": None,
        "writer_second_pass_reason": None,
        "writer_total_ms": None,
        "raw_trace_steps": _jsonable(_trace_calls(trace)) if trace is not None else None,
        "raw_result_type": type(recommendation).__name__ if recommendation is not None else None,
    }

    if intent_step:
        intent_result = intent_step.get("result") if isinstance(intent_step.get("result"), dict) else {}
        row["intent_component"] = intent_result.get("component")
        row["intent_confidence"] = intent_result.get("confidence")

    if search_step:
        search_args = search_step.get("args", {}) if isinstance(search_step.get("args"), dict) else {}
        search_result = search_step.get("result", {}) if isinstance(search_step.get("result"), dict) else {}
        row["retrieval_query"] = search_args.get("query")
        row["retrieved_rider_count"] = search_result.get("count")

    if component_check_step:
        component_result = (
            component_check_step.get("result")
            if isinstance(component_check_step.get("result"), dict)
            else {}
        )
        row["component_hit_count"] = component_result.get("component_hit_count")

    if evidence_step:
        evidence_result = evidence_step.get("result") if isinstance(evidence_step.get("result"), dict) else {}
        row["field_support"] = _jsonable(evidence_result.get("field_support"))
        row["evidence_strength"] = evidence_result.get("evidence_strength")
        row["consistency"] = evidence_result.get("consistency")

    if policy_step:
        policy_result = policy_step.get("result") if isinstance(policy_step.get("result"), dict) else {}
        policy_args = policy_step.get("args", {}) if isinstance(policy_step.get("args"), dict) else {}
        row["policy_mode"] = policy_result.get("mode")
        row["event_match_type"] = policy_args.get("event_match_type")
        row["matched_event_name"] = policy_args.get("matched_event_name")
        row["matched_reference_event"] = policy_args.get("matched_event_name")
        row["retrieval_source"] = policy_args.get("retrieval_source")
        row["retrieval_mode"] = policy_args.get("retrieval_source")
        row["exact_event_hit_count"] = policy_args.get("exact_event_hit_count")
        hit_count = _to_int(policy_args.get("exact_event_hit_count"))
        row["knowledge_base_exact_match"] = None if hit_count is None else hit_count > 0

    if writer_validation_step:
        writer_result = (
            writer_validation_step.get("result")
            if isinstance(writer_validation_step.get("result"), dict)
            else {}
        )
        row["writer_call_count"] = writer_result.get("writer_call_count")
        row["writer_first_pass_ok"] = writer_result.get("writer_first_pass_ok")
        row["writer_validation_failed"] = writer_result.get("writer_validation_failed")
        row["writer_second_pass_triggered"] = writer_result.get("writer_second_pass_triggered")
        row["writer_second_pass_reason"] = writer_result.get("writer_second_pass_reason")

    if writer_summary_step:
        writer_result = (
            writer_summary_step.get("result")
            if isinstance(writer_summary_step.get("result"), dict)
            else {}
        )
        row["writer_call_count"] = writer_result.get("writer_call_count", row["writer_call_count"])
        row["writer_first_pass_ok"] = writer_result.get("writer_first_pass_ok", row["writer_first_pass_ok"])
        row["writer_validation_failed"] = writer_result.get(
            "writer_validation_failed", row["writer_validation_failed"]
        )
        row["writer_second_pass_triggered"] = writer_result.get(
            "writer_second_pass_triggered", row["writer_second_pass_triggered"]
        )
        row["writer_second_pass_reason"] = writer_result.get(
            "writer_second_pass_reason", row["writer_second_pass_reason"]
        )
        row["writer_total_ms"] = writer_result.get("writer_total_ms")

    if status == "failure":
        error_lower = normalized_error.lower()

        if row["writer_validation_failed"] is True:
            row["failure_stage"] = "writer_output_validation"
        elif row["writer_second_pass_triggered"] is True and row["writer_first_pass_ok"] is False:
            row["failure_stage"] = "writer_repair_pass"
        elif "setuprecommendation" in error_lower or "output validation" in error_lower:
            row["failure_stage"] = "writer_output_validation"
        elif "validationerror" in error_lower:
            row["failure_stage"] = "output_schema_validation"
        else:
            row["failure_stage"] = "runtime"

        schema_paths: list[str] = []
        schema_messages: list[str] = []

        for line in normalized_error.split(" | "):
            line = line.strip()
            if not line:
                continue

            if line.startswith("ValidationError at "):
                rest = line.removeprefix("ValidationError at ").strip()
                if ": " in rest:
                    path, msg = rest.split(": ", 1)
                else:
                    path, msg = rest, ""
                schema_paths.append(path.strip())
                schema_messages.append(msg.strip())
                continue

            match = re.match(r"^([A-Za-z0-9_\.\[\]-]+)\s*:\s*(.+)$", line)
            if match and "." in match.group(1):
                schema_paths.append(match.group(1).strip())
                schema_messages.append(match.group(2).strip())

        row["schema_error_paths"] = schema_paths
        row["schema_error_messages"] = schema_messages
        row["schema_error_count"] = len(schema_paths)

        if failure_kind == "output_schema_failure":
            row["output_schema_failure"] = True
            row["runtime_failure"] = False

    row.update(evaluate_content_assertions(row))
    row.update(evaluate_event_alignment_assertions(row))

    return row


def run_one_scenario(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Run one scenario through the current recommender and return a diagnostic row."""
    query = scenario.get("question") or scenario.get("query")
    if not query or not str(query).strip():
        logger.error("Scenario %s is missing a query", scenario.get("id"))
        return extract_run_row(
            scenario,
            None,
            None,
            status="failure",
            error="Scenario is missing a non-empty query",
        )

    try:
        recommendation, trace = recommend_setup_with_trace(str(query))
        return extract_run_row(
            scenario,
            recommendation,
            trace,
            status="success",
            error=None,
        )
    except Exception as exc:
        logger.exception("Scenario %s failed", scenario.get("id"))

        error_parts = [f"{type(exc).__name__}: {exc}"]

        cause = exc.__cause__
        if cause is not None:
            error_parts.append(f"Caused by {type(cause).__name__}: {cause}")

            if isinstance(cause, ValidationError):
                for err in cause.errors():
                    loc = ".".join(str(part) for part in err.get("loc", []))
                    msg = err.get("msg", "unknown validation error")
                    err_type = err.get("type", "unknown")
                    error_parts.append(f"ValidationError at {loc}: {msg} [{err_type}]")

        context = exc.__context__
        if context is not None and context is not cause:
            error_parts.append(f"Context {type(context).__name__}: {context}")

            if isinstance(context, ValidationError):
                for err in context.errors():
                    loc = ".".join(str(part) for part in err.get("loc", []))
                    msg = err.get("msg", "unknown validation error")
                    err_type = err.get("type", "unknown")
                    error_parts.append(f"ValidationError at {loc}: {msg} [{err_type}]")

        return extract_run_row(
            scenario,
            None,
            None,
            status="failure",
            error=" | ".join(error_parts),
        )

def write_jsonl(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write rows to a JSONL file, replacing any previous file contents."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _normalize_ids(values: Sequence[str] | None) -> set[str]:
    """Normalize CLI ids into a string set."""
    out: set[str] = set()
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                out.add(item)
    return out


def main() -> None:
    """CLI entrypoint for manual scenario evaluation."""
    _setup_logging()

    parser = argparse.ArgumentParser(description="Run manual eval scenarios through the recommender.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to the YAML file containing top-level 'scenarios'.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to the JSONL output file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of scenarios to run after filtering.",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Optional scenario ids to run. Accepts space-separated or comma-separated ids.",
    )
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    scenarios = load_scenarios(args.input)
    selected_ids = _normalize_ids(args.ids)
    if selected_ids:
        scenarios = [s for s in scenarios if str(s.get("id")) in selected_ids]

    if args.seed is not None:
        random.seed(args.seed)

    if args.limit is not None and args.limit < len(scenarios):
        scenarios = random.sample(scenarios, args.limit)

    logger.info("Loaded %d scenarios from %s", len(scenarios), args.input)

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_id = scenario.get("id")
        logger.info("Running scenario %s", scenario_id)
        rows.append(run_one_scenario(scenario))

    write_jsonl(rows, args.output)

    total = len(rows)
    successes = sum(1 for row in rows if row.get("status") == "success")
    failures = total - successes

    failure_kind_counts: dict[str, int] = {}
    for row in rows:
        if row.get("status") != "failure":
            continue
        kind = row.get("failure_kind") or "unknown_failure"
        failure_kind_counts[str(kind)] = failure_kind_counts.get(str(kind), 0) + 1

    output_schema_failures = failure_kind_counts.get("output_schema_failure", 0)
    runtime_failures = failure_kind_counts.get("runtime_failure", 0)
    unknown_failures = failure_kind_counts.get("unknown_failure", 0)

    content_applied = sum(1 for row in rows if _content_apply_row(row))
    content_passed = sum(
        1 for row in rows if _content_apply_row(row) and row.get("content_assertions_passed") is True
    )
    content_failed = sum(
        1 for row in rows if _content_apply_row(row) and row.get("content_assertions_passed") is False
    )
    content_issue_count = sum(int(row.get("content_assertion_issue_count") or 0) for row in rows)

    event_alignment_applied = sum(1 for row in rows if _is_strict_event_scenario(row))
    event_alignment_passed = sum(
        1 for row in rows if _is_strict_event_scenario(row) and row.get("event_alignment_assertions_passed") is True
    )
    event_alignment_failed = sum(
        1 for row in rows if _is_strict_event_scenario(row) and row.get("event_alignment_assertions_passed") is False
    )
    event_alignment_issue_count = sum(int(row.get("event_alignment_issue_count") or 0) for row in rows)

    failure_summary = " ".join(
        f"{kind}={count}" for kind, count in sorted(failure_kind_counts.items())
    )

    print(
        "total="
        f"{total} successes={successes} failures={failures} "
        f"output_schema_failures={output_schema_failures} "
        f"runtime_failures={runtime_failures} "
        f"unknown_failures={unknown_failures} "
        f"{failure_summary} "
        f"content_assertions_passed={content_passed}/{content_applied} "
        f"content_assertions_failed={content_failed} "
        f"content_assertion_issues={content_issue_count} "
        f"event_alignment_passed={event_alignment_passed}/{event_alignment_applied} "
        f"event_alignment_failed={event_alignment_failed} "
        f"event_alignment_issues={event_alignment_issue_count} "
        f"output={Path(args.output)}"
    )


if __name__ == "__main__":
    main()
