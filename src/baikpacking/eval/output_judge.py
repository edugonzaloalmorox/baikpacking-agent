"""LLM-based output judge for live bikepacking recommendations."""

import asyncio
import json
import logging
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from baikpacking.agents.live_feedback import LiveFeedbackRecord, latest_feedback_by_run_id, load_live_feedback
from baikpacking.agents.recommender_agent import settings as recommender_settings
from baikpacking.logging_config import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_LIVE_RUNS_PATH = Path(__file__).resolve().parents[3] / "data/eval/live_runs.jsonl"
DEFAULT_LIVE_FEEDBACK_PATH = Path(__file__).resolve().parents[3] / "data/eval/live_feedback.jsonl"
DEFAULT_OUTPUT_JUDGMENTS_PATH = Path(__file__).resolve().parents[3] / "data/eval/output_judgments.jsonl"
DEFAULT_JUDGE_MODEL_NAME = recommender_settings.writer_model

Verdict = Literal["pass", "weak", "fail"]
FailureType = Literal["none", "relevance", "grounding", "usefulness", "missing_component"]


class OutputJudgeResult(BaseModel):
    """Structured result returned by the output judge."""

    model_config = ConfigDict(extra="ignore")

    relevance_score: int = Field(ge=0, le=2)
    grounding_honesty_score: int = Field(ge=0, le=2)
    usefulness_score: int = Field(ge=0, le=2)
    missing_requested_component: bool
    hallucination_or_overclaim: bool
    verdict: Verdict
    failure_type: FailureType
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _normalize_reason(cls, value: str) -> str:
        text = " ".join(str(value).split()).strip()
        if not text:
            raise ValueError("reason must not be empty")
        return text


class OutputJudgeInput(BaseModel):
    """Compact judge input built from a live run record."""

    model_config = ConfigDict(extra="ignore")

    query: str
    query_component: str = ""
    policy_mode: str = ""
    resolved_event_name: str = ""
    event_match_type: str = ""
    retrieval_source: str = ""
    retrieval_mode: str = ""
    rider_count: int = 0
    component_hit_count: int = 0
    evidence_strength: str = ""
    evidence_consistency: str = ""
    missing_fields: list[str] = Field(default_factory=list)
    component_relevance: dict[str, Any] = Field(default_factory=dict)
    retrieval_policy_issues: list[str] = Field(default_factory=list)
    recommended_setup: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    reasoning: str = ""
    final_answer_text: str = ""


class OutputJudgeRecord(BaseModel):
    """JSONL row written by the output judge."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    run_timestamp: str = ""
    judge_timestamp: str
    query: str
    resolved_event_name: str = ""
    event_match_type: str = ""
    retrieval_source: str = ""
    policy_mode: str = ""
    query_component: str = ""
    feedback: str = ""
    feedback_comment: str = ""
    model_name: str
    selection_reason: str = ""
    relevance_score: int = Field(ge=0, le=2)
    grounding_honesty_score: int = Field(ge=0, le=2)
    usefulness_score: int = Field(ge=0, le=2)
    missing_requested_component: bool
    hallucination_or_overclaim: bool
    verdict: Verdict
    failure_type: FailureType
    reason: str = Field(min_length=1)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_normalize_text(item) for item in value if _normalize_text(item)]
    if isinstance(value, tuple):
        return [_normalize_text(item) for item in value if _normalize_text(item)]
    text = _normalize_text(value)
    return [text] if text else []


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
            return None
    return None


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL rows, skipping malformed lines."""
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def load_live_runs(path: str | Path) -> list[dict[str, Any]]:
    """Load live runs from JSONL."""
    return load_jsonl_records(path)


def _latest_feedback_map(feedback_records: Iterable[LiveFeedbackRecord]) -> dict[str, LiveFeedbackRecord]:
    return latest_feedback_by_run_id(list(feedback_records))


def join_runs_with_feedback(
    runs: Iterable[Mapping[str, Any]],
    feedback_records: Iterable[LiveFeedbackRecord],
) -> list[dict[str, Any]]:
    """Attach latest feedback metadata to run dictionaries."""
    latest = _latest_feedback_map(feedback_records)
    joined: list[dict[str, Any]] = []
    for run in runs:
        run_id = _normalize_text(run.get("run_id"))
        feedback = latest.get(run_id)
        merged = dict(run)
        merged["feedback"] = feedback.feedback if feedback else ""
        merged["feedback_comment"] = feedback.comment if feedback else ""
        merged["feedback_timestamp"] = feedback.timestamp if feedback else ""
        joined.append(merged)
    return joined


def render_recommended_setup(recommended_setup: Mapping[str, Any]) -> str:
    """Render a compact human-readable setup summary."""
    if not isinstance(recommended_setup, Mapping):
        return ""

    labels = [
        ("bike_type", "Bike type"),
        ("wheels", "Wheels"),
        ("tyres", "Tyres"),
        ("drivetrain", "Drivetrain"),
        ("bags", "Bags"),
        ("sleep_system", "Sleep system"),
        ("lighting", "Lighting"),
        ("navigation", "Navigation"),
        ("water_capacity", "Water capacity"),
        ("notes", "Notes"),
    ]
    lines: list[str] = []
    for key, label in labels:
        value = recommended_setup.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(f"{label}: {value.strip()}")
    return "\n".join(lines)


def build_compact_judge_input(run: Mapping[str, Any]) -> OutputJudgeInput:
    """Convert a live run record into a compact judge input payload."""
    recommended_setup = run.get("recommended_setup")
    if not isinstance(recommended_setup, Mapping):
        recommended_setup = {}

    summary = _normalize_text(run.get("summary"))
    reasoning = _normalize_text(run.get("reasoning"))
    final_answer_parts = [part for part in [summary, render_recommended_setup(recommended_setup)] if part]

    return OutputJudgeInput(
        query=_normalize_text(run.get("query")),
        query_component=_normalize_text(run.get("query_component")),
        policy_mode=_normalize_text(run.get("policy_mode")),
        resolved_event_name=_normalize_text(run.get("resolved_event_name")),
        event_match_type=_normalize_text(run.get("event_match_type")),
        retrieval_source=_normalize_text(run.get("retrieval_source")),
        retrieval_mode=_normalize_text(run.get("retrieval_mode")),
        rider_count=_to_int(run.get("rider_count")),
        component_hit_count=_to_int(run.get("component_hit_count")),
        evidence_strength=_normalize_text(run.get("evidence_strength")),
        evidence_consistency=_normalize_text(run.get("evidence_consistency")),
        missing_fields=_as_string_list(run.get("missing_fields")),
        component_relevance=_safe_json(run.get("component_relevance")) or {},
        retrieval_policy_issues=_as_string_list(run.get("retrieval_policy_issues")),
        recommended_setup=_safe_json(recommended_setup) or {},
        summary=summary,
        reasoning=reasoning,
        final_answer_text="\n\n".join(final_answer_parts),
    )


def build_output_judge_prompt(payload: OutputJudgeInput) -> str:
    """Build a strict JSON-only prompt for the output judge."""
    rubric = {
        "relevance_score": "0=no answer, 1=partial, 2=clearly answers",
        "grounding_honesty_score": "0=unsupported/invented, 1=mostly cautious but slightly overstated, 2=honest and calibrated",
        "usefulness_score": "0=vague, 1=somewhat useful, 2=useful and actionable",
        "flags": {
            "missing_requested_component": "true when the requested component is not actually covered",
            "hallucination_or_overclaim": "true for unsupported brands, fabricated details, unsupported event claims, or excessive certainty",
        },
        "verdict_rules": [
            "fail if hallucination_or_overclaim is true",
            "fail if grounding_honesty_score == 0",
            "fail if relevance_score == 0",
            "weak if usefulness_score == 0",
            "weak if missing_requested_component is true",
            "otherwise pass",
        ],
        "failure_type_rules": [
            '"grounding" if hallucination_or_overclaim is true or grounding_honesty_score == 0',
            '"relevance" if relevance_score == 0',
            '"missing_component" if missing_requested_component is true',
            '"usefulness" if usefulness_score == 0',
            '"none" otherwise',
        ],
    }
    return (
        "You are evaluating the final answer of a bikepacking recommender.\n"
        "Judge the FINAL ANSWER only. Do not judge retrieval separately.\n"
        "Return strict JSON only, matching this schema exactly:\n"
        "{"
        '"relevance_score": 0, '
        '"grounding_honesty_score": 0, '
        '"usefulness_score": 0, '
        '"missing_requested_component": false, '
        '"hallucination_or_overclaim": false, '
        '"verdict": "pass", '
        '"failure_type": "none", '
        '"reason": "..."'
        "}\n\n"
        f"Rubric:\n{json.dumps(rubric, indent=2, ensure_ascii=False)}\n\n"
        f"INPUT:\n{json.dumps(payload.model_dump(mode='json'), indent=2, ensure_ascii=False)}"
    )


def _build_judge_agent() -> Agent:
    model = OpenAIChatModel(DEFAULT_JUDGE_MODEL_NAME)
    system_prompt = (
        "You are a strict output judge for bikepacking recommendations.\n"
        "Use only the provided input JSON.\n"
        "Return strict JSON only and no extra text.\n"
        "Judge the final answer, not retrieval."
    )
    return Agent(model=model, output_type=OutputJudgeResult, system_prompt=system_prompt)


judge_agent = _build_judge_agent()


def apply_judge_rules(result: OutputJudgeResult) -> OutputJudgeResult:
    """Normalize verdict and failure type using the repository's explicit rules."""
    verdict = "pass"
    failure_type: FailureType = "none"

    if result.hallucination_or_overclaim or result.grounding_honesty_score == 0:
        verdict = "fail"
        failure_type = "grounding"
    elif result.relevance_score == 0:
        verdict = "fail"
        failure_type = "relevance"
    elif result.missing_requested_component:
        verdict = "weak"
        failure_type = "missing_component"
    elif result.usefulness_score == 0:
        verdict = "weak"
        failure_type = "usefulness"

    return result.model_copy(update={"verdict": verdict, "failure_type": failure_type})


async def _run_output_judge_for_run_async(
    run: Mapping[str, Any],
    *,
    feedback: Mapping[str, Any] | None = None,
) -> OutputJudgeResult:
    payload = build_compact_judge_input(run)
    prompt = build_output_judge_prompt(payload)
    result = await judge_agent.run(prompt)
    judge_result = result.output
    if not isinstance(judge_result, OutputJudgeResult):
        judge_result = OutputJudgeResult.model_validate(_safe_json(judge_result) or {})
    return apply_judge_rules(judge_result)


def run_output_judge_for_run(
    run: Mapping[str, Any],
    *,
    feedback: Mapping[str, Any] | None = None,
) -> OutputJudgeResult:
    """Run the judge for one live run."""
    return asyncio.run(_run_output_judge_for_run_async(run, feedback=feedback))


def select_runs_for_judging(
    runs: Iterable[Mapping[str, Any]],
    feedback_records: Iterable[LiveFeedbackRecord],
    *,
    max_runs: int | None = None,
    sample_rate: float = 0.1,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Select a small set of live runs to judge."""
    joined = join_runs_with_feedback(runs, feedback_records)
    rng = random.Random(seed)

    prioritized: list[dict[str, Any]] = []
    sampled: list[dict[str, Any]] = []
    for run in joined:
        if _normalize_text(run.get("status")) != "success":
            continue

        reasons: list[str] = []
        if _normalize_text(run.get("feedback")) == "thumbs_down":
            reasons.append("negative_feedback")
        if run.get("retrieval_policy_issues"):
            reasons.append("retrieval_policy_issues")

        retrieval_source = _normalize_text(run.get("retrieval_source"))
        policy_mode = _normalize_text(run.get("policy_mode"))
        if retrieval_source != "exact_event" or policy_mode in {"pattern_based", "generic_fallback"}:
            reasons.append("fallback_like")

        selection_reason = ",".join(reasons)
        enriched = dict(run)
        enriched["selection_reason"] = selection_reason

        if reasons:
            prioritized.append(enriched)
        elif rng.random() < max(0.0, min(1.0, sample_rate)):
            enriched["selection_reason"] = "baseline_sample"
            sampled.append(enriched)

    selected = prioritized + sampled
    if max_runs is not None:
        selected = selected[: max(0, int(max_runs))]
    return selected


def build_output_judge_record(
    run: Mapping[str, Any],
    judge_result: OutputJudgeResult,
    *,
    model_name: str,
    judge_timestamp: str | None = None,
) -> OutputJudgeRecord:
    """Build a JSONL-friendly record for one judged run."""
    run_feedback = _normalize_text(run.get("feedback"))
    judge_payload = judge_result.model_dump(mode="json")
    return OutputJudgeRecord(
        run_id=_normalize_text(run.get("run_id")),
        run_timestamp=_normalize_text(run.get("timestamp")),
        judge_timestamp=judge_timestamp or datetime.now(timezone.utc).isoformat(),
        query=_normalize_text(run.get("query")),
        resolved_event_name=_normalize_text(run.get("resolved_event_name")),
        event_match_type=_normalize_text(run.get("event_match_type")),
        retrieval_source=_normalize_text(run.get("retrieval_source")),
        policy_mode=_normalize_text(run.get("policy_mode")),
        query_component=_normalize_text(run.get("query_component")),
        feedback=run_feedback,
        feedback_comment=_normalize_text(run.get("feedback_comment")),
        model_name=model_name,
        selection_reason=_normalize_text(run.get("selection_reason")),
        **judge_payload,
    )


def _load_existing_judged_run_ids(path: str | Path) -> set[str]:
    return {
        _normalize_text(row.get("run_id"))
        for row in load_jsonl_records(path)
        if _normalize_text(row.get("run_id"))
    }


def _append_jsonl_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_safe_json(record) or {}, ensure_ascii=False))
        handle.write("\n")


def write_output_judgments(
    judgments: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Append judgments to JSONL, skipping duplicates unless force is set."""
    path = Path(output_path)
    existing_run_ids = set() if force else _load_existing_judged_run_ids(path)
    written: list[dict[str, Any]] = []

    for judgment in judgments:
        run_id = _normalize_text(judgment.get("run_id"))
        if not run_id:
            continue
        if not force and run_id in existing_run_ids:
            continue
        payload = _safe_json(judgment)
        if isinstance(payload, dict):
            _append_jsonl_record(path, payload)
            written.append(payload)
    return written


async def _run_output_judge_batch_async(
    *,
    runs_path: str | Path = DEFAULT_LIVE_RUNS_PATH,
    feedback_path: str | Path = DEFAULT_LIVE_FEEDBACK_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_JUDGMENTS_PATH,
    max_runs: int | None = 25,
    sample_rate: float = 0.1,
    seed: int = 0,
    force: bool = False,
) -> list[dict[str, Any]]:
    runs = load_live_runs(runs_path)
    feedback_records = load_live_feedback(feedback_path)
    selected = select_runs_for_judging(
        runs,
        feedback_records,
        max_runs=max_runs,
        sample_rate=sample_rate,
        seed=seed,
    )

    existing_run_ids = set() if force else _load_existing_judged_run_ids(output_path)
    written: list[dict[str, Any]] = []
    for run in selected:
        run_id = _normalize_text(run.get("run_id"))
        if not run_id:
            continue
        if not force and run_id in existing_run_ids:
            continue
        judge_result = await _run_output_judge_for_run_async(run, feedback=run)
        record = build_output_judge_record(
            run,
            judge_result,
            model_name=DEFAULT_JUDGE_MODEL_NAME,
        ).model_dump(mode="json")
        _append_jsonl_record(Path(output_path), record)
        written.append(record)
        existing_run_ids.add(run_id)
    return written


def run_output_judge_batch(
    *,
    runs_path: str | Path = DEFAULT_LIVE_RUNS_PATH,
    feedback_path: str | Path = DEFAULT_LIVE_FEEDBACK_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_JUDGMENTS_PATH,
    max_runs: int | None = 25,
    sample_rate: float = 0.1,
    seed: int = 0,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Run the judge batch synchronously."""
    return asyncio.run(
        _run_output_judge_batch_async(
            runs_path=runs_path,
            feedback_path=feedback_path,
            output_path=output_path,
            max_runs=max_runs,
            sample_rate=sample_rate,
            seed=seed,
            force=force,
        )
    )


def summarize_output_judgments(path: str | Path) -> dict[str, Any]:
    """Compute compact aggregate metrics for the output judgments file."""
    rows = load_jsonl_records(path)
    if not rows:
        return {
            "total_judged": 0,
            "pass_rate": 0.0,
            "weak_rate": 0.0,
            "fail_rate": 0.0,
            "mean_relevance_score": 0.0,
            "mean_grounding_honesty_score": 0.0,
            "mean_usefulness_score": 0.0,
            "failure_type_distribution": {},
            "by_policy_mode": {},
            "by_feedback": {},
        }

    total = len(rows)
    verdict_counts = Counter(_normalize_text(row.get("verdict")) for row in rows)
    failure_counts = Counter(_normalize_text(row.get("failure_type")) for row in rows)
    policy_counts: dict[str, Counter[str]] = defaultdict(Counter)
    feedback_counts: dict[str, Counter[str]] = defaultdict(Counter)
    relevance_scores: list[int] = []
    grounding_scores: list[int] = []
    usefulness_scores: list[int] = []

    for row in rows:
        relevance_scores.append(int(row.get("relevance_score") or 0))
        grounding_scores.append(int(row.get("grounding_honesty_score") or 0))
        usefulness_scores.append(int(row.get("usefulness_score") or 0))
        policy_counts[_normalize_text(row.get("policy_mode"))][_normalize_text(row.get("verdict"))] += 1
        feedback_counts[_normalize_text(row.get("feedback"))][_normalize_text(row.get("verdict"))] += 1

    def _rate(key: str) -> float:
        return verdict_counts.get(key, 0) / total if total else 0.0

    return {
        "total_judged": total,
        "pass_count": verdict_counts.get("pass", 0),
        "weak_count": verdict_counts.get("weak", 0),
        "fail_count": verdict_counts.get("fail", 0),
        "pass_rate": _rate("pass"),
        "weak_rate": _rate("weak"),
        "fail_rate": _rate("fail"),
        "verdict_counts": dict(verdict_counts),
        "mean_relevance_score": mean(relevance_scores),
        "mean_grounding_honesty_score": mean(grounding_scores),
        "mean_usefulness_score": mean(usefulness_scores),
        "failure_type_distribution": dict(failure_counts),
        "by_policy_mode": {mode or "unknown": dict(counts) for mode, counts in policy_counts.items()},
        "by_feedback": {label or "none": dict(counts) for label, counts in feedback_counts.items()},
    }


def _format_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        f"total_judged: {summary.get('total_judged', 0)}",
        f"pass_count: {summary.get('pass_count', 0)}",
        f"weak_count: {summary.get('weak_count', 0)}",
        f"fail_count: {summary.get('fail_count', 0)}",
        f"pass_rate: {summary.get('pass_rate', 0.0):.2f}",
        f"weak_rate: {summary.get('weak_rate', 0.0):.2f}",
        f"fail_rate: {summary.get('fail_rate', 0.0):.2f}",
        f"mean_relevance_score: {summary.get('mean_relevance_score', 0.0):.2f}",
        f"mean_grounding_honesty_score: {summary.get('mean_grounding_honesty_score', 0.0):.2f}",
        f"mean_usefulness_score: {summary.get('mean_usefulness_score', 0.0):.2f}",
        "failure_type_distribution:",
    ]
    for key, value in sorted((summary.get("failure_type_distribution") or {}).items()):
        lines.append(f"  {key}: {value}")
    lines.append("by_policy_mode:")
    for key, value in sorted((summary.get("by_policy_mode") or {}).items()):
        lines.append(f"  {key}: {value}")
    lines.append("by_feedback:")
    for key, value in sorted((summary.get("by_feedback") or {}).items()):
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def print_summary(path: str | Path) -> None:
    """Print terminal-friendly aggregate metrics for a judgments file."""
    summary = summarize_output_judgments(path)
    print(_format_summary(summary))


def main() -> None:
    """CLI entrypoint for the output judge."""
    import argparse

    setup_logging()

    parser = argparse.ArgumentParser(description="Run the bikepacking output judge.")
    parser.add_argument("--runs-path", default=str(DEFAULT_LIVE_RUNS_PATH))
    parser.add_argument("--feedback-path", default=str(DEFAULT_LIVE_FEEDBACK_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_JUDGMENTS_PATH))
    parser.add_argument("--max-runs", type=int, default=25)
    parser.add_argument("--sample-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Print summary metrics for the output file and exit.")
    args = parser.parse_args()

    if args.summary:
        print_summary(args.output_path)
        return

    written = asyncio.run(
        _run_output_judge_batch_async(
            runs_path=args.runs_path,
            feedback_path=args.feedback_path,
            output_path=args.output_path,
            max_runs=args.max_runs,
            sample_rate=args.sample_rate,
            seed=args.seed,
            force=args.force,
        )
    )
    summary = summarize_output_judgments(args.output_path)
    print(f"evaluated: {len(written)}")
    print(_format_summary(summary))


if __name__ == "__main__":
    main()
