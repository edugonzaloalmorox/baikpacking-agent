import json
import logging
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import anyio
import logfire
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from baikpacking.agents.event_resolution import (
    KNOWN_EVENTS,
    _clean_event_candidate,
    _count_titleish_words,
    _event_hint_descriptors,
    _extract_capitalized_spans,
    _extract_event_name as _extract_event_name_impl,
    _extract_known_event_alias,
    _is_valid_event_name,
    _looks_like_event_name,
    _score_event_candidate,
    resolve_event,
)
from baikpacking.agents.event_context_resolution import (
    _build_descriptor_query as _build_descriptor_query_impl,
    fetch_event_context_summary,
    infer_event_archetype as _infer_event_archetype_impl,
)
from baikpacking.agents.models import SetupCore, SetupRecommendation, WriterRecommendationDraft
from baikpacking.agents.output_validation import _fill_requested_component_from_riders
from baikpacking.agents.postprocess import (
    _infer_event_from_riders as _postprocess_infer_event_from_riders,
    _infer_year_from_title,
    _postprocess_recommendation as _postprocess_recommendation_impl,
)
from baikpacking.agents.policy import select_policy
from baikpacking.agents.review_feedback import (
    DEFAULT_REVIEWS_PATH,
    find_relevant_reviews,
    format_review_context,
    load_reviews,
)
from baikpacking.agents.query_intent import _build_retrieval_intent_bundle, _classify_query_intent
from baikpacking.agents.evidence_summary import _rider_component_hit_count, summarize_evidence
from baikpacking.agents.orchestration_models import RetrievalExecutionResult
from baikpacking.agents.retrieval_planning import build_retrieval_plan
from baikpacking.agents.writer_input import WriterInput, _compact_riders
from baikpacking.embedding import embed_text
from baikpacking.logging_config import setup_logging
from baikpacking.tools.call_trace import CallTrace, record_trace_call, time_and_record
from baikpacking.tools.pg_vector_search import PgVectorSearchDeps
from baikpacking.tools.riders import run_search_similar_riders

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


class AgentSettings(BaseSettings):
    """Settings for the bikepacking recommender agent."""

    writer_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AGENT_",
        extra="ignore",
    )


settings = AgentSettings()

DEFAULT_TOP_K_RIDERS = 5
DEFAULT_MAX_CHUNKS_PER_RIDER = 2
DEFAULT_TOP_K_CHUNKS = 80

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _normalize_event_title(value: Optional[str]) -> str:
    text = (value or "").lower()
    text = re.sub(r"\b(19|20)\d{2}\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _infer_retrieval_grounding(
    event_resolution: Any,
    event_context_summary: Any,
    riders: List[Any],
    fallback_used: bool,
) -> Dict[str, Any]:
    event_keys = [
        getattr(event_resolution, "canonical_name", None),
        getattr(event_resolution, "display_name", None),
        getattr(event_resolution, "raw_query_event", None),
    ]
    normalized_event_keys = {
        key for key in (_normalize_event_title(key) for key in event_keys) if key
    }

    title_counts: Counter[str] = Counter()
    exact_scope_hit_count = 0
    exact_event_hit_count = 0

    for rider in riders or []:
        source_scope = getattr(rider, "_source_scope", None)
        if source_scope == "exact_event":
            exact_scope_hit_count += 1

        title = getattr(rider, "event_title", None)
        normalized_title = _normalize_event_title(title)
        if not normalized_title:
            continue
        title_counts[title] += 1
        if any(
            normalized_title == key or normalized_title in key or key in normalized_title
            for key in normalized_event_keys
        ):
            exact_event_hit_count += 1

    if exact_scope_hit_count > 0 or exact_event_hit_count > 0:
        retrieval_source = "exact_event"
    elif fallback_used or getattr(event_context_summary, "similar_events", None) or getattr(event_context_summary, "event_family", None):
        retrieval_source = "similar_event"
    else:
        retrieval_source = "unknown_global"

    if exact_scope_hit_count > 0:
        matched_event_name = getattr(event_resolution, "canonical_name", None) or getattr(
            event_resolution,
            "display_name",
            None,
        )
    else:
        matched_event_name = title_counts.most_common(1)[0][0] if title_counts else None

    return {
        "retrieval_source": retrieval_source,
        "exact_event_hit_count": max(exact_scope_hit_count, exact_event_hit_count),
        "matched_event_name": matched_event_name,
    }


def _append_unique(items: List[str], value: Optional[str]) -> None:
    if not value:
        return
    value = value.strip()
    if value and value not in items:
        items.append(value)


def _has_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def _extract_event_name(user_query: str) -> str:
    return _extract_event_name_impl(user_query)

def _query_surface_hint(user_query: str) -> Optional[str]:
    q = f" {(user_query or '').lower()} "
    if " road " in q or " all-road " in q or " all road " in q:
        return "road"
    if " gravel " in q:
        return "gravel"
    if " trail " in q or " mtb " in q or " mountain bike " in q:
        return "trail"
    return None


def _infer_year_from_title(title: Optional[str]) -> Optional[int]:
    if not title:
        return None
    match = _YEAR_RE.search(title)
    return int(match.group(0)) if match else None


def _infer_event_from_riders(rec: SetupRecommendation) -> Optional[str]:
    return _postprocess_infer_event_from_riders(rec)


def _postprocess_recommendation(rec: SetupRecommendation) -> SetupRecommendation:
    return _postprocess_recommendation_impl(rec)


def infer_event_archetype(flags: Dict[str, bool]) -> Dict[str, Any]:
    return _infer_event_archetype_impl(flags)


def _keyword_flags(text: str) -> Dict[str, bool]:
    from baikpacking.agents.event_context_resolution import _keyword_flags as _impl
    return _impl(text)


def _extract_metrics(text: str) -> Dict[str, Optional[int]]:
    from baikpacking.agents.event_context_resolution import _extract_metrics as _impl
    return _impl(text)


def _surface_descriptors(surface_family: str) -> List[str]:
    from baikpacking.agents.event_context_resolution import _surface_descriptors as _impl
    return _impl(surface_family)


def _flag_descriptors(flags: Dict[str, bool]) -> List[str]:
    from baikpacking.agents.event_context_resolution import _flag_descriptors as _impl
    return _impl(flags)


def _metric_descriptors(metrics: Dict[str, Optional[int]]) -> List[str]:
    from baikpacking.agents.event_context_resolution import _metric_descriptors as _impl
    return _impl(metrics)


def _build_descriptor_query(
    event_name: str,
    event_context: str,
    user_question: str,
) -> Dict[str, Any]:
    return _build_descriptor_query_impl(
        event_name=event_name,
        event_context=event_context,
        user_question=user_question,
    )


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def _writer_missing_fields(rec: Optional[Any]) -> List[str]:
    if rec is None or rec.recommended_setup is None:
        return ["recommended_setup"]

    missing: List[str] = []
    for field_name in ["bike_type", "wheels", "lights", "tyres", "drivetrain", "bags", "sleep_system"]:
        if not _clean_text(getattr(rec.recommended_setup, field_name, None)):
            missing.append(field_name)
    return missing


def _writer_validation_issues(rec: Optional[Any]) -> List[str]:
    issues: List[str] = []
    if rec is None:
        return ["no_writer_output"]

    if not _clean_text(rec.summary):
        issues.append("empty_summary")
    if not _clean_text(rec.reasoning):
        issues.append("empty_reasoning")
    if rec.recommended_setup is None or rec.recommended_setup.is_empty():
        issues.append("empty_recommended_setup")
    return issues


def _writer_snapshot(rec: Optional[Any]) -> Optional[Dict[str, Any]]:
    if rec is None:
        return None

    return {
        "event": rec.event,
        "summary": rec.summary,
        "reasoning": rec.reasoning,
        "recommended_setup": rec.recommended_setup.model_dump() if rec.recommended_setup else None,
        "missing_fields": _writer_missing_fields(rec),
    }


def _build_writer_repair_prompt(
    writer_input_json: str,
    first_draft: Optional[SetupRecommendation],
    issues: List[str],
    error_text: Optional[str] = None,
    review_context: Optional[str] = None,
) -> str:
    payload = {
        "issues": issues,
        "error": error_text,
        "first_draft": _writer_snapshot(first_draft),
        "writer_input": json.loads(writer_input_json),
    }
    if review_context:
        payload["review_feedback"] = review_context
    return (
        "Repair the previous bikepacking recommendation draft.\n"
        "Return only valid JSON matching WriterRecommendationDraft.\n"
        "Keep all grounded claims tied to the provided similar_riders.\n"
        "Do not invent gear, brands, or specs.\n"
        "If a field still cannot be grounded, leave it empty/null and explain that in reasoning.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _build_writer_prompt(writer_input_json: str, review_context: Optional[str] = None) -> str:
    """Build the writer prompt with an optional human-review hint block."""
    prompt = WRITER_PROMPT
    if review_context:
        prompt = (
            f"{prompt}\n\n"
            "Human review hints:\n"
            f"{review_context}"
        )
    return f"{prompt}\n\n{writer_input_json}"


def _run_writer_call(
    prompt: str,
    *,
    deps: Any,
    stage: str,
    args: Dict[str, Any],
) -> Tuple[Any, float]:
    """Run one writer call and record its timing in the call trace."""
    t0 = time.perf_counter()
    try:
        result = writer_agent.run_sync(prompt).output
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        record_trace_call(
            deps=deps,
            tool_name=stage,
            args=args,
            result={
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            elapsed_ms=elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    record_trace_call(
        deps=deps,
        tool_name=stage,
        args=args,
        result={
            "ok": True,
            "output_type": type(result).__name__,
        },
        elapsed_ms=elapsed_ms,
    )
    return result, elapsed_ms


WRITER_PROMPT = """
You are a bikepacking equipment and ultra-distance cycling expert.

Return only valid JSON matching WriterRecommendationDraft.

Grounding rules:
- ALL gear details must come from similar_riders.
- event_context is only for understanding the target event characteristics.
- query_component tells you what part of the setup the user is asking about.
- component_hit_count tells you how many retrieved riders explicitly mention the requested component.
- Do not invent gear, brands, or specs.
- If a field cannot be grounded, leave it empty/null and explain that in reasoning.
- If component_hit_count is 0 for a component-specific question, say evidence is sparse and avoid specific grounded claims.
- grounding_mode tells you whether the recommendation is grounded in exact_event, similar_event, or unknown_global retrieval.
- If grounding_mode is similar_event or unknown_global, do not phrase the answer as if it came from exact event evidence.
- Prefer concise output:
  - summary: 3-4 sentences
  - reasoning: 3-5 sentences

Output rules:
- event must be the requested event_name
- only mention similar events if the exact event is absent from the retrieved riders
- recommended_setup should contain as many grounded fields as possible without guessing
Writer-owned fields: event, summary, reasoning, recommended_setup.
Code-assembled fields: similar_riders, trace metadata, postprocessing additions, and validation details.
""".strip()

writer_model = OpenAIChatModel(settings.writer_model)

writer_agent = Agent(
    model=writer_model,
    output_type=WriterRecommendationDraft,
    system_prompt=WRITER_PROMPT,
    retries=0,
    output_retries=0,
)


class _RecommenderAgentCompat:
    """Compatibility shim for older tests and callers.

    The active runtime entrypoints are recommend_setup() and
    recommend_setup_with_trace(). This wrapper preserves the old
    async .run(...) contract by returning an object with .output.
    """

    async def run(self, user_query: str) -> Any:
        rec, _trace = await anyio.to_thread.run_sync(recommend_setup_with_trace, user_query)
        return type("CompatResult", (), {"output": rec})()


recommender_agent = _RecommenderAgentCompat()


def _build_deps(call_trace: Optional[CallTrace] = None) -> PgVectorSearchDeps:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set.")

    return PgVectorSearchDeps(
        embed_query=embed_text,
        database_url=database_url,
        call_trace=call_trace,
    )


def recommend_setup_with_trace(user_query: str) -> Tuple[SetupRecommendation, CallTrace]:
    with logfire.span("recommender.run", user_query=user_query):
        trace = CallTrace()
        deps = _build_deps(call_trace=trace)

        event_resolution = resolve_event(user_query)
        event_name = event_resolution.display_name
        intent = _classify_query_intent(user_query)

        record_trace_call(
            deps=deps,
            tool_name="intent_classification",
            args={"user_query": user_query},
            result=intent.model_dump(),
            elapsed_ms=0.0,
        )

        event_context_summary = time_and_record(
            deps=deps,
            tool_name="event_web_search",
            args={"event_title": event_name},
            fn=lambda: fetch_event_context_summary(
                event_resolution=event_resolution,
                deps=deps,
            ),
        )

        event_context_text = event_context_summary.web_context_text

        retrieval_plan = build_retrieval_plan(
            event_resolution=event_resolution,
            event_context_summary=event_context_summary,
            intent=intent,
            user_query=user_query,
        )

        first_query = retrieval_plan.primary_query
        second_query = retrieval_plan.fallback_query

        top_k_riders = DEFAULT_TOP_K_RIDERS
        max_chunks_per_rider = DEFAULT_MAX_CHUNKS_PER_RIDER
        top_k_chunks = DEFAULT_TOP_K_CHUNKS

        retrieval_query = retrieval_plan.primary_query

        record_trace_call(
            deps=deps,
            tool_name="search_similar_riders_attempt",
            args={
                "attempt": 1,
                "query": retrieval_query,
                "query_component": intent.component,
                "top_k_riders": top_k_riders,
                "max_chunks_per_rider": max_chunks_per_rider,
                "top_k_chunks": top_k_chunks,
            },
            result={"ok": True},
            elapsed_ms=0.0,
        )

        riders = time_and_record(
            deps=deps,
            tool_name="search_similar_riders",
            args={
                "query": retrieval_query,
                "query_component": intent.component,
                "component_terms": intent.component_terms,
                "top_k_riders": top_k_riders,
                "max_chunks_per_rider": max_chunks_per_rider,
                "top_k_chunks": top_k_chunks,
            },
            fn=lambda: run_search_similar_riders(
                query=retrieval_query,
                query_component=intent.component,
                component_terms=intent.component_terms,
                top_k_riders=top_k_riders,
                max_chunks_per_rider=max_chunks_per_rider,
                top_k_chunks=top_k_chunks,
                deps=deps,
            ),
        )

        component_hit_count = _rider_component_hit_count(riders, intent.component_terms)

        record_trace_call(
            deps=deps,
            tool_name="component_evidence_check",
            args={
                "query_component": intent.component,
                "component_terms": intent.component_terms,
            },
            result={
                "component_hit_count": component_hit_count,
                "rider_count": len(riders or []),
            },
            elapsed_ms=0.0,
        )

        has_exact_event_scope = any(getattr(r, "_source_scope", "") == "exact_event" for r in riders or [])
        should_try_fallback = (
            bool(second_query)
            and not has_exact_event_scope
            and (
                not riders
                or len(riders) < 3
                or (intent.component != "full_setup" and component_hit_count == 0)
            )
        )

        if should_try_fallback:
            fallback_query = second_query

            record_trace_call(
                deps=deps,
                tool_name="search_similar_riders_attempt",
                args={
                    "attempt": 2,
                    "query": fallback_query,
                    "query_component": intent.component,
                    "top_k_riders": top_k_riders,
                    "max_chunks_per_rider": max_chunks_per_rider,
                    "top_k_chunks": top_k_chunks,
                },
                result={"ok": True},
                elapsed_ms=0.0,
            )

            fallback_riders = time_and_record(
                deps=deps,
                tool_name="search_similar_riders",
                args={
                    "query": fallback_query,
                    "query_component": intent.component,
                    "component_terms": intent.component_terms,
                    "top_k_riders": top_k_riders,
                    "max_chunks_per_rider": max_chunks_per_rider,
                    "top_k_chunks": top_k_chunks,
                },
                fn=lambda: run_search_similar_riders(
                    query=fallback_query,
                    query_component=intent.component,
                    component_terms=intent.component_terms,
                    top_k_riders=top_k_riders,
                    max_chunks_per_rider=max_chunks_per_rider,
                    top_k_chunks=top_k_chunks,
                    deps=deps,
                ),
            )

            fallback_component_hit_count = _rider_component_hit_count(
                fallback_riders,
                intent.component_terms,
            )

            if (
                not riders
                or len(riders) < 3
                or fallback_component_hit_count > component_hit_count
            ):
                riders = fallback_riders
                component_hit_count = fallback_component_hit_count
                retrieval_query = fallback_query

        if not riders:
            raise RuntimeError("No similar riders returned; cannot produce grounded recommendation.")

        grounding = _infer_retrieval_grounding(
            event_resolution=event_resolution,
            event_context_summary=event_context_summary,
            riders=riders,
            fallback_used=bool(second_query and retrieval_query == second_query),
        )

        retrieval_result = RetrievalExecutionResult(
            riders=riders,
            used_query=retrieval_query,
            fallback_used=bool(second_query and retrieval_query == second_query),
            fallback_reason=None,
            retrieval_source=grounding["retrieval_source"],
            exact_event_hit_count=grounding["exact_event_hit_count"],
            matched_event_name=grounding["matched_event_name"],
            component_hit_count=0,
        )

        evidence_summary = summarize_evidence(
            riders=riders,
            intent=intent,
            event_resolution=event_resolution,
            retrieval_result=retrieval_result,
        )

        policy = select_policy(
            event_match_type=event_resolution.match_type,
            matched_event_name=retrieval_result.matched_event_name,
            retrieval_source=retrieval_result.retrieval_source,
            exact_event_hit_count=retrieval_result.exact_event_hit_count,
            evidence_strength=evidence_summary.evidence_strength,
        )

        record_trace_call(
            deps=deps,
            tool_name="evidence_summary",
            args={
                "query_component": intent.component,
                "event_name": event_name,
            },
            result=evidence_summary.model_dump(),
            elapsed_ms=0.0,
        )

        record_trace_call(
            deps=deps,
            tool_name="policy_selection",
            args={
                "event_name": event_name,
                "event_match_type": event_resolution.match_type,
                "matched_event_name": retrieval_result.matched_event_name,
                "retrieval_source": retrieval_result.retrieval_source,
                "exact_event_hit_count": retrieval_result.exact_event_hit_count,
                "evidence_strength": evidence_summary.evidence_strength,
            },
            result=policy.model_dump(),
            elapsed_ms=0.0,
        )

        all_reviews = load_reviews(DEFAULT_REVIEWS_PATH)
        relevant_reviews = find_relevant_reviews(
            all_reviews,
            expected_event=event_name,
            expected_component=intent.component,
            limit=3,
        )
        review_feedback_context = format_review_context(relevant_reviews)
        record_trace_call(
            deps=deps,
            tool_name="review_feedback_lookup",
            args={
                "event_name": event_name,
                "query_component": intent.component,
                "reviews_path": str(DEFAULT_REVIEWS_PATH),
            },
            result={
                "review_count": len(all_reviews),
                "matched_review_count": len(relevant_reviews),
                "matched_run_keys": [review.run_key for review in relevant_reviews],
            },
            elapsed_ms=0.0,
        )

        compact_riders = _compact_riders(riders)

        writer_input = WriterInput(
            user_query=user_query,
            event_name=event_name,
            event_context=event_context_text[:2500],
            descriptor_query=retrieval_query,
            query_component=intent.component,
            component_hit_count=component_hit_count,
            grounding_mode=retrieval_result.retrieval_source,
            similar_riders=compact_riders,
        )

        writer_input_json = writer_input.model_dump_json(indent=2)
        writer_prompt = _build_writer_prompt(writer_input_json, review_feedback_context or None)
        writer_call_count = 0
        writer_total_ms = 0.0
        writer_first_pass_ok = False
        writer_validation_failed = False
        writer_second_pass_triggered = False
        writer_second_pass_reason: Optional[str] = None
        first_pass_error: Optional[str] = None
        first_pass_issues: List[str] = []
        final_rec: Optional[SetupRecommendation] = None

        def _finalize_writer_output(output_rec: Any) -> SetupRecommendation:
            final_rec = SetupRecommendation(
                event=getattr(output_rec, "event", None) or event_name,
                summary=getattr(output_rec, "summary", "") or "",
                reasoning=getattr(output_rec, "reasoning", "") or "",
                recommended_setup=getattr(output_rec, "recommended_setup", None) or SetupCore(),
                similar_riders=riders,
            )
            final_rec = _fill_requested_component_from_riders(
                rec=final_rec,
                riders=riders,
                query_component=intent.component,
                policy=policy,
            )
            if not final_rec.event or not final_rec.event.strip():
                final_rec.event = event_name
            return _postprocess_recommendation(final_rec)

        try:
            writer_call_count += 1
            first_rec, first_elapsed_ms = _run_writer_call(
                writer_prompt,
                deps=deps,
                stage="writer_call",
                args={
                    "pass": 1,
                    "query_component": intent.component,
                    "component_hit_count": component_hit_count,
                    "descriptor_query": retrieval_query,
                },
            )
            writer_total_ms += first_elapsed_ms
            final_rec = _finalize_writer_output(first_rec)
            first_pass_issues = _writer_validation_issues(final_rec)
            writer_first_pass_ok = not first_pass_issues
            writer_validation_failed = not writer_first_pass_ok
        except Exception as exc:
            writer_validation_failed = True
            first_pass_error = f"{type(exc).__name__}: {exc}"
            first_pass_issues = [first_pass_error]
            logger.warning("First writer pass failed for query=%r: %s", user_query, first_pass_error)

        if writer_validation_failed:
            writer_second_pass_triggered = True
            writer_second_pass_reason = (
                first_pass_error
                or ("validation_failed:" + ",".join(first_pass_issues) if first_pass_issues else "validation_failed")
            )

            record_trace_call(
                deps=deps,
                tool_name="writer_repair_triggered",
                args={
                    "query_component": intent.component,
                    "component_hit_count": component_hit_count,
                    "descriptor_query": retrieval_query,
                },
                result={
                    "reason": writer_second_pass_reason,
                    "issues": first_pass_issues,
                },
                elapsed_ms=0.0,
            )

            repair_prompt = _build_writer_repair_prompt(
                writer_input_json=writer_input_json,
                first_draft=final_rec,
                issues=first_pass_issues,
                error_text=first_pass_error,
                review_context=review_feedback_context or None,
            )

            try:
                writer_call_count += 1
                repaired_rec, repair_elapsed_ms = _run_writer_call(
                    repair_prompt,
                    deps=deps,
                    stage="writer_repair_call",
                    args={
                        "pass": 2,
                        "reason": writer_second_pass_reason,
                        "query_component": intent.component,
                        "component_hit_count": component_hit_count,
                        "descriptor_query": retrieval_query,
                    },
                )
                writer_total_ms += repair_elapsed_ms
                final_rec = _finalize_writer_output(repaired_rec)
                repair_issues = _writer_validation_issues(final_rec)
                if repair_issues:
                    writer_validation_failed = True
                    writer_second_pass_reason = "repair_validation_failed:" + ",".join(repair_issues)
                else:
                    writer_validation_failed = False
            except Exception as exc:
                writer_validation_failed = True
                writer_second_pass_reason = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Writer repair pass failed for query=%r: %s",
                    user_query,
                    writer_second_pass_reason,
                )
                if final_rec is None:
                    raise

        if final_rec is None:
            raise RuntimeError("Writer failed to produce a recommendation.")

        record_trace_call(
            deps=deps,
            tool_name="writer_validation",
            args={
                "query_component": intent.component,
                "component_hit_count": component_hit_count,
            },
            result={
                "writer_call_count": writer_call_count,
                "writer_first_pass_ok": writer_first_pass_ok,
                "writer_validation_failed": writer_validation_failed,
                "writer_second_pass_triggered": writer_second_pass_triggered,
                "writer_second_pass_reason": writer_second_pass_reason,
                "missing_fields": _writer_missing_fields(final_rec),
                "issues": first_pass_issues,
            },
            elapsed_ms=0.0,
        )

        record_trace_call(
            deps=deps,
            tool_name="writer_stage_summary",
            args={
                "query_component": intent.component,
                "component_hit_count": component_hit_count,
                "descriptor_query": retrieval_query,
            },
            result={
                "writer_call_count": writer_call_count,
                "writer_first_pass_ok": writer_first_pass_ok,
                "writer_validation_failed": writer_validation_failed,
                "writer_second_pass_triggered": writer_second_pass_triggered,
                "writer_second_pass_reason": writer_second_pass_reason,
                "writer_total_ms": round(writer_total_ms, 3),
            },
            elapsed_ms=writer_total_ms,
        )

        return final_rec, trace

        
        
def recommend_setup(user_query: str) -> SetupRecommendation:
    rec, _trace = recommend_setup_with_trace(user_query)
    return rec
