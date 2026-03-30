import os
import re
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
from baikpacking.agents.models import SetupRecommendation
from baikpacking.agents.output_validation import _fill_requested_component_from_riders
from baikpacking.agents.postprocess import (
    _infer_event_from_riders as _postprocess_infer_event_from_riders,
    _infer_year_from_title,
    _postprocess_recommendation as _postprocess_recommendation_impl,
)
from baikpacking.agents.policy import select_policy
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
    exact_event_hit_count = 0

    for rider in riders or []:
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

    if exact_event_hit_count > 0:
        retrieval_source = "exact_event"
    elif fallback_used or getattr(event_context_summary, "similar_events", None) or getattr(event_context_summary, "event_family", None):
        retrieval_source = "similar_event"
    else:
        retrieval_source = "unknown_global"

    matched_event_name = title_counts.most_common(1)[0][0] if title_counts else None

    return {
        "retrieval_source": retrieval_source,
        "exact_event_hit_count": exact_event_hit_count,
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


WRITER_PROMPT = """
You are a bikepacking equipment and ultra-distance cycling expert.

Return only valid JSON matching SetupRecommendation.

Grounding rules:
- ALL gear details must come from similar_riders.
- event_context is only for understanding the target event characteristics.
- query_component tells you what part of the setup the user is asking about.
- component_hit_count tells you how many retrieved riders explicitly mention the requested component.
- Do not invent gear, brands, or specs.
- If a field cannot be grounded, leave it empty/null and explain that in reasoning.
- If component_hit_count is 0 for a component-specific question, say evidence is sparse and avoid specific grounded claims.
- Use similar_riders exactly as provided in the input for the output similar_riders content.
- Prefer concise output:
  - summary: 3-4 sentences
  - reasoning: 3-5 sentences

Output rules:
- event must be the requested event_name
- only mention similar events if the exact event is absent from the retrieved riders
- recommended_setup should contain as many grounded fields as possible without guessing
""".strip()

writer_model = OpenAIChatModel(settings.writer_model)

writer_agent = Agent(
    model=writer_model,
    output_type=SetupRecommendation,
    system_prompt=WRITER_PROMPT,
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

        should_try_fallback = (
            bool(second_query)
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

        compact_riders = _compact_riders(riders)

        writer_input = WriterInput(
            user_query=user_query,
            event_name=event_name,
            event_context=event_context_text[:2500],
            descriptor_query=retrieval_query,
            query_component=intent.component,
            component_hit_count=component_hit_count,
            similar_riders=compact_riders,
        )

        rec = writer_agent.run_sync(writer_input.model_dump_json(indent=2)).output
        rec.similar_riders = riders

        rec = _fill_requested_component_from_riders(
            rec=rec,
            riders=riders,
            query_component=intent.component,
            policy=policy,
        )

        if not rec.event or not rec.event.strip():
            rec.event = event_name

        return _postprocess_recommendation(rec), trace

        
        
def recommend_setup(user_query: str) -> SetupRecommendation:
    rec, _trace = recommend_setup_with_trace(user_query)
    return rec
