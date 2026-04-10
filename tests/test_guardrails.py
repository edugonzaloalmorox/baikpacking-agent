from __future__ import annotations

from types import SimpleNamespace

import pytest

from baikpacking.agents.guardrails import RecommendationGuardBlocked, should_recommend
from baikpacking.agents.models import QueryIntent
from baikpacking.agents.orchestration_models import EventResolutionResult
from baikpacking.agents.recommender_agent import recommend_setup_with_trace
from baikpacking.tools.call_trace import CallTrace


def test_should_recommend_blocks_irrelevant_query():
    event_resolution = EventResolutionResult(
        display_name="Unknown event",
        match_type="unknown",
        confidence=0.0,
        is_trusted_exact=False,
    )
    intent = QueryIntent(component="full_setup", confidence=0.25, component_terms=[], asks_for_recommendation=True)

    decision = should_recommend("What fee for a burger king", event_resolution, intent)

    assert decision.allow_recommendation is False
    assert decision.guard_type == "out_of_domain"


def test_should_recommend_blocks_weak_event_and_low_intent():
    event_resolution = EventResolutionResult(
        display_name="Atlas Mountain Race",
        canonical_name="Atlas Mountain Race",
        match_type="weak_candidate",
        confidence=0.4,
        is_trusted_exact=False,
    )
    intent = QueryIntent(component="full_setup", confidence=0.2, component_terms=[], asks_for_recommendation=False)

    decision = should_recommend("Atlas Mountain Race", event_resolution, intent)

    assert decision.allow_recommendation is False
    assert decision.guard_type == "combined_safety"


def test_should_recommend_allows_strong_relevant_query():
    event_resolution = EventResolutionResult(
        display_name="Atlas Mountain Race",
        canonical_name="Atlas Mountain Race",
        match_type="exact",
        confidence=0.98,
        is_trusted_exact=True,
    )
    intent = QueryIntent(component="tyres", confidence=0.9, component_terms=["tyre", "tyres"], asks_for_recommendation=True)

    decision = should_recommend("What tyres should I use for Atlas Mountain Race?", event_resolution, intent)

    assert decision.allow_recommendation is True
    assert decision.guard_type == "allow"
    assert decision.allow_exact_grounding is True


def test_should_recommend_allows_component_query_without_exact_grounding():
    event_resolution = EventResolutionResult(
        display_name="Atlas Mountain Race",
        canonical_name="Atlas Mountain Race",
        match_type="weak_candidate",
        confidence=0.55,
        is_trusted_exact=False,
    )
    intent = QueryIntent(component="tyres", confidence=0.86, component_terms=["tyre", "tyres"], asks_for_recommendation=True)

    decision = should_recommend("What tyres should I use?", event_resolution, intent)

    assert decision.allow_recommendation is True
    assert decision.guard_type == "weak_event"
    assert decision.allow_exact_grounding is False


def test_recommend_setup_blocks_before_retrieval(monkeypatch: pytest.MonkeyPatch):
    from baikpacking.agents import recommender_agent as mod

    def boom(*args, **kwargs):
        raise AssertionError("retrieval should not run for blocked queries")

    monkeypatch.setattr(mod, "_build_deps", lambda *args, **kwargs: SimpleNamespace(call_trace=CallTrace()))
    monkeypatch.setattr(
        mod,
        "resolve_event",
        lambda query: EventResolutionResult(
            display_name="Unknown event",
            match_type="unknown",
            confidence=0.0,
            is_trusted_exact=False,
        ),
    )
    monkeypatch.setattr(
        mod,
        "_classify_query_intent",
        lambda query: QueryIntent(component="full_setup", confidence=0.25, component_terms=[], asks_for_recommendation=True),
    )
    monkeypatch.setattr(mod, "fetch_event_context_summary", boom)

    with pytest.raises(RecommendationGuardBlocked):
        recommend_setup_with_trace("What fee for a burger king")
