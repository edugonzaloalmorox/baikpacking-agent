from types import SimpleNamespace

from baikpacking.scripts.run_eval_scenarios import (
    evaluate_content_assertions,
    evaluate_event_alignment_assertions,
    extract_run_row,
)


def _run(**kwargs):
    base = {
        "expected_intent": "full_setup",
        "intent_component": None,
        "recommended_setup": {
            "bike_type": "gravel bike",
            "tyres": "45mm semi-slick tyres",
            "bags": "frame bag",
            "drivetrain": "GRX 1x",
        },
    }
    base.update(kwargs)
    return base


def _event_run(**kwargs):
    base = {
        "expected_event": "Atlas Mountain Race",
        "expected_intent": "full_setup",
        "matched_event_name": "Atlas Mountain Race",
        "retrieval_source": "exact_event",
        "exact_event_hit_count": 3,
        "event_match_type": "exact",
    }
    base.update(kwargs)
    return base


def _row_with_policy(
    *,
    expected_event="Atlas Mountain Race",
    scenario_id="alias_amr",
    group="alias_variation",
    expected_intent="component_bike_type",
    summary="",
    reasoning="",
    retrieval_source="similar_event",
    exact_event_hit_count=0,
    matched_event_name="Atlas Mountain Race",
    event_match_type="unknown",
):
    scenario = {
        "id": scenario_id,
        "group": group,
        "query": "What bike for AMR?",
        "why": "alias-style scenario",
        "expected_event": expected_event,
        "expected_intent": expected_intent,
        "expected_behavior": "Correctly maps AMR to Atlas and answers bike type",
        "notes": "",
    }
    recommendation = SimpleNamespace(
        event="Atlas Mountain Race",
        summary=summary,
        reasoning=reasoning,
        recommended_setup={
            "bike_type": "gravel",
            "tyres": None,
            "bags": None,
            "drivetrain": None,
        },
        similar_riders=[],
    )
    trace = SimpleNamespace(
        calls=[
            {
                "tool": "policy_selection",
                "args": {
                    "event_match_type": event_match_type,
                    "matched_event_name": matched_event_name,
                    "retrieval_source": retrieval_source,
                    "exact_event_hit_count": exact_event_hit_count,
                },
                "result": {"mode": "pattern_based"},
            }
        ]
    )
    return extract_run_row(scenario, recommendation, trace=trace, status="success", error=None)


def test_valid_full_setup_passes():
    result = evaluate_content_assertions(_run())

    assert result["content_assertions_passed"] is True
    assert result["content_assertion_issues"] == []
    assert result["content_assertion_issue_count"] == 0


def test_missing_bike_type_is_flagged():
    result = evaluate_content_assertions(_run(recommended_setup={"tyres": "45mm", "bags": "frame bag"}))

    assert result["content_assertions_passed"] is False
    assert "missing_bike_type" in result["content_assertion_issues"]


def test_drivetrain_with_storage_terms_is_flagged():
    result = evaluate_content_assertions(_run(recommended_setup={"bike_type": "gravel", "tyres": "45mm", "bags": "frame bag", "drivetrain": "GRX drivetrain with frame bag"}))

    assert "drivetrain_contains_storage_terms" in result["content_assertion_issues"]


def test_bags_with_gearing_is_flagged():
    result = evaluate_content_assertions(_run(recommended_setup={"bike_type": "gravel", "tyres": "45mm", "bags": "2x gearing and top tube bag", "drivetrain": "GRX 1x"}))

    assert "bags_contains_drivetrain_terms" in result["content_assertion_issues"]


def test_tyres_with_storage_is_flagged():
    result = evaluate_content_assertions(_run(recommended_setup={"bike_type": "gravel", "tyres": "frame bag with 45mm tyres", "bags": "frame bag", "drivetrain": "GRX 1x"}))

    assert "tyres_contains_storage_terms" in result["content_assertion_issues"]


def test_tyres_without_tyre_signals_is_flagged():
    result = evaluate_content_assertions(_run(recommended_setup={"bike_type": "gravel", "tyres": "light cargo mount", "bags": "frame bag", "drivetrain": "GRX 1x"}))

    assert "tyres_missing_tyre_signals" in result["content_assertion_issues"]


def test_drivetrain_without_drivetrain_signals_is_flagged():
    result = evaluate_content_assertions(
        _run(recommended_setup={"bike_type": "gravel", "tyres": "45mm", "bags": "frame bag", "drivetrain": "stable ride feel"})
    )

    assert "drivetrain_missing_drivetrain_signals" in result["content_assertion_issues"]


def test_lighting_with_wahoo_elemnt_is_flagged():
    result = evaluate_content_assertions(
        _run(
            recommended_setup={
                "bike_type": "gravel",
                "tyres": "45mm",
                "bags": "frame bag",
                "drivetrain": "GRX 1x",
                "lighting": "Wahoo ELEMNT",
            }
        )
    )

    assert "lighting_contains_navigation_terms" in result["content_assertion_issues"]
    assert "lighting_missing_lighting_signals" in result["content_assertion_issues"]


def test_navigation_with_lezyne_front_light_is_flagged():
    result = evaluate_content_assertions(
        _run(
            recommended_setup={
                "bike_type": "gravel",
                "tyres": "45mm",
                "bags": "frame bag",
                "drivetrain": "GRX 1x",
                "navigation": "Lezyne front light",
            }
        )
    )

    assert "navigation_contains_lighting_terms" in result["content_assertion_issues"]
    assert "navigation_missing_navigation_signals" in result["content_assertion_issues"]


def test_drivetrain_with_nav_and_lighting_text_is_flagged():
    result = evaluate_content_assertions(
        _run(
            recommended_setup={
                "bike_type": "gravel",
                "tyres": "45mm",
                "bags": "frame bag",
                "drivetrain": "Garmin nav and Lezyne lights",
            }
        )
    )

    assert "drivetrain_contains_navigation_terms" in result["content_assertion_issues"]
    assert "drivetrain_contains_lighting_terms" in result["content_assertion_issues"]
    assert "drivetrain_missing_drivetrain_signals" in result["content_assertion_issues"]


def test_non_full_setup_skips_checks():
    result = evaluate_content_assertions(
        {
            "expected_intent": "tyres",
            "intent_component": "tyres",
            "recommended_setup": {
                "bike_type": None,
                "tyres": None,
                "bags": None,
                "drivetrain": None,
            },
        }
    )

    assert result["content_assertions_passed"] is True
    assert result["content_assertion_issues"] == []
    assert result["content_assertion_issue_count"] == 0


def test_exact_known_event_with_zero_hits_is_flagged():
    result = evaluate_event_alignment_assertions(
        _event_run(exact_event_hit_count=0, retrieval_source="similar_event")
    )

    assert result["event_alignment_assertions_passed"] is False
    assert "expected_exact_event_but_no_exact_hits" in result["event_alignment_issues"]
    assert "expected_exact_event_but_used_similar_event" in result["event_alignment_issues"]


def test_event_match_type_exact_with_zero_hits_is_flagged():
    result = evaluate_event_alignment_assertions(
        _event_run(exact_event_hit_count=0, retrieval_source="exact_event", event_match_type="exact")
    )

    assert "event_match_type_exact_but_zero_exact_hits" in result["event_alignment_issues"]


def test_unknown_generic_scenario_skips_strict_event_checks():
    result = evaluate_event_alignment_assertions(
        {
            "expected_event": "Unknown event",
            "expected_intent": "full_setup",
            "matched_event_name": "Some race",
            "retrieval_source": "similar_event",
            "exact_event_hit_count": 0,
            "event_match_type": "exact",
        }
    )

    assert result["event_alignment_assertions_passed"] is True
    assert result["event_alignment_issues"] == []
    assert result["event_alignment_issue_count"] == 0


def test_alias_fallback_allowed_passes_when_honest():
    row = _row_with_policy(
        summary="Based on similar events, a gravel bike with 45mm tyres makes sense.",
        reasoning="Similar ultra-endurance events suggest this is a safe starting point.",
    )

    assert row["event_grounding_mode"] == "similar_event_fallback_allowed"
    assert row["event_alignment_assertions_passed"] is True
    assert row["event_alignment_issues"] == []
    assert row["knowledge_base_exact_match"] is False


def test_alias_fallback_allowed_flags_overclaiming_exact_grounding():
    row = _row_with_policy(
        summary="For the Atlas Mountain Race, riders often use a gravel bike.",
        reasoning="Riders in Atlas Mountain Race favor this setup.",
    )

    assert row["event_alignment_assertions_passed"] is False
    assert "similar_event_fallback_not_disclosed" in row["event_alignment_issues"]
    assert "exact_grounding_claim_with_similar_event_retrieval" in row["event_alignment_issues"]


def test_exact_grounding_scenario_still_fails_when_exact_hits_missing():
    row = extract_run_row(
        {
            "id": "exact_atlas_full_setup",
            "group": "exact_known_event",
            "query": "Suggest a bikepacking setup for Atlas Mountain Race",
            "why": "Tests exact event grounding and full setup generation",
            "expected_event": "Atlas Mountain Race",
            "expected_intent": "full_setup",
            "expected_behavior": "Uses Atlas riders only, no fallback, grounded recommendations",
            "notes": "",
        },
        SimpleNamespace(
            event="Atlas Mountain Race",
            summary="summary",
            reasoning="reasoning",
            recommended_setup={
                "bike_type": "gravel bike",
                "tyres": "45mm",
                "bags": "frame bag",
                "drivetrain": "GRX 1x",
            },
            similar_riders=[],
        ),
        trace=SimpleNamespace(
            calls=[
                {
                    "tool": "policy_selection",
                    "args": {
                        "event_match_type": "exact",
                        "matched_event_name": "Atlas Mountain Race",
                        "retrieval_source": "similar_event",
                        "exact_event_hit_count": 0,
                    },
                    "result": {"mode": "strict_grounded"},
                }
            ]
        ),
        status="success",
        error=None,
    )

    assert row["event_alignment_assertions_passed"] is False
    assert "expected_exact_event_but_no_exact_hits" in row["event_alignment_issues"]
    assert "expected_exact_event_but_used_similar_event" in row["event_alignment_issues"]


def test_extract_run_row_merges_content_assertions():
    scenario = {
        "id": "s1",
        "group": "full",
        "query": "Recommend a full setup for event X",
        "why": "check content rules",
        "expected_event": "Event X",
        "expected_intent": "full_setup",
        "expected_behavior": "grounded full setup",
        "notes": "",
    }
    recommendation = SimpleNamespace(
        event="Event X",
        summary="summary",
        reasoning="reasoning",
        recommended_setup={
            "bike_type": "gravel",
            "tyres": "45mm",
            "bags": "frame bag",
            "drivetrain": "GRX 1x",
        },
        similar_riders=[],
    )
    trace = SimpleNamespace(
        calls=[
            {
                "tool": "policy_selection",
                "args": {
                    "event_match_type": "exact",
                    "matched_event_name": "Event X",
                    "retrieval_source": "exact_event",
                    "exact_event_hit_count": 2,
                },
                "result": {"mode": "strict_grounded"},
            }
        ]
    )

    row = extract_run_row(scenario, recommendation, trace=trace, status="success", error=None)

    assert row["content_assertions_passed"] is True
    assert row["content_assertion_issues"] == []
    assert row["event_alignment_assertions_passed"] is True
    assert row["event_alignment_issues"] == []
