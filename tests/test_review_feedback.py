from pathlib import Path

from baikpacking.agents.review_feedback import (
    ScenarioReview,
    build_run_key,
    find_relevant_reviews,
    format_review_context,
    load_reviews,
    save_review,
)


def test_review_feedback_roundtrip_and_matching(tmp_path: Path):
    review_path = tmp_path / "scenario_reviews.jsonl"
    run = {
        "scenario_id": "exact_atlas_full_setup",
        "timestamp": "2026-04-07T18:21:12.252546+00:00",
        "expected_event": "Atlas Mountain Race",
        "expected_component": "full_setup",
        "expected_policy_mode": "strict_grounded",
    }

    review = ScenarioReview(
        run_key=build_run_key(run),
        scenario_id="exact_atlas_full_setup",
        run_timestamp=run["timestamp"],
        expected_event="Atlas Mountain Race",
        expected_component="full_setup",
        expected_policy_mode="strict_grounded",
        review_status="approved",
        human_label="good",
        corrected_event="Atlas Mountain Race",
        corrected_component="full_setup",
        corrected_policy_mode="strict_grounded",
        review_notes="Good exact grounding.",
    )
    save_review(review, review_path)

    loaded = load_reviews(review_path)
    assert len(loaded) == 1
    assert loaded[0].run_key == review.run_key

    matched = find_relevant_reviews(
        loaded,
        expected_event="Atlas Mountain Race",
        expected_component="full_setup",
    )
    assert matched and matched[0].run_key == review.run_key

    context = format_review_context(matched)
    assert "Human review hints" not in context
    assert "corrected_policy_mode=strict_grounded" in context
