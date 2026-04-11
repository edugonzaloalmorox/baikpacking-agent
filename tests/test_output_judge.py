from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from baikpacking.agents.live_feedback import build_live_feedback_record, load_live_feedback
from baikpacking.eval.output_judge import (
    OutputJudgeResult,
    apply_judge_rules,
    build_compact_judge_input,
    build_output_judge_prompt,
    join_runs_with_feedback,
    load_jsonl_records,
    run_output_judge_batch,
    run_output_judge_for_run,
    select_runs_for_judging,
    summarize_output_judgments,
)


def _sample_run(run_id: str = "run-1") -> dict[str, object]:
    return {
        "run_id": run_id,
        "timestamp": "2026-04-10T12:00:00+00:00",
        "query": "What tyres do you recommend for Atlas Mountain Race?",
        "status": "success",
        "resolved_event_name": "Atlas Mountain Race",
        "event_match_type": "exact",
        "retrieval_source": "exact_event",
        "retrieval_mode": "exact_only",
        "policy_mode": "strict_grounded",
        "query_component": "tyres",
        "recommended_setup": {
            "bike_type": "gravel bike",
            "wheels": "700c",
            "tyres": "45mm semi-slick",
        },
        "summary": "Use a grounded tyre setup.",
        "reasoning": "Exact event evidence supports this.",
        "rider_count": 3,
        "component_hit_count": 2,
        "evidence_strength": "strong",
        "evidence_consistency": "mostly_consistent",
        "missing_fields": [],
        "component_relevance": {"passed": True},
        "retrieval_policy_issues": [],
    }


def test_load_jsonl_records_skips_malformed_lines(tmp_path: Path):
    path = tmp_path / "live_runs.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"run_id": "a", "query": "one"}),
                "{not json}",
                json.dumps({"run_id": "b", "query": "two"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_jsonl_records(path)
    assert [row["run_id"] for row in rows] == ["a", "b"]


def test_join_runs_with_feedback_uses_latest_feedback(tmp_path: Path):
    feedback_path = tmp_path / "live_feedback.jsonl"
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    feedback_path.write_text(
        "\n".join(
            [
                build_live_feedback_record(
                    run_id="run-1",
                    feedback="thumbs_up",
                    comment="first",
                    timestamp="2026-04-10T12:00:00+00:00",
                ).model_dump_json(),
                build_live_feedback_record(
                    run_id="run-1",
                    feedback="thumbs_down",
                    comment="latest",
                    timestamp="2026-04-10T12:05:00+00:00",
                ).model_dump_json(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    feedback_records = load_live_feedback(feedback_path)
    joined = join_runs_with_feedback([_sample_run()], feedback_records)

    assert joined[0]["feedback"] == "thumbs_down"
    assert joined[0]["feedback_comment"] == "latest"


def test_build_compact_judge_input_renders_final_answer():
    payload = build_compact_judge_input(_sample_run())

    assert payload.query_component == "tyres"
    assert payload.final_answer_text.startswith("Use a grounded tyre setup.")
    assert "Bike type: gravel bike" in payload.final_answer_text
    assert "Tyres: 45mm semi-slick" in payload.final_answer_text


def test_apply_judge_rules_derives_verdict_and_failure_type():
    weak = apply_judge_rules(
        OutputJudgeResult(
            relevance_score=2,
            grounding_honesty_score=2,
            usefulness_score=0,
            missing_requested_component=False,
            hallucination_or_overclaim=False,
            verdict="pass",
            failure_type="none",
            reason="Useful but too vague.",
        )
    )
    assert weak.verdict == "weak"
    assert weak.failure_type == "usefulness"

    grounding = apply_judge_rules(
        OutputJudgeResult(
            relevance_score=2,
            grounding_honesty_score=0,
            usefulness_score=2,
            missing_requested_component=False,
            hallucination_or_overclaim=False,
            verdict="pass",
            failure_type="none",
            reason="Unsupported.",
        )
    )
    assert grounding.verdict == "fail"
    assert grounding.failure_type == "grounding"


def test_select_runs_for_judging_prioritizes_feedback_and_fallbacks(tmp_path: Path):
    feedback_path = tmp_path / "live_feedback.jsonl"
    feedback_path.write_text(
        build_live_feedback_record(
            run_id="run-2",
            feedback="thumbs_down",
            comment="bad",
            timestamp="2026-04-10T12:05:00+00:00",
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    runs = [
        _sample_run("run-1"),
        {**_sample_run("run-2"), "retrieval_source": "similar_event"},
        {**_sample_run("run-3"), "retrieval_source": "similar_event"},
    ]
    selected = select_runs_for_judging(runs, load_live_feedback(feedback_path), max_runs=10, sample_rate=0.0)
    assert [row["run_id"] for row in selected] == ["run-2", "run-3"]
    assert selected[0]["selection_reason"].startswith("negative_feedback")
    assert "fallback_like" in selected[1]["selection_reason"]


def test_run_output_judge_for_run_uses_mocked_agent_and_strict_prompt(monkeypatch):
    import baikpacking.eval.output_judge as mod

    captured = {"prompt": None}

    async def fake_run(prompt):
        captured["prompt"] = prompt
        return SimpleNamespace(
            output={
                "relevance_score": 2,
                "grounding_honesty_score": 2,
                "usefulness_score": 2,
                "missing_requested_component": False,
                "hallucination_or_overclaim": False,
                "verdict": "weak",
                "failure_type": "usefulness",
                "reason": "Good answer.",
            }
        )

    monkeypatch.setattr(mod.judge_agent, "run", fake_run)

    result = run_output_judge_for_run(_sample_run())
    assert result.verdict == "pass"
    assert result.failure_type == "none"
    assert captured["prompt"] is not None
    assert "Return strict JSON only" in captured["prompt"]


def test_existing_judgment_skips_duplicate_run(tmp_path: Path, monkeypatch):
    import baikpacking.eval.output_judge as mod

    runs_path = tmp_path / "live_runs.jsonl"
    output_path = tmp_path / "output_judgments.jsonl"
    runs_path.write_text(json.dumps(_sample_run()) + "\n", encoding="utf-8")
    output_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "judge_timestamp": "2026-04-10T12:30:00+00:00",
                "query": "What tyres do you recommend for Atlas Mountain Race?",
                "resolved_event_name": "Atlas Mountain Race",
                "event_match_type": "exact",
                "retrieval_source": "exact_event",
                "policy_mode": "strict_grounded",
                "query_component": "tyres",
                "feedback": "",
                "feedback_comment": "",
                "model_name": "gpt-4o-mini",
                "selection_reason": "fallback_like",
                "relevance_score": 2,
                "grounding_honesty_score": 2,
                "usefulness_score": 2,
                "missing_requested_component": False,
                "hallucination_or_overclaim": False,
                "verdict": "pass",
                "failure_type": "none",
                "reason": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    async def fake_run(prompt):
        raise AssertionError("judge should not run for an existing run_id")

    monkeypatch.setattr(mod.judge_agent, "run", fake_run)

    written = run_output_judge_batch(
        runs_path=runs_path,
        feedback_path=tmp_path / "live_feedback.jsonl",
        output_path=output_path,
        max_runs=10,
        sample_rate=0.0,
        force=False,
    )

    assert written == []
    assert len(load_jsonl_records(output_path)) == 1


def test_summarize_output_judgments(tmp_path: Path):
    path = tmp_path / "output_judgments.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": "run-1",
                        "judge_timestamp": "2026-04-10T12:30:00+00:00",
                        "query": "q1",
                        "resolved_event_name": "Atlas Mountain Race",
                        "event_match_type": "exact",
                        "retrieval_source": "exact_event",
                        "policy_mode": "strict_grounded",
                        "query_component": "tyres",
                        "feedback": "thumbs_up",
                        "feedback_comment": "",
                        "model_name": "gpt-4o-mini",
                        "selection_reason": "fallback_like",
                        "relevance_score": 2,
                        "grounding_honesty_score": 2,
                        "usefulness_score": 2,
                        "missing_requested_component": False,
                        "hallucination_or_overclaim": False,
                        "verdict": "pass",
                        "failure_type": "none",
                        "reason": "ok",
                    }
                ),
                json.dumps(
                    {
                        "run_id": "run-2",
                        "judge_timestamp": "2026-04-10T12:31:00+00:00",
                        "query": "q2",
                        "resolved_event_name": "Atlas Mountain Race",
                        "event_match_type": "exact",
                        "retrieval_source": "similar_event",
                        "policy_mode": "generic_fallback",
                        "query_component": "drivetrain",
                        "feedback": "thumbs_down",
                        "feedback_comment": "bad",
                        "model_name": "gpt-4o-mini",
                        "selection_reason": "negative_feedback",
                        "relevance_score": 1,
                        "grounding_honesty_score": 1,
                        "usefulness_score": 0,
                        "missing_requested_component": True,
                        "hallucination_or_overclaim": False,
                        "verdict": "weak",
                        "failure_type": "missing_component",
                        "reason": "partial",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_output_judgments(path)
    assert summary["total_judged"] == 2
    assert summary["pass_count"] == 1
    assert summary["weak_count"] == 1
    assert summary["fail_count"] == 0
    assert summary["failure_type_distribution"]["none"] == 1
    assert summary["failure_type_distribution"]["missing_component"] == 1
