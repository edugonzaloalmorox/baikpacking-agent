from baikpacking.tools.event_context import EventContextSummary


def test_event_context_summary_accepts_null_constraints():
    summary = EventContextSummary.model_validate(
        {
            "summary": "Remote ultra",
            "surface": "gravel",
            "route_character": "mixed terrain",
            "climate_notes": "dry",
            "resupply_notes": "sparse",
            "constraints": None,
        }
    )

    assert summary.constraints == []
