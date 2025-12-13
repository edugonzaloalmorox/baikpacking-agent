import pytest
from baikpacking.agents.recommender_agent import (
    recommender_agent,
    _postprocess_recommendation,
)


@pytest.mark.asyncio
async def test_recommender_basic():
    query = "I'm doing GranGuanche Audax Trail 2025, prefer 45mm tyres."
    result = await recommender_agent.run(query)
    rec = _postprocess_recommendation(result.output)

    # Basic sanity check on the main output
    assert isinstance(rec.summary, str)
    assert len(rec.summary) > 20
    assert isinstance(rec.event, str)
    assert "Gran" in rec.event or "Guanche" in rec.event

    # Similar riders: don't fail the test if empty, just assert the type
    assert isinstance(rec.similar_riders, list)

    # Optional: if you want visibility during development
    if not rec.similar_riders:
        # This will show in `-s` mode
        print("WARNING: similar_riders is empty in this run")