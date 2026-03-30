from baikpacking.agents.models import ChunkInfo, SetupRecommendation, SimilarRider
from baikpacking.agents.postprocess import (
    _infer_event_from_riders,
    _infer_year_from_title,
    _postprocess_recommendation,
)


def test_infer_year_from_title():
    assert _infer_year_from_title("GranGuanche 2025") == 2025
    assert _infer_year_from_title("No year here") is None


def test_infer_event_from_riders_picks_most_common_title():
    rec = SetupRecommendation(
        summary="x",
        reasoning="y",
        similar_riders=[
            SimilarRider(rider_id=1, best_score=0.8, event_title="Tour Divide"),
            SimilarRider(rider_id=2, best_score=0.7, event_title="Tour Divide"),
            SimilarRider(rider_id=3, best_score=0.9, event_title="Atlas Mountain Race"),
        ],
    )

    assert _infer_event_from_riders(rec) == "Tour Divide"


def test_postprocess_recommendation_infers_event_years_and_sorts():
    rec = SetupRecommendation(
        summary="x",
        reasoning="y",
        similar_riders=[
            SimilarRider(
                rider_id=1,
                best_score=0.5,
                event_title="Other Event 2023",
                chunks=[ChunkInfo(score=0.1, text="a", chunk_index=None)],
            ),
            SimilarRider(
                rider_id=2,
                best_score=0.9,
                event_title="Target Event 2024",
                chunks=[ChunkInfo(score=0.1, text="b", chunk_index=None)],
            ),
        ],
    )

    rec.event = "Target Event"
    out = _postprocess_recommendation(rec)

    assert out.similar_riders[0].rider_id == 2
    assert out.similar_riders[0].year == 2024
    assert out.similar_riders[0].chunks[0].chunk_index == 0
