from baikpacking.tools.riders import _synthesize_exact_event_chunk_rank


def test_synthesize_exact_event_chunk_rank_marks_exact_event_scope():
    rank = _synthesize_exact_event_chunk_rank([10, 11, 12], top_k_riders=2)

    assert list(rank) == [10, 11]
    assert rank[10]["source_scope"] == "exact_event"
    assert rank[10]["weighted_best_score"] == 1.0
    assert rank[10]["chunks"] == []
