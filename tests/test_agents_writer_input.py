from types import SimpleNamespace

from baikpacking.agents.writer_input import _compact_riders, _event_context_to_text


def test_compact_riders_limits_chunks_and_key_items_and_infers_year():
    rider = SimpleNamespace(
        name="Alice",
        event_title="GranGuanche 2025",
        year=None,
        best_score=0.91,
        bike_type="Gravel bike",
        wheels="700c",
        tyres="45mm",
        drivetrain="1x",
        bags="Frame bag",
        sleep_system="Bivy",
        key_items=["  Dynamo  ", "GPS", "", "Quilt", "Stove", "Extra item"],
        chunks=[
            SimpleNamespace(text="Chunk 1", chunk_index=None),
            SimpleNamespace(content="Chunk 2"),
            SimpleNamespace(text="Chunk 3"),
        ],
    )

    compact = _compact_riders([rider])

    assert len(compact) == 1
    assert compact[0].year == 2025
    assert compact[0].key_items == ["Dynamo", "GPS", "Quilt", "Stove"]
    assert len(compact[0].chunks) == 2
    assert compact[0].chunks[0].chunk_index is None
    assert compact[0].chunks[1].chunk_index == 1


def test_event_context_to_text_joins_non_empty_fields():
    event_context_obj = SimpleNamespace(
        context=SimpleNamespace(
            summary="Remote ultra",
            surface="Mostly paved",
            route_character="Long crossings",
            climate_notes="Windy",
            resupply_notes="Sparse at night",
            constraints=["carry lights", "self-supported"],
        )
    )

    text = _event_context_to_text(event_context_obj)

    assert "Remote ultra" in text
    assert "Mostly paved" in text
    assert "carry lights self-supported" in text
