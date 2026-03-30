def test_embed_index_imports_cleanly():
    import baikpacking.pipelines.embed_index as embed_index
    from baikpacking.db import data_loader

    assert callable(embed_index.run_embed_riders)
    assert callable(data_loader.fetch_riders)
    assert callable(data_loader.fetch_riders_missing_embeddings)
    assert callable(data_loader.truncate_rider_embeddings)
    assert callable(data_loader.upsert_rider_embeddings)
