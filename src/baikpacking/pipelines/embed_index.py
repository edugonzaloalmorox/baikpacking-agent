from baikpacking.embedding.embed import embed_riders_from_json
from baikpacking.embedding.qdrant_utils import upsert_chunks_to_qdrant


def run_embed_and_index(
    json_path: str = "data/dotwatcher_bikes_cleaned.json",
    max_chars: int = 800,
    overlap: int = 100,
) -> None:
    """
    Pipeline step:
      1) read cleaned riders JSON
      2) create embeddings (with chunking)
      3) upsert into Qdrant
    """
    chunks = embed_riders_from_json(
        json_path=json_path,
        max_chars=max_chars,
        overlap=overlap,
    )
    print(f"Total embedded chunks: {len(chunks)}")

    upsert_chunks_to_qdrant(chunks)


if __name__ == "__main__":
    run_embed_and_index()
