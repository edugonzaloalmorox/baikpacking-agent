import argparse
from typing import List, Tuple

from baikpacking.embedding.config import Settings
from baikpacking.embedding.embed import embed_riders_rows
from baikpacking.db.db_connection import get_pg_connection
from baikpacking.db.data_loader import (
    fetch_riders,
    truncate_rider_embeddings,
    upsert_rider_embeddings,
)

settings = Settings()


def run_embed_riders(rebuild: bool = False, expected_dim: int = 1024) -> None:
    """
    Rebuild or incrementally upsert rider embeddings into Postgres/pgvector.

    Flow:
      1) fetch riders from Postgres
      2) embed 1 vector per rider
      3) upsert into rider_embeddings (PK = rider_id)
    """
    with get_pg_connection(autocommit=False) as conn:
        if rebuild:
            truncate_rider_embeddings(conn)

        rows = fetch_riders(conn)
        print(f"Riders fetched: {len(rows)}")

        pairs = embed_riders_rows(rows, expected_dim=expected_dim)
        print(f"Embeddings computed: {len(pairs)}")

        # Convert (rider_id, vector) -> (rider_id, vector, model)
        records: List[Tuple[int, List[float], str]] = [
            (rider_id, vec, settings.embedding_model) for (rider_id, vec) in pairs
        ]

        n = upsert_rider_embeddings(conn, records, page_size=500)
        print(f"Upsert complete. Rows written: {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed riders into Postgres (pgvector).")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Truncate rider_embeddings before inserting (full rebuild).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1024,
        help="Expected embedding dimension. Must match rider_embeddings.embedding vector(dim).",
    )
    args = parser.parse_args()

    run_embed_riders(rebuild=args.rebuild, expected_dim=args.dim)


if __name__ == "__main__":
    main()