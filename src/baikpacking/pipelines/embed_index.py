import argparse

from baikpacking.embedding.config import Settings
from baikpacking.db.db_connection import get_pg_connection
from baikpacking.db.data_loader import (
    build_and_embed_chunks,
    truncate_rider_chunks,
)

settings = Settings()


def run_embed_riders(
    rebuild: bool = False,
    expected_dim: int = 1024,
    embed_all: bool = False,
) -> None:
    """Rebuild or incrementally embed rider chunks into Postgres/pgvector."""
    with get_pg_connection(autocommit=False) as conn:
        if rebuild:
            truncate_rider_chunks(conn)

        only_missing = not (rebuild or embed_all)
        mode = "ALL riders" if not only_missing else "ONLY riders missing chunks"
        print(f"Mode: {mode}")

        stats = build_and_embed_chunks(
            conn,
            model_name=settings.embedding_model,
            only_missing=only_missing,
            batch_size=128,
            dry_run=False,
        )
        print(f"Upsert complete. Stats: {stats}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed riders into Postgres (pgvector).")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Truncate rider_embeddings before inserting (full rebuild).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Embed all riders (overrides incremental behavior).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1024,
        help="Expected embedding dimension. Must match rider_embeddings.embedding vector(dim).",
    )
    args = parser.parse_args()

    run_embed_riders(
        rebuild=args.rebuild,
        expected_dim=args.dim,
        embed_all=args.all,
    )


if __name__ == "__main__":
    main()
