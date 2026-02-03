import os
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2.extensions import connection as PGConnection


# Single source of truth for DB connection
DB_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://baikpacking:baikpacking@localhost:5433/baikpacking",
)


@contextmanager
def get_pg_connection(autocommit: bool = False) -> Iterator[PGConnection]:
    """
    Context manager that yields a PostgreSQL connection.

    - Uses DATABASE_URL
    - Ensures connections are always closed
    - Optional autocommit for loaders / pipelines
    """
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = autocommit
    try:
        yield conn
    finally:
        conn.close()
