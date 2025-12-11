import os
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2.extensions import connection as PGConnection


DB_DSN = os.getenv("DOTWATCHER_DB_DSN", "postgresql://localhost:5432/dotwatcher")


@contextmanager
def get_pg_connection() -> Iterator[PGConnection]:
    """
    Context manager that yields a PostgreSQL connection using DOTWATCHER_DB_DSN.
    """
    conn = psycopg2.connect(DB_DSN)
    try:
        yield conn
    finally:
        conn.close()