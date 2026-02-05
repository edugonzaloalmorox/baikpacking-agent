import os
from contextlib import contextmanager
from typing import Iterator, Optional
from urllib.parse import urlparse

import psycopg2
from psycopg2.extensions import connection as PGConnection

try:
    # Optional: load local .env for CLI runs (Docker/CI usually inject env already)
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


def _default_dsn() -> str:
    return "postgresql://baikpacking:baikpacking@localhost:5433/baikpacking"


def _mask_dsn(dsn: str) -> str:
    """
    Return DSN with password removed for safe logging.
    """
    try:
        u = urlparse(dsn)
        if u.password:
            netloc = u.netloc.replace(f":{u.password}", ":***")
            return u._replace(netloc=netloc).geturl()
        return dsn
    except Exception:
        return "<unparseable dsn>"


def get_db_dsn() -> str:
    """
    Single source of truth for DB connection.
    Prefers DATABASE_URL, falls back to a local dev DSN.
    """
    return os.getenv("DATABASE_URL") or _default_dsn()


@contextmanager
def get_pg_connection(autocommit: bool = False) -> Iterator[PGConnection]:
    """
    Context manager that yields a PostgreSQL connection.

    - Uses DATABASE_URL (or default local DSN)
    - Ensures connections are always closed
    - Optional autocommit for loaders / pipelines
    """
    dsn = get_db_dsn()
    conn = psycopg2.connect(dsn)
    conn.autocommit = autocommit
    try:
        yield conn
    finally:
        conn.close()


def ping_db() -> dict:
    """
    Lightweight connectivity check. Safe to print/log.
    """
    dsn = get_db_dsn()
    with get_pg_connection(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("select current_database(), current_user, inet_server_addr(), inet_server_port();")
            db, user, host, port = cur.fetchone()
    return {
        "dsn": _mask_dsn(dsn),
        "database": db,
        "user": user,
        "server_host": str(host),
        "server_port": int(port),
    }
