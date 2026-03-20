# src/evaluation/db.py
"""
Single source of truth for the TruLens database path and session factory.

Architecture note: in production, we replace the sqlite URL with a Postgres
connection string and all three processes (eval script, dashboard, Streamlit)
will share the same DB with proper concurrent-write support:
"""

from pathlib import Path
from functools import lru_cache

from trulens.core import TruSession

from src.config import settings

# Absolute path — derived from settings so it moves if processed_data_dir changes,
DB_PATH = Path(settings.processed_data_dir).resolve() / "trulens.sqlite"
DB_URL  = f"sqlite:///{DB_PATH}"


def get_session(reset: bool = False) -> TruSession:
    """
    Return a TruSession connected to the shared SQLite database.

    Args:
        reset: If True, wipe all records before returning. Only ever pass
               True from an explicit CLI command — never from dashboard or
               eval code that runs automatically.

    The session is NOT cached with lru_cache because TruLens manages its
    own internal singleton — calling TruSession(database_url=X) twice with
    the same URL returns the same underlying connection. The factory pattern
    here is for path consistency, not connection pooling.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    session = TruSession(database_url=DB_URL)
    if reset:
        session.reset_database()
    return session