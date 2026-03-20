# scripts/run_admin_dashboard.py
"""
scripts/run_admin_dashboard.py  —  App 2: TruLens Admin Dashboard

Audience: MLEs and evaluators only.
Reads from the shared TruLens SQLite DB defined in src/evaluation/db.py.

Concurrent operation:
    Terminal 1: uvicorn src.api.main:app --reload --port 8000
    Terminal 2: streamlit run ui/app.py --server.port 8501
    Terminal 3: python scripts/run_admin_dashboard.py      # port 8502

IMPORTANT: Never call session.reset_database() here.
           The dashboard is read-only. Resets belong in run_trulens_eval.py
           behind the explicit --reset-db flag.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from trulens.dashboard import run_dashboard

from src.evaluation.db import get_session, DB_PATH


def main() -> None:
    logger.info(f"TruLens DB: {DB_PATH}")

    if not DB_PATH.exists():
        logger.warning(
            f"Database not found at {DB_PATH}. "
            "Run evaluations first:\n"
            "  python scripts/run_trulens_eval.py --mode both"
        )

    session = get_session(reset=False)

    logger.info("TruLens admin dashboard starting at http://localhost:8502")
    logger.info("App 1 (customer UI) runs at http://localhost:8501 — independent port.")

    run_dashboard(session, port=8502, _dev=False)


if __name__ == "__main__":
    main()