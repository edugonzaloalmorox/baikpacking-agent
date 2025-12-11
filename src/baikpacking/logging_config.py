import logging
import os


def setup_logging(default_level: int = logging.INFO) -> None:
    """
    Basic logging configuration for the project.

    Call this once at startup (e.g. in your CLI, FastAPI app, or __main__).
    """
    if logging.getLogger().handlers:
        # Already configured (e.g. by a framework or tests) – don't touch it.
        return

    log_level_name = os.getenv("LOG_LEVEL", "").upper()
    level = getattr(logging, log_level_name, default_level)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
