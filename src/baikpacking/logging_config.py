import logging
import os
import logfire
from pydantic_ai import settings
from dotenv import load_dotenv

load_dotenv()


def setup_logging(default_level: int = logging.INFO) -> None:
    """
    Configures standard logging AND Logfire instrumentation.
    """
    # 1. Standard Python Logging Setup
    log_level_name = os.getenv("LOG_LEVEL", "").upper()
    level = getattr(logging, log_level_name, default_level)

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    # 2. Logfire Initialization
    # This automatically picks up LOGFIRE_TOKEN from your environment
    logfire.configure(
        send_to_logfire=os.getenv("LOGFIRE_TOKEN") is not None,
    )

    # 3. Automatic Instrumentation
    # This captures all Pydantic AI agent runs and tool calls
    logfire.instrument_pydantic_ai()
    
    # This captures your Web Search HTTP requests
    logfire.instrument_httpx(capture_all=True)
