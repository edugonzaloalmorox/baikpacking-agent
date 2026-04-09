"""Reflex configuration for the bikepacking UI."""



import sys
from pathlib import Path

import reflex as rx


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


config = rx.Config(
    app_name="reflex_ui",
    frontend_port=3000,
    backend_port=8000,
    api_url="http://localhost:8000",
)