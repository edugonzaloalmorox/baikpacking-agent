import os
import sys
from pathlib import Path

import reflex as rx


HERE = Path(__file__).resolve()

try:
    candidate_root = HERE.parents[2]
    if (candidate_root / "src").exists():
        ROOT = candidate_root
    else:
        ROOT = HERE.parent
except IndexError:
    ROOT = HERE.parent

SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


config = rx.Config(
    app_name="reflex_ui",
    frontend_port=3000,
    backend_port=3000,
    api_url=os.getenv("REFLEX_API_URL", "http://localhost:3000"),
)