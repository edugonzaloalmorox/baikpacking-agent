import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

log = logging.getLogger("baikpacking.trace")


def _short(obj: Any, max_chars: int = 900) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        s = str(obj)
    if len(s) > max_chars:
        return s[: max_chars - 20] + "\n...<truncated>..."
    return s


@dataclass
class AgentTracer:
    enabled: bool = False
    step: int = 0

    def next_step(self, title: str) -> None:
        if not self.enabled:
            return
        self.step += 1
        log.info("┌─[STEP %02d] %s", self.step, title)

    def model_output(self, content_preview: str) -> None:
        if not self.enabled:
            return
        log.info("│ model: %s", (content_preview or "").strip()[:400])

    def tool_call(self, name: str, args: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        log.info("│ tool_call: %s args=%s", name, _short(args, 700).replace("\n", " "))

    def tool_result(self, name: str, result: Any) -> None:
        if not self.enabled:
            return
        log.info("│ tool_result: %s -> %s", name, _short(result, 900).replace("\n", " "))

    def end_step(self) -> None:
        if not self.enabled:
            return
        log.info("└─")