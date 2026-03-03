import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic_ai import RunContext, Tool

logger = logging.getLogger(__name__)


def _clip(v: Any, n: int = 240) -> Any:
    try:
        s = str(v)
    except Exception:
        return "<unprintable>"
    return s if len(s) <= n else s[: n - 1] + "…"


@dataclass
class CallTraceEvent:
    tool: str
    args: Dict[str, Any]
    result: Any
    elapsed_ms: float


@dataclass
class CallTrace:
    """
    Collects tool-call events during an agent run.

    Tools should call:
      trace.add(tool=..., args=..., result=..., elapsed_ms=...)
    Runner can serialize:
      trace.calls
    """
    _events: List[CallTraceEvent] = field(default_factory=list)

    def add(self, tool: str, args: Dict[str, Any], result: Any, elapsed_ms: float) -> None:
        self._events.append(
            CallTraceEvent(
                tool=str(tool),
                args=args or {},
                result=result,
                elapsed_ms=float(elapsed_ms),
            )
        )

    def __str__(self) -> str:
        lines = []
        for i, e in enumerate(self.calls, 1):
            lines.append(
                f"[{i}] {e['tool']} ({e['elapsed_ms']:.1f} ms) "
                f"args={_clip(e['args'])} result={_clip(e['result'])}"
            )
        return "\n".join(lines)

    @property
    def calls(self) -> List[Dict[str, Any]]:
        """JSON-serializable view."""
        return [
            {
                "tool": e.tool,
                "args": e.args,
                "result": e.result,
                "elapsed_ms": e.elapsed_ms,
            }
            for e in self._events
        ]


def _get_trace(ctx: RunContext) -> Optional[CallTrace]:
    deps = getattr(ctx, "deps", None)
    return getattr(deps, "call_trace", None)


@Tool
def trace_tool_call(
    ctx: RunContext,
    tool_name: str,
    stage: str,
    note: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Lightweight tracing tool the LLM can call to prove the loop is running.
    """
    t0 = time.perf_counter()
    payload: Dict[str, Any] = {
        "tool_name": tool_name,
        "stage": stage,
        "note": note,
        "extra": extra or {},
    }

    trace = _get_trace(ctx)
    if trace is not None:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        try:
            trace.add(
                tool="trace_tool_call",
                args={"tool_name": tool_name, "stage": stage, "note": note, "extra": extra or {}},
                result={"ok": True},
                elapsed_ms=elapsed_ms,
            )
        except Exception:
            logger.debug("trace.add failed inside trace_tool_call", exc_info=True)

    return {"ok": True, **payload}