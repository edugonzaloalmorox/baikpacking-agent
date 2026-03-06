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
    Collects call events during a run.

    JSON-serializable events are exposed through:
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

    def record(
        self,
        tool: str,
        *,
        args: Optional[Dict[str, Any]] = None,
        result: Any = None,
        elapsed_ms: Optional[float] = None,
    ) -> None:
        """
        Convenience wrapper for adding a trace event.
        """
        self.add(
            tool=tool,
            args=args or {},
            result={} if result is None else result,
            elapsed_ms=0.0 if elapsed_ms is None else float(elapsed_ms),
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


def get_call_trace_from_deps(deps: Any) -> Optional[CallTrace]:
    """
    Plain Python helper for orchestration code.
    """
    return getattr(deps, "call_trace", None) if deps is not None else None


def _get_trace(ctx: RunContext) -> Optional[CallTrace]:
    deps = getattr(ctx, "deps", None)
    return get_call_trace_from_deps(deps)


def record_trace_call(
    *,
    deps: Any,
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    result: Any = None,
    elapsed_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Plain Python tracing helper for deterministic orchestration.

    Use this from application code instead of calling the @Tool wrapper.
    """
    trace = get_call_trace_from_deps(deps)
    if trace is not None:
        try:
            trace.record(
                tool=tool_name,
                args=args or {},
                result={} if result is None else result,
                elapsed_ms=elapsed_ms,
            )
        except Exception:
            logger.debug("trace.record failed inside record_trace_call", exc_info=True)

    return {
        "ok": True,
        "tool_name": tool_name,
        "args": args or {},
        "result": {} if result is None else result,
        "elapsed_ms": 0.0 if elapsed_ms is None else float(elapsed_ms),
    }


def time_and_record(
    *,
    deps: Any,
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    fn,
) -> Any:
    """
    Execute a plain Python function, measure elapsed time, and record it.

    Example:
        result = time_and_record(
            deps=deps,
            tool_name="search_similar_riders",
            args={"query": query, "top_k_riders": 5},
            fn=lambda: run_search_similar_riders(...),
        )
    """
    t0 = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    summary: Any
    if isinstance(result, dict):
        summary = result
    elif isinstance(result, list):
        summary = {"count": len(result)}
    elif isinstance(result, str):
        summary = {"chars": len(result)}
    else:
        summary = {"type": type(result).__name__}

    record_trace_call(
        deps=deps,
        tool_name=tool_name,
        args=args or {},
        result=summary,
        elapsed_ms=elapsed_ms,
    )
    return result


@Tool
def trace_tool_call(
    ctx: RunContext,
    tool_name: str,
    stage: str,
    note: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Lightweight tracing tool for agent-driven loops.

    For deterministic orchestration in Python, use record_trace_call(...)
    instead of calling this tool wrapper directly.
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
                args={
                    "tool_name": tool_name,
                    "stage": stage,
                    "note": note,
                    "extra": extra or {},
                },
                result={"ok": True},
                elapsed_ms=elapsed_ms,
            )
        except Exception:
            logger.debug("trace.add failed inside trace_tool_call", exc_info=True)

    return {"ok": True, **payload}