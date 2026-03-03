import time
from typing import Any, Dict, Optional

from pydantic_ai import RunContext

from baikpacking.tools.call_trace import CallTrace


def trace_tool(ctx: RunContext, tool: str, args: Dict[str, Any], result: Any, t0: float) -> None:
    deps = getattr(ctx, "deps", None)
    trace: Optional[CallTrace] = getattr(deps, "call_trace", None)
    if trace is None:
        return
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    trace.add(tool, args=args, result=result, elapsed_ms=elapsed_ms)