import time
from typing import Any, Callable, Dict, Optional

from pydantic_ai import RunContext


def traced_tool(
    tool_func: Callable[..., Any],
    *,
    tool_name: Optional[str] = None,
) -> Callable[..., Any]:
    """
    Wrap a pydantic-ai tool function to capture call args + timing in deps.call_trace.

    Works with @Tool-decorated functions because pydantic-ai will still treat the wrapper
    as the callable tool implementation.
    """
    name = tool_name or getattr(tool_func, "__name__", "unknown_tool")

    def _wrapped(ctx: RunContext, *args: Any, **kwargs: Any) -> Any:
        trace = getattr(getattr(ctx, "deps", None), "call_trace", None)

        t0 = time.perf_counter()
        result = tool_func(ctx, *args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if trace is not None:
            # Best-effort args capture (avoid giant payloads)
            arg_dict: Dict[str, Any] = {}
            if args:
                arg_dict["_args"] = [repr(a)[:200] for a in args]
            for k, v in kwargs.items():
                # don't explode logs with long strings / huge lists
                rv = v
                if isinstance(v, str) and len(v) > 500:
                    rv = v[:500] + "…"
                elif isinstance(v, list) and len(v) > 50:
                    rv = {"type": "list", "len": len(v)}
                arg_dict[k] = rv

            trace.add(name, arg_dict, result, elapsed_ms)

        return result

    _wrapped.__name__ = f"traced_{name}"
    return _wrapped