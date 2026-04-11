"""Application state for the bikepacking chat UI."""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import httpx
import reflex as rx
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

DEFAULT_QUERY = "What tyres do you recommend for Atlas Mountain Race?"
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
EXAMPLE_PROMPTS = [
    "What tyres do you recommend for Atlas Mountain Race?",
    "Recommend me a setup for Tour Divide",
    "What bags should I use for Transpyrenees?",
    "What drivetrain works well for Badlands?",
    "Give me a full setup for Silk Road Mountain Race",
]


class ChatTurn(BaseModel):
    """One conversation turn rendered in the UI."""

    id: str
    role: Literal["user", "assistant", "error"]
    content: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str = ""
    resolved_event_name: str = ""
    resolved_event_match_type: str = ""
    resolved_event_chip_label: str = ""
    intent_component: str = ""
    intent_chip_label: str = ""
    policy_mode: str = ""
    policy_chip_label: str = ""
    policy_notes: str = ""
    evidence_rider_count: str = ""
    evidence_component_hit_count: str = ""
    evidence_strength: str = ""
    evidence_consistency: str = ""
    reasoning: str = ""
    setup_lines: list[str] = Field(default_factory=list)
    field_support_lines: list[str] = Field(default_factory=list)
    retrieval_plan_json: str = ""
    trace_json: str = ""
    has_debug: bool = False
    feedback_status: str = ""
    feedback_comment: str = ""
    feedback_form_open: bool = False
    feedback_error: str = ""
    error: str = ""


def _normalize_base_url(value: Optional[str]) -> str:
    base_url = (value or DEFAULT_API_BASE_URL).strip()
    return base_url.rstrip("/") or DEFAULT_API_BASE_URL


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        else:
            text = str(value).strip()
            if text:
                return text
    return "—"


def _safe_get(data: Any, *path: str, default: Any = None) -> Any:
    current = data
    for key in path:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
        if current is None:
            return default
    return current if current is not None else default


def _safe_json(data: Any) -> str:
    if data is None:
        return ""
    try:
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        return str(data)


def _extract_api_error(payload: Any, fallback: str) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        return _safe_json(payload)
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    return fallback


def _setup_pairs(response: dict[str, Any]) -> list[tuple[str, str]]:
    setup = _safe_get(response, "recommendation", "recommended_setup", default={})
    if not isinstance(setup, dict):
        return []

    fields = [
        ("Bike", ("bike", "bike_type")),
        ("Wheels", ("wheels",)),
        ("Tyres", ("tyres",)),
        ("Drivetrain", ("drivetrain",)),
        ("Bags", ("bags",)),
        ("Sleep system", ("sleep_system",)),
        ("Lighting", ("lighting",)),
        ("Navigation", ("navigation",)),
        ("Water capacity", ("water_capacity",)),
        ("Notes", ("notes",)),
    ]
    pairs: list[tuple[str, str]] = []
    for label, keys in fields:
        value = ""
        for key in keys:
            candidate = setup.get(key)
            if isinstance(candidate, str) and candidate.strip():
                value = candidate.strip()
                break
        if value:
            pairs.append((label, value))
    return pairs


def _setup_lines_from_pairs(pairs: list[tuple[str, str]]) -> list[str]:
    return [f"{label}: {value}" for label, value in pairs]


def _field_support_lines(response: dict[str, Any]) -> list[str]:
    evidence = _safe_get(response, "evidence", default={})
    if not isinstance(evidence, dict):
        return []
    field_support = evidence.get("field_support", {})
    if not isinstance(field_support, dict):
        return []
    lines = []
    for key, value in field_support.items():
        if isinstance(value, str) and value.strip():
            lines.append(f"{key}: {value.strip()}")
    return lines


def _build_assistant_text(response: dict[str, Any]) -> str:
    summary = _first_non_empty(_safe_get(response, "recommendation", "summary"))
    event_name = _first_non_empty(_safe_get(response, "resolved_event", "display_name"))

    intro = f"For {event_name}, " if event_name != "—" else ""
    if summary == "—":
        summary = "Here’s a grounded recommendation."

    return f"{intro}{summary}"


def _build_assistant_turn(response: dict[str, Any]) -> ChatTurn:
    recommendation = _safe_get(response, "recommendation", default={}) or {}
    evidence = _safe_get(response, "evidence", default={}) or {}
    policy = _safe_get(response, "policy", default={}) or {}
    debug = _safe_get(response, "debug", default={}) or {}
    resolved_event = _safe_get(response, "resolved_event", default={}) or {}
    intent = _safe_get(response, "intent", default={}) or {}

    return ChatTurn(
        id=uuid.uuid4().hex,
        role="assistant",
        content=_build_assistant_text(response),
        run_id=_first_non_empty(_safe_get(response, "run_id")),
        resolved_event_name=_first_non_empty(resolved_event.get("display_name")),
        resolved_event_match_type=_first_non_empty(resolved_event.get("match_type")),
        resolved_event_chip_label=_first_non_empty(resolved_event.get("display_name"), "Grounded answer"),
        intent_component=_first_non_empty(intent.get("component")),
        intent_chip_label=_first_non_empty(intent.get("component"), "Advice"),
        policy_mode=_first_non_empty(policy.get("mode")),
        policy_chip_label=_first_non_empty(policy.get("mode"), "Policy"),
        policy_notes=_first_non_empty(
            ", ".join(
                str(item).strip()
                for item in (policy.get("notes") or [])
                if isinstance(item, str) and item.strip()
            )
        ),
        evidence_rider_count=_first_non_empty(evidence.get("rider_count")),
        evidence_component_hit_count=_first_non_empty(evidence.get("component_hit_count")),
        evidence_strength=_first_non_empty(evidence.get("evidence_strength")),
        evidence_consistency=_first_non_empty(evidence.get("consistency")),
        reasoning=_first_non_empty(recommendation.get("reasoning"), "No reasoning was returned by the backend."),
        setup_lines=_setup_lines_from_pairs(_setup_pairs(response)),
        field_support_lines=_field_support_lines(response),
        retrieval_plan_json=_safe_json(debug.get("retrieval_plan")) if isinstance(debug, dict) else "",
        trace_json=_safe_json(debug.get("trace", [])) if isinstance(debug, dict) else "",
        has_debug=bool(debug),
    )


def _build_error_turn(message: str) -> ChatTurn:
    return ChatTurn(
        id=uuid.uuid4().hex,
        role="error",
        content=message,
        error=message,
    )


class BikepackingState(rx.State):
    """Shared state for the chat-first recommendation UI."""

    query: str = DEFAULT_QUERY
    loading: bool = False
    error: str = ""
    messages: list[ChatTurn] = []
    include_debug: bool = True
    api_base_url: str = _normalize_base_url(os.getenv("API_BASE_URL"))
    loading_stage_key: str = ""
    loading_stage_label: str = ""
    loading_stage_history: list[dict[str, Any]] = []

    def set_query(self, value: str) -> None:
        """Update the composer text."""
        self.query = value

    def load_example(self, value: str) -> None:
        """Insert a sample prompt into the composer."""
        self.query = value
        self.error = ""

    def clear_error(self) -> None:
        """Clear any visible local error state."""
        self.error = ""

    def _reset_loading_progress(self) -> None:
        """Clear transient loading-stage state."""
        self.loading_stage_key = ""
        self.loading_stage_label = ""
        self.loading_stage_history = []

    def _set_loading_stage(self, stage: dict[str, Any]) -> None:
        """Record the current recommendation stage."""
        stage_key = str(stage.get("stage_key") or "")
        stage_label = str(stage.get("stage_label") or "")
        if not stage_key or not stage_label:
            return

        current = {
            "stage_key": stage_key,
            "stage_label": stage_label,
            "timestamp": stage.get("timestamp"),
            "details": stage.get("details") or {},
        }

        history = list(self.loading_stage_history or [])

        if history:
            last = history[-1]
            if last.get("stage_key") == stage_key and last.get("stage_label") == stage_label:
                self.loading_stage_key = stage_key
                self.loading_stage_label = stage_label
                return

        history.append(current)
        self.loading_stage_history = history
        self.loading_stage_key = stage_key
        self.loading_stage_label = stage_label

    def _update_turn(self, run_id: str, **updates: Any) -> None:
        """Update one assistant turn in place by run id."""
        for index, turn in enumerate(self.messages):
            if turn.role == "assistant" and turn.run_id == run_id:
                self.messages[index] = turn.model_copy(update=updates)
                return

    def open_feedback_form(self, run_id: str) -> None:
        """Reveal the thumbs-down comment form for a turn."""
        self._update_turn(run_id, feedback_form_open=True, feedback_error="")

    def set_feedback_comment(self, run_id: str, value: str) -> None:
        """Update the in-progress feedback comment for a turn."""
        self._update_turn(run_id, feedback_comment=value)

    async def submit_feedback(self, run_id: str, feedback: str) -> None:
        """Send user feedback for a prior recommendation."""
        run_id = (run_id or "").strip()
        if not run_id:
            async with self:
                self.error = "Missing run id for feedback submission."
            return

        target_turn = None
        for turn in reversed(self.messages):
            if turn.role == "assistant" and turn.run_id == run_id:
                target_turn = turn
                break

        if target_turn is None:
            async with self:
                self.error = "Could not find the assistant turn for that feedback."
            return

        comment = target_turn.feedback_comment.strip() if target_turn.feedback_comment else ""
        payload: dict[str, Any] = {"run_id": run_id, "feedback": feedback}
        if comment:
            payload["comment"] = comment

        url = f"{self.api_base_url}/feedback"
        try:
            timeout = httpx.Timeout(30.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)

            if resp.status_code >= 400:
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text
                raise RuntimeError(_extract_api_error(body, f"HTTP {resp.status_code} from feedback API"))

            data = resp.json()
            if not isinstance(data, dict):
                raise RuntimeError("Unexpected feedback response shape.")

            async with self:
                self._update_turn(
                    run_id,
                    feedback_status=str(data.get("feedback") or feedback),
                    feedback_form_open=False,
                    feedback_error="",
                )
        except httpx.RequestError as exc:
            logger.exception("bikepacking_ui_feedback_request_error")
            async with self:
                self._update_turn(run_id, feedback_error="Could not reach the feedback service.")
                self.error = str(exc)
        except Exception as exc:
            logger.exception("bikepacking_ui_feedback_error")
            async with self:
                self._update_turn(run_id, feedback_error="Could not record that feedback right now.")
                self.error = str(exc)

    @rx.var
    def can_send(self) -> bool:
        return bool((self.query or "").strip()) and not self.loading

    @rx.var
    def has_messages(self) -> bool:
        return bool(self.messages)

    @rx.var
    def message_count(self) -> int:
        return len(self.messages or [])

    @rx.var
    def latest_debug_trace(self) -> str:
        for turn in reversed(self.messages or []):
            if turn.role == "assistant" and turn.trace_json:
                return turn.trace_json
        return ""
    
    @rx.event(background=True)
    async def submit_query(self) -> None:
        """Send the current prompt to the FastAPI recommender."""
        query = (self.query or "").strip()
        if not query:
            async with self:
                self.error = "Enter a bikepacking question before sending it."
            return

        user_turn = ChatTurn(
            id=uuid.uuid4().hex,
            role="user",
            content=query,
        )

        async with self:
            self.messages.append(user_turn)
            self.query = ""
            self.error = ""
            self.loading = True
            self._reset_loading_progress()
            self.loading_stage_key = "starting"
            self.loading_stage_label = "Starting recommendation"

        stream_url = f"{self.api_base_url}/recommend/stream"
        fallback_url = f"{self.api_base_url}/recommend"
        payload = {"query": query, "include_debug": self.include_debug}

        try:
            timeout = httpx.Timeout(90.0, connect=10.0, read=None)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", stream_url, json=payload) as resp:
                    if resp.status_code >= 400:
                        raise httpx.RequestError(
                            f"HTTP {resp.status_code} from recommendation stream",
                            request=resp.request,
                        )

                    final_response: dict[str, Any] | None = None
                    saw_progress = False

                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            event = json.loads(line)
                        except Exception:
                            continue

                        if not isinstance(event, dict):
                            continue

                        kind = str(event.get("kind") or "")

                        if kind == "progress":
                            progress = event.get("progress")
                            if isinstance(progress, dict):
                                saw_progress = True
                                async with self:
                                    self._set_loading_stage(progress)
                                await asyncio.sleep(0)
                            continue

                        if kind == "final":
                            response = event.get("response")
                            if isinstance(response, dict):
                                final_response = response
                            break

                        if kind == "error":
                            raise RuntimeError(_extract_api_error(event, "Recommendation stream failed."))

                    if final_response is None:
                        if not saw_progress:
                            raise httpx.RequestError(
                                "Recommendation stream ended without a final response.",
                                request=resp.request,
                            )
                        raise RuntimeError("Recommendation stream ended without a final response.")

                    assistant_turn = _build_assistant_turn(final_response)
                    async with self:
                        self.messages.append(assistant_turn)

        except httpx.RequestError as exc:
            logger.exception("bikepacking_ui_request_error")
            try:
                timeout = httpx.Timeout(90.0, connect=10.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(fallback_url, json=payload)

                if resp.status_code >= 400:
                    try:
                        body = resp.json()
                    except Exception:
                        body = resp.text
                    raise RuntimeError(_extract_api_error(body, f"HTTP {resp.status_code} from recommender API"))

                data = resp.json()
                if not isinstance(data, dict):
                    raise RuntimeError("Unexpected API response shape.")

                assistant_turn = _build_assistant_turn(data)
                async with self:
                    self.messages.append(assistant_turn)

            except Exception:
                message = (
                    f"I couldn’t reach the recommendation service at {self.api_base_url}. "
                    "Please try again in a moment."
                )
                async with self:
                    self.messages.append(_build_error_turn(message))
                    self.error = str(exc)

            finally:
                async with self:
                    self._reset_loading_progress()

        except Exception as exc:
            logger.exception("bikepacking_ui_recommend_error")
            message = "I hit a problem generating that recommendation. Please try again."
            async with self:
                self.messages.append(_build_error_turn(message))
                self.error = str(exc)

        finally:
            async with self:
                self.loading = False
                #self._reset_loading_progress()