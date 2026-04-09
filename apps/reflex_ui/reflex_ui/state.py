"""Application state for the bikepacking Reflex UI."""



import json
import logging
import os
from typing import Any, Optional

import httpx
import reflex as rx


logger = logging.getLogger(__name__)

DEFAULT_QUERY = "What tyres do you recommend for Atlas Mountain Race?"
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


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


def _format_bool(value: Any) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "—"


def _format_int(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _first_non_empty(value)


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


class BikepackingState(rx.State):
    """Shared state for the recommendation UI."""

    query: str = DEFAULT_QUERY
    loading: bool = False
    error: str = ""
    response: dict[str, Any] = {}
    include_debug: bool = True
    show_debug: bool = False
    api_base_url: str = _normalize_base_url(os.getenv("API_BASE_URL"))

    def set_query(self, value: str) -> None:
        """Update the query input."""
        self.query = value

    def load_example(self, value: str) -> None:
        """Load an example query into the input."""
        self.query = value
        self.error = ""

    def toggle_debug(self) -> None:
        """Toggle the visibility of the debug panel."""
        self.show_debug = not self.show_debug

    async def submit_query(self) -> None:
        """Call the FastAPI recommender endpoint and store the JSON response."""
        query = (self.query or "").strip()
        if not query:
            self.error = "Enter a bikepacking question before generating a recommendation."
            return

        self.loading = True
        self.error = ""
        self.response = {}

        url = f"{self.api_base_url}/recommend"
        payload = {"query": query, "include_debug": self.include_debug}

        try:
            timeout = httpx.Timeout(90.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)

            if resp.status_code >= 400:
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text
                raise RuntimeError(_extract_api_error(body, f"HTTP {resp.status_code} from recommender API"))

            data = resp.json()
            if not isinstance(data, dict):
                raise RuntimeError("Unexpected API response shape.")

            self.response = data
        except httpx.RequestError as exc:
            logger.exception("bikepacking_ui_request_error")
            self.error = f"Could not reach the recommender API at {self.api_base_url}: {exc}"
        except Exception as exc:
            logger.exception("bikepacking_ui_recommend_error")
            self.error = str(exc)
        finally:
            self.loading = False

    @rx.var
    def has_response(self) -> bool:
        return bool(self.response)

    @rx.var
    def resolved_event_name(self) -> str:
        return _first_non_empty(_safe_get(self.response, "resolved_event", "display_name"), "—")

    @rx.var
    def resolved_event_match_type(self) -> str:
        return _first_non_empty(_safe_get(self.response, "resolved_event", "match_type"), "—")

    @rx.var
    def resolved_event_confidence(self) -> str:
        confidence = _safe_get(self.response, "resolved_event", "confidence")
        if isinstance(confidence, (int, float)):
            return f"{confidence:.2f}"
        return "—"

    @rx.var
    def intent_component(self) -> str:
        return _first_non_empty(_safe_get(self.response, "intent", "component"), "—")

    @rx.var
    def intent_confidence(self) -> str:
        confidence = _safe_get(self.response, "intent", "confidence")
        if isinstance(confidence, (int, float)):
            return f"{confidence:.2f}"
        return "—"

    @rx.var
    def recommendation_summary(self) -> str:
        return _first_non_empty(_safe_get(self.response, "recommendation", "summary"), "No recommendation returned.")

    @rx.var
    def recommendation_reasoning(self) -> str:
        return _first_non_empty(_safe_get(self.response, "recommendation", "reasoning"), "—")

    @rx.var
    def setup_bike(self) -> str:
        return _first_non_empty(
            _safe_get(self.response, "recommendation", "recommended_setup", "bike"),
            _safe_get(self.response, "recommendation", "recommended_setup", "bike_type"),
        )

    @rx.var
    def setup_wheels(self) -> str:
        return _first_non_empty(_safe_get(self.response, "recommendation", "recommended_setup", "wheels"))

    @rx.var
    def setup_tyres(self) -> str:
        return _first_non_empty(_safe_get(self.response, "recommendation", "recommended_setup", "tyres"))

    @rx.var
    def setup_drivetrain(self) -> str:
        return _first_non_empty(_safe_get(self.response, "recommendation", "recommended_setup", "drivetrain"))

    @rx.var
    def setup_bags(self) -> str:
        return _first_non_empty(_safe_get(self.response, "recommendation", "recommended_setup", "bags"))

    @rx.var
    def setup_sleep_system(self) -> str:
        return _first_non_empty(_safe_get(self.response, "recommendation", "recommended_setup", "sleep_system"))

    @rx.var
    def setup_lighting(self) -> str:
        return _first_non_empty(_safe_get(self.response, "recommendation", "recommended_setup", "lighting"))

    @rx.var
    def setup_navigation(self) -> str:
        return _first_non_empty(_safe_get(self.response, "recommendation", "recommended_setup", "navigation"))

    @rx.var
    def setup_water_capacity(self) -> str:
        return _first_non_empty(_safe_get(self.response, "recommendation", "recommended_setup", "water_capacity"))

    @rx.var
    def setup_notes(self) -> str:
        return _first_non_empty(_safe_get(self.response, "recommendation", "recommended_setup", "notes"))

    @rx.var
    def evidence_rider_count(self) -> str:
        return _format_int(_safe_get(self.response, "evidence", "rider_count"))

    @rx.var
    def evidence_component_hit_count(self) -> str:
        return _format_int(_safe_get(self.response, "evidence", "component_hit_count"))

    @rx.var
    def evidence_strength(self) -> str:
        return _first_non_empty(_safe_get(self.response, "evidence", "evidence_strength"))

    @rx.var
    def evidence_consistency(self) -> str:
        return _first_non_empty(_safe_get(self.response, "evidence", "consistency"))

    @rx.var
    def policy_mode(self) -> str:
        return _first_non_empty(_safe_get(self.response, "policy", "mode"))

    @rx.var
    def policy_allow_specific_brands(self) -> str:
        return _format_bool(_safe_get(self.response, "policy", "allow_specific_brands"))

    @rx.var
    def policy_allow_specific_specs(self) -> str:
        return _format_bool(_safe_get(self.response, "policy", "allow_specific_specs"))

    @rx.var
    def policy_allow_event_specific_claims(self) -> str:
        return _format_bool(_safe_get(self.response, "policy", "allow_event_specific_claims"))

    @rx.var
    def policy_notes(self) -> str:
        notes = _safe_get(self.response, "policy", "notes", default=[])
        if isinstance(notes, list) and notes:
            return " • ".join(str(item) for item in notes if str(item).strip())
        return "—"

    @rx.var
    def has_debug(self) -> bool:
        return bool(_safe_get(self.response, "debug"))

    @rx.var
    def debug_retrieval_plan(self) -> str:
        return _safe_json(_safe_get(self.response, "debug", "retrieval_plan"))

    @rx.var
    def debug_trace(self) -> str:
        return _safe_json(_safe_get(self.response, "debug", "trace", default=[]))

    @rx.var
    def response_json(self) -> str:
        return _safe_json(self.response)
