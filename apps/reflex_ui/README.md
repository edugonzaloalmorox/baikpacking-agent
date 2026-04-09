# Reflex UI

This is the local Reflex frontend for `bikepacking-agent`.

## Install

From the repository root:

```bash
uv sync
```

## Run the backend API

In one terminal:

```bash
uv run uvicorn baikpacking.api.main:app --host 0.0.0.0 --port 8000
```

## Run the Reflex UI

In a second terminal:

```bash
cd apps/reflex_ui
API_BASE_URL=http://127.0.0.1:8000 uv run reflex run
```

If your backend runs on another host or port, point `API_BASE_URL` there instead.
The Reflex backend for the UI is configured to run on port `8001` so it does
not conflict with the FastAPI recommender on port `8000`.

## Notes

- The UI calls `POST /recommend` on the backend.
- The UI requests debug traces by default and renders them in a collapsible panel.
- If the backend is unavailable, the UI shows a clean error banner instead of crashing.
