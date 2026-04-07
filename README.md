# bAIpacking Agent

Bikepacking recommendation system built around a real end-to-end knowledge base pipeline, grounded retrieval, and LLM-assisted recommendation generation.

## Problem

Bikepacking race setup advice is scattered across article pages, rider interviews, and equipment lists. A reviewer or rider usually wants answers like:

- What bike and tire setup worked for a similar event?
- What drivetrain or wheel choice is common for this terrain?
- Which riders used a setup close to my event or constraints?

This project turns that unstructured material into a structured database so it can be searched, filtered, embedded, and used for grounded recommendations.

## What The System Does

The project builds a searchable bikepacking knowledge base from DotWatcher [_“Bikes of …”_](https://dotwatcher.cc/features/bikes-of?) articles, then uses that knowledge base at runtime to produce recommendations for new queries.

For now, the system is best understood as a structured pipeline with LLM-assisted generation, not as a fully autonomous agent:

- deterministic ingestion and cleaning
- database loading and embedding
- retrieval over structured and vector-backed data
- policy-based fallback when evidence is weak
- one LLM writer stage that turns retrieved evidence into a final recommendation

The result is a grounded recommender that is more transparent than a pure chat app and more flexible than a fixed rules engine.

## Why This Project Is Interesting

- It has a real end-to-end KB build path, not just a demo prompt.
- Retrieval is event-aware and falls back safely when exact evidence is missing.
- The runtime path is modular, traceable, and testable.
- Evaluation exists for retrieval and there is an offline response-eval path, even though the latter is still less mature.

## Active Architecture

### KB Build Pipeline

The active knowledge-base path is:

```text
src/baikpacking/scraper/clean_json.py
-> src/baikpacking/pipelines/scrape_dotwatcher.py
-> src/baikpacking/db/data_loader.py
-> src/baikpacking/pipelines/embed_index.py
-> src/baikpacking/embedding/embed.py
-> src/baikpacking/tools/riders.py
```

What each stage does:

- `src/baikpacking/scraper/clean_json.py`: cleans and normalizes DotWatcher article text into rider records
- `src/baikpacking/pipelines/scrape_dotwatcher.py`: scrapes new DotWatcher “Bikes of …” pages and writes snapshots
- `src/baikpacking/db/data_loader.py`: loads cleaned rider data into PostgreSQL
- `src/baikpacking/pipelines/embed_index.py`: embeds riders into the vector index
- `src/baikpacking/embedding/embed.py`: talks to Ollama and builds embedding vectors
- `src/baikpacking/tools/riders.py`: runtime retrieval over riders and evidence chunks

### Runtime Recommender

The current runtime recommender is centered on:

- `src/baikpacking/agents/recommender_agent.py`
- supporting modules under `src/baikpacking/agents/`
- runtime entrypoint `src/baikpacking/scripts/run_recommender.py`

The recommender is a staged system. It performs:

1. event resolution
2. query intent classification
3. event context resolution
4. retrieval planning
5. pgvector / structured rider search
6. evidence summarization
7. policy selection and fallback
8. postprocessing and output validation
9. final LLM-assisted response writing

The tools are not an open-ended autonomous loop. They are modular stages with clear responsibilities and trace logging.

### Supporting Modules

The runtime path uses these modules heavily:

- `src/baikpacking/agents/event_resolution.py`
- `src/baikpacking/agents/event_context_resolution.py`
- `src/baikpacking/agents/query_intent.py`
- `src/baikpacking/agents/retrieval_planning.py`
- `src/baikpacking/agents/evidence_summary.py`
- `src/baikpacking/agents/policy.py`
- `src/baikpacking/agents/postprocess.py`
- `src/baikpacking/agents/output_validation.py`
- `src/baikpacking/tools/pg_vector_search.py`
- `src/baikpacking/tools/riders.py`

## Tools And Components

- `PostgreSQL + pgvector`: stores structured riders and vector embeddings
- `Ollama`: provides the embedding model used during KB indexing
- `Pydantic AI`: wraps the writer step and instrumentation
- `Logfire`: captures tracing when configured
- `Docker Compose`: runs local Postgres
- `uv`: reproducible Python dependency management and execution

## Repository Structure

- `src/baikpacking/agents/`: runtime orchestration, policy, validation, and writer logic
- `src/baikpacking/pipelines/`: KB ingestion and embedding pipelines
- `src/baikpacking/scraper/`: scraping and cleaning of DotWatcher data
- `src/baikpacking/db/`: schema, loaders, and DB connections
- `src/baikpacking/embedding/`: embedding config and Ollama client
- `src/baikpacking/tools/`: retrieval, tracing, and event-context utilities
- `src/baikpacking/scripts/`: runnable entrypoints
- `tests/`: unit and smoke tests for the active code path
- `data/`: current snapshots, cached eval artifacts, and generated outputs
- `archive/`: legacy retrieval, eval, notebooks, and older code paths kept for reference

## Setup

### 1. Install Dependencies

Use the lockfile-based workflow:

```bash
uv sync
```

### 2. Configure Environment

Create or edit `.env` with your local values. The code reads these common variables:

- `DATABASE_URL`
- `EMB_EMBEDDING_MODEL`
- `OLLAMA_HOST`
- `LOG_LEVEL`
- `LOGFIRE_TOKEN`
- `AGENT_WRITER_MODEL`
- `EVENT_CONTEXT_CACHE_PATH`
- `EVENT_CONTEXT_CACHE_TTL_S`

Typical local defaults are already present in code, but the project still expects a usable local database and embedding runtime.

### 3. Start Local Services

```bash
make pg-up
make ollama-up
```

If `EMB_EMBEDDING_MODEL` is not already set, export it before starting Ollama:

```bash
export EMB_EMBEDDING_MODEL=mxbai-embed-large:335m
```

## How To Run The KB Pipeline

The shortest path is:

```bash
make kb-update
```

That command runs the active incremental KB flow:

```text
scrape -> clean -> load -> embed
```

Useful subcommands:

```bash
make kb-scrape   # scrape new DotWatcher articles
make kb-clean    # clean the latest raw snapshot
make kb-load     # load the latest cleaned snapshot into Postgres
make kb-embed    # embed riders into pgvector
make kb-check    # show latest snapshot files
make kb-status   # show row counts and missing embeddings
```

For a one-shot local bootstrap:

```bash
make dev
```

## How To Run The Recommender

Run the current runtime entrypoint with:

```bash
uv run python -m baikpacking.scripts.run_recommender
```

This:

- runs the recommender on a sample query
- prints the recommendation and tool trace
- appends a JSONL row to `data/eval/sample_eval_rows.jsonl`

The recommender implementation lives in:

- `src/baikpacking/agents/recommender_agent.py`
- `src/baikpacking/scripts/run_recommender.py`

## Testing

The active test suite is under `tests/`.

Run all tests:

```bash
uv run pytest
```

Run a focused smoke test for the runtime recommender:

```bash
uv run pytest tests/test_recommender.py -q
```

The tests cover event resolution, query intent, retrieval planning, policy behavior, evidence summarization, output validation, and a recommender smoke path.

## Evaluation

Evaluation is split into two levels.

### Current Active Evaluation

The current runtime evaluation artifact is the sample JSONL row written by:

```bash
uv run python -m baikpacking.scripts.run_recommender
```

That file captures:

- the user question
- the model answer
- the structured output
- the tool trace

This is useful for offline inspection and response-quality review, but it is still a lightweight evaluation path.

The manual scenario runner writes a richer per-run record to `data/eval/scenario_runs.jsonl` from:

```bash
uv run python -m baikpacking.scripts.run_eval_scenarios
```

Those rows now include the scenario contract fields, runtime policy/retrieval metadata, content assertions, event-alignment assertions, and failure classification such as `output_schema_failure`.

For a quick local view of that file, launch the eval dashboard:

```bash
streamlit run src/baikpacking/apps/eval_dashboard.py
```

### Retrieval Evaluation

To DO

## Monitoring And Tracing

The project has developer-facing tracing, not a full production monitoring stack.

What is available:

- `src/baikpacking/logging_config.py` configures standard logging and Logfire
- `LOGFIRE_TOKEN` enables sending traces to Logfire
- `pydantic_ai` instrumentation captures agent runs and tool calls
- `src/baikpacking/tools/call_trace.py` records tool-call traces inside the runtime path
- `src/baikpacking/scripts/run_recommender.py` prints the tool trace and stores it in JSONL
- `make kb-status` gives a quick DB health check by counting riders and embeddings

Typical inspection flow:

```bash
uv run python -m baikpacking.scripts.run_recommender
make kb-status
```

If `LOGFIRE_TOKEN` is configured, inspect traces in the Logfire project linked to that token. If it is not configured, tracing remains local.

## Reproducibility

To reproduce the current local setup:

1. Install dependencies with `uv sync`
2. Start Postgres with `make pg-up`
3. Start Ollama with `make ollama-up`
4. Set the needed environment variables in `.env`
5. Build or refresh the KB with `make kb-update`
6. Run the recommender with `uv run python -m baikpacking.scripts.run_recommender`
7. Run tests with `uv run pytest`

This is intentionally a local, reproducible workflow rather than a cloud-deployed system.

## Limitations And Future Improvements

- The system is not fully agentic; it is a structured pipeline with an LLM writer stage.
- Final answers are grounded, but source-level citation support is still limited.
- Response evaluation exists, but it is less mature than retrieval evaluation.
- Runtime and eval retrieval backends are not fully unified.
- Monitoring is present through logs and Logfire, but the documentation and production controls are still thin.

## Notes For Reviewers

- The active repo is centered on the cleaned KB pipeline and the runtime recommender path for now.
