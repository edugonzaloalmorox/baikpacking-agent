# 🚵‍♂️ bAIpacking Agent

Capstone project for the [AI Engineering Buildcamp: From RAG to Agents](https://maven.com/alexey-grigorev/from-rag-to-agents)

## The problem

Bikepacking setup advice is scattered across race articles, rider interviews, and equipment lists. A reviewer or rider usually wants a fast answer to questions like:

- What bike setup worked for this event?
- What tyres, drivetrain, or bags fit this terrain?
- What do similar riders actually use when exact event evidence is limited?

This system turns that unstructured material into a searchable knowledge base and a grounded recommendation pipeline.


![alt text](docs/front_page.png)

## Architecture

bAIpacking is now partially dockerized:

- `postgres` container
  - stores the knowledge base and embeddings in PostgreSQL with `pgvector`
  - is initialized with `docker/init/01-enable-pgvector.sql`
  - is restored from `backups/baikpacking.dump` during bootstrap
- `ollama` container
  - serves embedding inference
  - persists pulled models in a Docker volume mounted at `/root/.ollama`
- `api` container
  - serves the FastAPI application
  - talks to Postgres at `postgres:5432`
  - talks to Ollama at `http://ollama:11434`
- Reflex UI
  - runs locally for now
  - talks to the API on `http://localhost:8001`

Exposed host ports:

- `localhost:5433` for Postgres
- `localhost:11434` for Ollama
- `localhost:8001` for the API
- `localhost:3000` for the Reflex frontend
- `localhost:8000` for the Reflex backend

The serving path ends at `Recommendation` and returns through the API. `Data Eval`, `human feedback`, and `LLM Judge` are evaluation artifacts, not part of the live request path.

![alt text](docs/architecture.png)

## Knowledge Base And Retrieval

### Knowledge Base

The active knowledge base is stored in PostgreSQL and built from [DotWatcher bikepacking set ups](https://dotwatcher.cc/features) and related event contexts

![alt text](docs/bike.png)

Relevant pieces include:

- cleaned rider records
- article metadata
- rider chunks
- vector embeddings for retrieval

The main runtime retrieval path uses:

- `riders`
- `rider_chunks`
- `articles`
-  `similarity search` between bikes and key items

### Retrieval Strategy

Retrieval is event-aware and tries to stay honest about grounding strength:

- `exact_event`
  - preferred when the resolved event has direct KB coverage
  - exact article titles and exact-scoped riders are used first
- `similar_event`
  - used when exact grounding is thin or unavailable but similar event families are reliable
- `unknown_global`
  - used when neither exact nor similar grounding is reliable enough

The recommender also tracks policy modes to set the "tone" of the recommendation and prevent hallucinations.

- `strict_grounded`
- `pattern_based`
- `generic_fallback`

Those policy modes control how much event-specific language and specificity the writer may use.


## Agents And LLM

The recommender is centered on `src/baikpacking/agents/recommender_agent.py` and the supporting modules under `src/baikpacking/agents/`.

The online request path is:

1. `POST /recommend` or the local CLI entrypoint
2. event resolution
3. query intent classification
4. guardrail check
5. event web context lookup
6. retrieval planning
7. similar-rider search in PostgreSQL / pgvector
8. evidence summarization
9. policy selection
10. optional review-hint lookup
11. writer draft generation
12. validation, optional repair, and postprocessing

The architecture is intentionally small:

- `event_resolution.py` resolves aliases and canonical event names
- `event_context_resolution.py` fetches event-level context from the web and derives event-family hints
- `retrieval_planning.py` builds the exact-vs-similar retrieval query plan
- `tools/riders.py` runs the actual rider retrieval against Postgres / pgvector
- `evidence_summary.py` summarizes evidence strength
- `policy.py` selects `strict_grounded`, `pattern_based`, or `generic_fallback`
- `review_feedback.py` loads optional scenario-review hints from `data/eval/scenario_reviews.jsonl`
- `writer_input.py` defines the writer-facing input contract
- `models.py` defines `WriterRecommendationDraft` and the final `SetupRecommendation`
- `postprocess.py` and `output_validation.py` assemble and validate the final output

The writer stage does not own the full final recommendation object. It produces a compact draft, and code assembles the final `SetupRecommendation`.


### Runtime And Eval Separation

The live API writes append-only telemetry to `data/eval/live_runs.jsonl` on every request, and the `/feedback` endpoint appends user feedback to `data/eval/live_feedback.jsonl`.

That data is then consumed by the offline judge in `src/baikpacking/eval/output_judge.py`, which reads live runs and feedback and writes judgments to `data/eval/output_judgments.jsonl`.

So the judge and feedback files are evaluation artifacts, not part of the serving-time recommendation loop.

## Repository Structure

Important directories and entrypoints:

- `src/baikpacking/agents/`
  - runtime orchestration, policy, postprocessing, validation, review feedback, and writer contracts
- `src/baikpacking/tools/`
  - rider retrieval, event context helpers, tracing, and pgvector search
- `src/baikpacking/pipelines/`
  - KB scraping, cleaning, loading, and embedding workflows
- `src/baikpacking/scraper/`
  - DotWatcher scraping and cleaning utilities
- `src/baikpacking/db/`
  - schema and database helpers
- `src/baikpacking/embedding/`
  - embedding configuration and Ollama client
- `src/baikpacking/apps/eval_dashboard.py`
  - local Streamlit eval dashboard
- `src/baikpacking/scripts/run_recommender.py`
  - runtime smoke path and sample eval row writer
- `src/baikpacking/scripts/run_eval_scenarios.py`
  - manual scenario evaluation runner
- `tests/`
  - unit, regression, writer, policy, retrieval, and scenario-eval tests
- `data/eval/`
  - scenario input/output artifacts, cached context, and human review JSONL files
- `archive/`
  - legacy code and older evaluation artifacts kept for reference

## Local Setup

### Prerequisites

- Docker Desktop or another Docker Engine that can run Docker Compose
- `uv`
- a local `backups/baikpacking.dump` file

If Docker is not running, `make docker-check` will fail and `make docker-up` can start Docker Desktop on macOS.

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment

Create or edit `.env` with local values. Common variables include:

- `DATABASE_URL`
- `EMB_EMBEDDING_MODEL`
- `EMB_OLLAMA_BASE_URL`
- `OLLAMA_HOST`
- `LOGFIRE_TOKEN`
- `OPENAI_API_KEY`

For local host commands, use values like:

- `DATABASE_URL=postgresql://baikpacking:baikpacking@localhost:5433/baikpacking`
- `EMB_EMBEDDING_MODEL=mxbai-embed-large:335m`
- `EMB_OLLAMA_BASE_URL=http://localhost:11434`

Inside Docker, the API container gets the service-network values from `docker-compose.yml`:

- `DATABASE_URL=postgresql://baikpacking:baikpacking@postgres:5432/baikpacking`
- `EMB_OLLAMA_BASE_URL=http://ollama:11434`
- `EMB_EMBEDDING_MODEL=mxbai-embed-large:335m`

### 3. Put The Postgres Backup In Place

The backup file should exist at `backups/baikpacking.dump`.

- the `backups/` directory is tracked with `backups/.gitkeep`
- the dump file itself does not need to be committed
- `make pg-bootstrap` and `make pg-restore` both expect the dump to be present locally

### 4. Bootstrap Postgres

For an existing Postgres volume, start the container with:

```bash
make pg-up
```

For a fresh machine or a full reset, use:

```bash
make pg-bootstrap
```

`make pg-bootstrap` removes the current Docker Compose volumes for the project, recreates Postgres, waits for readiness, verifies `pgvector`, and restores the database backup from `backups/baikpacking.dump`.

`pgvector` is enabled automatically during first initialization by the SQL script mounted at `/docker-entrypoint-initdb.d/01-enable-pgvector.sql`.

Useful Postgres targets:

- `make pg-up` starts the existing Postgres container and waits for readiness
- `make pg-check` checks the running container, `pg_isready`, and `pgvector`
- `make pg-reset` removes the Docker Compose DB volumes
- `make pg-restore` restores the local backup into an already running database
- `make pg-vector-check` prints the `vector` extension state

### 5. Bootstrap Ollama

```bash
make ollama-up
make ollama-pull
```

`make ollama-up` starts the Dockerized Ollama service and keeps models in the persistent `/root/.ollama` volume.

`make ollama-pull` pulls `$(EMB_EMBEDDING_MODEL)` into that container. Set `EMB_EMBEDDING_MODEL` in `.env` before running it.

### 6. Start The API In Docker

The API is defined in `docker-compose.yml` and runs in Docker.

This checkout does not define a `make api-up` helper, so use Docker Compose directly for the containerized API.

```bash
docker compose up -d api
```

Use `docker compose ps api` and `docker compose logs -f api` to inspect it. The API container uses the Docker-network values for Postgres and Ollama listed above.

### 7. Run The Reflex UI Locally

```bash
make ui
```

This runs Reflex locally and points it at `http://localhost:8001`.

### Recommended Bootstrap Path

For a fresh machine, use this order:

1. configure `.env`
2. make sure `backups/baikpacking.dump` exists locally
3. run `make pg-bootstrap`
4. run `make ollama-up`
5. run `make ollama-pull`
6. run `docker compose up -d api`
7. run `make kb-update` if you need to rebuild the knowledge base
8. run `make ui`

`make up` is a shorter infra bootstrap: it starts Postgres, starts Ollama, pulls the model, and refreshes the KB. `make dev` does the same and then prints the DB status.

## Run The KB Pipeline

The active KB flow is:

```text
scrape -> clean -> load -> embed
```

Run the full pipeline:

```bash
make kb-update
```


## Run The Recommender

Run the current runtime entrypoint:

```bash
uv run python -m baikpacking.scripts.run_recommender
```

This prints:

- the recommendation
- the retrieved grounding riders
- the tool trace

It also appends a small runtime eval row to `data/eval/sample_eval_rows.jsonl`.

## API

The FastAPI service can be run either locally or in Docker.

Run it locally:

```bash
make api
```

Run the Dockerized service:

```bash
docker compose up -d api
```

Quick curl examples:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/ready
curl -X POST http://localhost:8001/recommend \
  -H 'Content-Type: application/json' \
  -d '{"query":"What tyres should I use for Atlas Mountain Race?","include_debug":true}'
```

The `POST /recommend` response includes the query, resolved event, intent,
recommendation, evidence, policy, and optional debug trace data.
`GET /health` is a liveness check; `GET /ready` verifies database connectivity.
Each live `/recommend` request also appends a deterministic live-eval row to
`data/eval/live_runs.jsonl`.

## Bootstrap Targets

The main Makefile targets are:

- `make docker-check` checks that Docker is available
- `make docker-up` starts Docker Desktop if needed
- `make pg-up` starts an existing Postgres container
- `make pg-bootstrap` recreates Postgres from scratch and restores the backup
- `make ollama-up` starts Dockerized Ollama with persistent model storage
- `make ollama-pull` pulls the embedding model into Ollama
- `make api` runs the FastAPI app locally on `localhost:8001`
- `make ui` runs the Reflex UI locally
- `make up` boots Postgres, Ollama, the model, and the KB refresh
- `make dev` runs `make up` and then prints database status
- `make app` runs the local API and Reflex UI together
- `make stop-app` stops the local API/UI ports

## Reflex UI

A local Reflex frontend lives in `apps/reflex_ui/`.

Run it with the backend API running locally:

```bash
cd apps/reflex_ui
API_BASE_URL=http://127.0.0.1:8000 uv run reflex run
```

The UI renders the `/recommend` response as a chat-first assistant with
collapsed technical details. See `apps/reflex_ui/README.md` for a shorter
setup note.

## Testing

The test suite lives under `tests/`.

Run everything:

```bash
uv run pytest
```

Useful focused subsets:

```bash
uv run pytest tests/test_event_resolution.py tests/test_event_context_resolution.py tests/test_retrieval_planning.py
uv run pytest tests/test_policy.py tests/test_writer_stage.py tests/test_agents_writer_input.py
uv run pytest tests/test_eval_scenario_content_assertions.py tests/test_review_feedback.py
uv run pytest tests/test_riders.py tests/test_recommender.py
```

What these tests cover:

- event resolution and alias handling
- event context and retrieval planning
- rider retrieval and evidence summarization
- policy selection and fallback behavior
- writer-stage contract stability
- manual scenario assertions
- human review storage and matching

## Evaluation

Evaluation is deterministic first, with optional human review on top.

### Manual Scenario Evaluation

The manual scenario contract lives in:

- `data/eval/manual_scenarios.yaml`

Run the scenario suite with:

```bash
uv run python -m baikpacking.scripts.run_eval_scenarios
```

That runner writes per-scenario results to:

- `data/eval/scenario_runs.jsonl`

Each run records:

- scenario contract fields
- resolved event and intent
- retrieval source and policy mode
- content assertion results
- event alignment assertion results
- failure kind and runtime/schema stability

The current deterministic checks cover:

- content assertions
- event alignment
- schema/runtime stability
- failure classification such as `output_schema_failure`


### Eval Dashboard

Open the local eval dashboard with:

```bash
uv run streamlit run src/baikpacking/apps/eval_dashboard.py
```

The dashboard reads:

- `data/eval/scenario_runs.jsonl`

It shows:

- summary cards
- failure and issue breakdowns
- drill-down rows
- selected-row details

### Human Review Loop

The dashboard also supports reviewer feedback.

Saved reviews are stored separately in:

- `data/eval/scenario_reviews.jsonl`

Each review can include:

- `review_status`
- `human_label`
- `corrected_event`
- `corrected_component`
- `corrected_policy_mode`
- `review_notes`

The recommender loads matching review hints at runtime for the same event/component combination and uses them as prompt context. Reviews are not treated as hard labels or a learned model.

There is a LLM judge that evaluates the quality of the answer offline. It uses data/eval/live_runs.jsonl

## Monitoring And Tracing

The project has developer-facing tracing, not a full production monitoring stack.

Available instrumentation:

- `src/baikpacking/logging_config.py` configures standard logging and Logfire
- `pydantic_ai` instrumentation captures agent runs and tool calls
- `src/baikpacking/tools/call_trace.py` records tool-call traces in the runtime path
- `src/baikpacking/scripts/run_recommender.py` prints the tool trace and stores a sample runtime eval row
- `make kb-status` gives a quick DB health check by counting riders and embeddings

Typical inspection flow:

```bash
uv run python -m baikpacking.scripts.run_recommender
make kb-status
```

If `LOGFIRE_TOKEN` is configured, inspect traces in Logfire. Otherwise tracing remains local.


## Want to review the system? 

- Run the recommender: `uv run python -m baikpacking.scripts.run_recommender`
- Run eval: `uv run python -m baikpacking.scripts.run_eval_scenarios`
- Open the dashboard: `uv run streamlit run src/baikpacking/apps/eval_dashboard.py`
- Inspect eval output: `data/eval/scenario_runs.jsonl`
- Inspect reviewer feedback: `data/eval/scenario_reviews.jsonl`
- Inspect runtime smoke logs: `data/eval/sample_eval_rows.jsonl`

Any questions or comments please reach out: eduardogonzaloalmorox@gmail.com
