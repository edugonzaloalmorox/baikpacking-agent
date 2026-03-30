# Active vs Archived

## Active Paths

The current production path is:

- KB build pipeline: `src/baikpacking/scraper/clean_json.py` -> `src/baikpacking/pipelines/scrape_dotwatcher.py` -> `src/baikpacking/db/data_loader.py` -> `src/baikpacking/pipelines/embed_index.py` -> `src/baikpacking/embedding/embed.py` -> `src/baikpacking/tools/riders.py`
- Runtime recommender: `src/baikpacking/agents/recommender_agent.py` plus its supporting modules in `src/baikpacking/agents/`
- Runtime entrypoint: `src/baikpacking/scripts/run_recommender.py`
- Active data: `data/dotwatcher_bikes_cleaned.json`, `data/eval/event_context_cache.jsonl`, and the current `data/snapshots/{raw,clean}/` files
- Active tests: the recommender, parsing, validation, policy, evidence, retrieval-planning, and event-context tests under `tests/`

## Archived

Archived under `archive/`:

- Legacy eval code: `archive/legacy_eval/`
- Legacy eval / debug scripts: `archive/legacy_scripts/`
- Legacy agent: `archive/legacy_agents/response_judge_agent.py`
- Legacy retrieval backend: `archive/legacy_retrieval/`
- Legacy tests for old embed/search paths: `archive/legacy_tests/`
- Legacy notebooks: `archive/legacy_notebooks/`
- Old reports and outputs: `archive/legacy_outputs/`
- Old raw/eval datasets and snapshots: `archive/legacy_data/`
- Old package layout stub: `archive/legacy_package_layout/`
- Old docs: `archive/docs_old/`
- Build artifact metadata: `archive/build_artifacts/`
- Dead domain package stub: `archive/legacy_domain/`
- Legacy utility stubs: `archive/legacy_utils/`

## Left In Place For Review

These remain in the active tree because they are generated artifacts or lower-confidence clutter, not source code:

- `src/baikpacking/__pycache__/`
- `src/baikpacking/agents/__pycache__/`
- `src/baikpacking/db/__pycache__/`
- `src/baikpacking/embedding/__pycache__/`
- `src/baikpacking/pipelines/__pycache__/`
- `src/baikpacking/scraper/__pycache__/`
- `src/baikpacking/scripts/__pycache__/`
- `src/baikpacking/tools/__pycache__/`

## Notes

- No runtime imports were rewritten as part of this archival pass.
- The active KB and recommender code paths remain in place and should continue to import cleanly.
