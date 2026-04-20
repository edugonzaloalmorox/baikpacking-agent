SHELL := /bin/bash

UV ?= uv run
PY ?= python

-include .env
export

# -------------------------
# General configuration
# -------------------------

PG_PORT ?= 5433
PG_CONTAINER ?= baikpacking-postgres
PG_USER ?= baikpacking
PG_DB ?= baikpacking
PG_BACKUP ?= backups/baikpacking.dump


API_PORT ?= 8001
REFLEX_FRONTEND_PORT ?= 3000
REFLEX_BACKEND_PORT ?= 8000
API_BASE_URL ?= http://127.0.0.1:$(API_PORT)

DATA_DIR := data
SNAP_RAW_DIR := $(DATA_DIR)/snapshots/raw
SNAP_CLEAN_DIR := $(DATA_DIR)/snapshots/clean

RAW_SNAP_GLOB := dotwatcher_bikes_raw_new_*.jsonl
CLEAN_SNAP_GLOB := dotwatcher_bikes_cleaned_new_*.json

# -------------------------
# Phony targets
# -------------------------

.PHONY: \
	help \
	check-env \
	docker-check docker-up \
	pg-up pg-check pg-reset pg-restore pg-bootstrap pg-vector-check \
	ollama-up ollama-check \
	kb-scrape kb-clean kb-load kb-embed kb-update kb-check kb-backfill kb-load-file kb-status \
	up dev \
	api ui app stop-app stop-ports

# -------------------------
# Help
# -------------------------

help:
	@echo ""
	@echo "Environment:"
	@echo "  make check-env   validate OPENAI_API_KEY formatting"
	@echo ""
	@echo "Docker / Postgres:"
	@echo "  make docker-check check Docker daemon"
	@echo "  make docker-up    start Docker Desktop if needed"
	@echo "  make pg-up        start Postgres via docker compose"
	@echo "  make pg-check     verify Postgres container and readiness"
	@echo "  make pg-reset        remove Docker Compose DB volumes"
	@echo ""
	@echo "Ollama:"
	@echo "  make ollama-up    start Ollama and pull embedding model"
	@echo "  make ollama-check verify Ollama status"
	@echo ""
	@echo "Knowledge base:"
	@echo "  make kb-scrape    run incremental scraper"
	@echo "  make kb-clean     clean latest raw snapshot"
	@echo "  make kb-load      load latest cleaned snapshot into DB"
	@echo "  make kb-embed     generate embeddings"
	@echo "  make kb-update    scrape -> clean -> load -> embed"
	@echo "  make kb-check     show latest snapshot files"
	@echo "  make kb-backfill  load full cleaned dataset"
	@echo "  make kb-load-file FILE=path/to/file.json"
	@echo "  make kb-status    inspect DB counts"
	@echo ""
	@echo "App:"
	@echo "  make api          run FastAPI backend on port $(API_PORT)"
	@echo "  make ui           run Reflex frontend"
	@echo "  make app          run FastAPI + Reflex together"
	@echo "  make stop-app     stop ports $(REFLEX_FRONTEND_PORT), $(REFLEX_BACKEND_PORT), $(API_PORT)"
	@echo ""
	@echo "Bootstrap:"
	@echo "  make up           pg-up + ollama-up + kb-update"
	@echo "  make dev          prepare infra and KB for development"
	@echo ""
	@echo "  make pg-vector-check verify pgvector extension"
	@echo "  make pg-restore      restore DB from backup"
	@echo "  make pg-bootstrap    rebuild Postgres from scratch and restore backup"

# -------------------------
# Environment checks
# -------------------------

check-env:
	@echo "Checking OPENAI_API_KEY format..."
	@if echo "$(OPENAI_API_KEY)" | grep -q "^'"; then \
		echo "❌ OPENAI_API_KEY starts with a quote"; exit 1; \
	fi
	@if echo "$(OPENAI_API_KEY)" | grep -q "'$$"; then \
		echo "❌ OPENAI_API_KEY ends with a quote"; exit 1; \
	fi
	@echo "✅ OPENAI_API_KEY format looks correct"

# -------------------------
# Docker
# -------------------------

docker-check:
	@docker info >/dev/null 2>&1 && \
		echo "✅ Docker daemon is running" || \
		echo "❌ Docker daemon is NOT running"

docker-up:
	@echo ""
	@echo "🐳 [docker] Checking Docker daemon..."
	@if docker info >/dev/null 2>&1; then \
		echo "✅ Docker daemon already running"; \
	else \
		echo "⚠️  Docker daemon not running"; \
		echo "🚀 Starting Docker Desktop..."; \
		open -a Docker; \
		echo "⏳ Waiting for Docker daemon to be ready..."; \
		until docker info >/dev/null 2>&1; do \
			sleep 2; \
			echo "  ... waiting"; \
		done; \
		echo "✅ Docker daemon is now running"; \
	fi
	@echo ""

# -------------------------
# Postgres
# -------------------------

pg-up: docker-up
	@echo ""
	@echo "🐘 [pg] Starting Postgres (docker compose up -d postgres)..."
	@docker compose up -d postgres
	@echo "⏳ [pg] Waiting for Postgres health/ready..."
	@until docker exec -i $(PG_CONTAINER) pg_isready -U $(PG_USER) -d $(PG_DB) >/dev/null 2>&1; do \
		sleep 2; \
		echo "  ... waiting"; \
	done
	@echo "✅ [pg] Postgres is ready"
	@echo ""

pg-check: docker-up
	@echo ""
	@echo "🐘 [pg] Status check (host port: $(PG_PORT))"
	@if docker ps --format '{{.Names}}' | grep -q '^$(PG_CONTAINER)$$'; then \
		echo "✅ Container running: $(PG_CONTAINER)"; \
	else \
		echo "❌ Container NOT running: $(PG_CONTAINER)"; \
		echo "👉 Run: make pg-up"; \
		exit 1; \
	fi
	@docker exec -i $(PG_CONTAINER) psql -U $(PG_USER) -d $(PG_DB) -c "SELECT extname FROM pg_extension WHERE extname = 'vector';" | grep -q vector && \
	echo "✅ pgvector extension is enabled" || \
	( echo "❌ pgvector extension is NOT enabled"; exit 1 )
	@echo ""

pg-reset: docker-up
	@echo ""
	@echo "⚠️  [pg] RESET requested: this will DELETE Docker Compose volumes for the current project."
	@docker compose down -v --remove-orphans
	@echo "✅ [pg] Reset complete"
	@echo "👉 Run: make pg-up"
	@echo ""

pg-vector-check: pg-up
	@echo ""
	@echo "🧩 [pg] Checking pgvector extension..."
	@docker exec -i $(PG_CONTAINER) psql -U $(PG_USER) -d $(PG_DB) -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
	@echo ""

pg-restore: pg-up
	@echo ""
	@echo "📦 [pg] Restoring database from $(PG_BACKUP)..."
	@if [[ ! -f "$(PG_BACKUP)" ]]; then \
		echo "❌ Backup file not found: $(PG_BACKUP)"; \
		exit 1; \
	fi
	@docker exec -i $(PG_CONTAINER) pg_restore -U $(PG_USER) -d $(PG_DB) /backups/$$(basename $(PG_BACKUP))
	@echo "✅ [pg] Restore complete"
	@echo ""


pg-bootstrap: docker-up
	@echo ""
	@echo "🚧 [pg] Rebuilding Postgres from scratch and restoring backup..."
	@if [[ ! -f "$(PG_BACKUP)" ]]; then \
		echo "❌ Backup file not found: $(PG_BACKUP)"; \
		exit 1; \
	fi
	@docker compose down -v --remove-orphans
	@docker compose up -d postgres
	@echo "⏳ [pg] Waiting for Postgres health/ready..."
	@until docker exec -i $(PG_CONTAINER) pg_isready -U $(PG_USER) -d $(PG_DB) >/dev/null 2>&1; do \
		sleep 2; \
		echo "  ... waiting"; \
	done
	@echo "🧩 [pg] Verifying pgvector extension..."
	@docker exec -i $(PG_CONTAINER) psql -U $(PG_USER) -d $(PG_DB) -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
	@echo "📦 [pg] Restoring backup..."
	@docker exec -i $(PG_CONTAINER) pg_restore -U $(PG_USER) -d $(PG_DB) /backups/$$(basename "$(PG_BACKUP)")
	@echo "✅ [pg] Bootstrap complete"
	@echo ""

# -------------------------
# Ollama
# -------------------------

ollama-up:
	@echo ""
	@echo "🔌 [ollama] Checking Ollama availability..."
	@if ! command -v ollama >/dev/null 2>&1; then \
		echo "❌ Ollama is not installed or not in PATH."; \
		echo "👉 Install from https://ollama.com"; \
		exit 1; \
	fi
	@echo "✅ Ollama binary found"

	@echo ""
	@echo "🚀 [ollama] Starting Ollama server (ollama serve)..."
	@if pgrep -f "ollama serve" >/dev/null 2>&1; then \
		echo "✅ Ollama server already running"; \
	else \
		echo "➡️  Ollama not running, starting server..."; \
		ollama serve >/tmp/ollama.log 2>&1 & \
		sleep 2; \
	fi

	@echo ""
	@echo "🔍 [ollama] Verifying server responds on http://localhost:11434 ..."
	@if command -v curl >/dev/null 2>&1; then \
		if ! curl -sf http://localhost:11434 >/dev/null 2>&1; then \
			echo "❌ Ollama server did not respond on localhost:11434"; \
			echo "👉 Check logs: /tmp/ollama.log"; \
			exit 1; \
		fi; \
	else \
		echo "⚠️ curl not found; skipping HTTP check (install curl for stronger checks)"; \
	fi
	@echo "✅ Ollama server is responding"

	@echo ""
	@echo "📦 [ollama] Pulling embedding model: $$EMB_EMBEDDING_MODEL"
	@if [[ -z "$$EMB_EMBEDDING_MODEL" ]]; then \
		echo "❌ EMB_EMBEDDING_MODEL is not set in environment"; \
		echo "👉 export EMB_EMBEDDING_MODEL=mxbai-embed-large:335m"; \
		exit 1; \
	fi
	@ollama pull $$EMB_EMBEDDING_MODEL
	@echo "✅ Embedding model ready: $$EMB_EMBEDDING_MODEL"
	@echo ""

ollama-check:
	@echo ""
	@echo "🔍 [ollama] Status check"
	@if ! command -v ollama >/dev/null 2>&1; then \
		echo "❌ Ollama is not installed or not in PATH."; \
		exit 1; \
	fi
	@pgrep -f "ollama serve" >/dev/null 2>&1 && \
		echo "✅ Ollama server process is running" || \
		echo "❌ Ollama server process is NOT running"

	@if command -v curl >/dev/null 2>&1; then \
		curl -sf http://localhost:11434 >/dev/null 2>&1 && \
			echo "✅ Ollama responds on localhost:11434" || \
			echo "❌ Ollama does NOT respond on localhost:11434"; \
	fi

	@echo ""
	@echo "📦 Available models:"
	@ollama list || true
	@echo ""

# -------------------------
# Knowledge base
# -------------------------

kb-scrape:
	$(UV) $(PY) -m baikpacking.pipelines.scrape_dotwatcher

kb-clean:
	$(UV) $(PY) -m baikpacking.scraper.clean_json --update-latest

kb-load:
	@CLEAN_LATEST="$$(ls -1t $(SNAP_CLEAN_DIR)/$(CLEAN_SNAP_GLOB) 2>/dev/null | head -n 1)"; \
	if [[ -z "$$CLEAN_LATEST" ]]; then \
		echo "No clean new-only snapshots found in $(SNAP_CLEAN_DIR)."; \
		exit 1; \
	fi; \
	echo "Latest clean snapshot: $$CLEAN_LATEST"; \
	$(UV) $(PY) -m baikpacking.db.data_loader --input "$$CLEAN_LATEST"

kb-embed:
	$(UV) $(PY) -m baikpacking.pipelines.embed_index

kb-update:
	@$(MAKE) kb-scrape
	@NEW_MARKER="$(SNAP_RAW_DIR)/.last_scrape_new"; \
	if [[ ! -f "$$NEW_MARKER" ]]; then \
		echo "No new raw snapshot created by scrape. Skipping clean/load/embed."; \
		exit 0; \
	fi; \
	RAW_LATEST="$$(tail -n 1 "$$NEW_MARKER")"; \
	echo "New raw snapshot created: $$RAW_LATEST"; \
	$(MAKE) kb-clean; \
	CLEAN_LATEST="$$(ls -1t $(SNAP_CLEAN_DIR)/$(CLEAN_SNAP_GLOB) 2>/dev/null | head -n 1)"; \
	if [[ -z "$$CLEAN_LATEST" ]]; then \
		echo "No clean new-only snapshots found in $(SNAP_CLEAN_DIR). Skipping DB load/embed."; \
		rm -f "$$NEW_MARKER"; \
		exit 0; \
	fi; \
	echo "Latest clean snapshot: $$CLEAN_LATEST"; \
	$(UV) $(PY) -m baikpacking.db.data_loader --input "$$CLEAN_LATEST"; \
	$(MAKE) kb-embed; \
	rm -f "$$NEW_MARKER"

kb-check:
	@echo ""
	@echo "Latest raw new-only snapshot:"
	@ls -1t $(SNAP_RAW_DIR)/$(RAW_SNAP_GLOB) 2>/dev/null | head -n 3 || true
	@echo ""
	@echo "Latest clean new-only snapshot:"
	@ls -1t $(SNAP_CLEAN_DIR)/$(CLEAN_SNAP_GLOB) 2>/dev/null | head -n 3 || true
	@echo ""

kb-backfill:
	$(UV) $(PY) -m baikpacking.db.data_loader --input data/dotwatcher_bikes_cleaned.json

kb-load-file:
	@if [[ -z "$(FILE)" ]]; then echo "Usage: make kb-load-file FILE=path/to/snapshot.json"; exit 1; fi
	$(UV) $(PY) -m baikpacking.db.data_loader --input "$(FILE)"

kb-status: pg-up
	@echo ""
	@echo "📊 [status] Checking database state..."
	@docker exec -i $(PG_CONTAINER) psql -U $(PG_USER) -d $(PG_DB) -c "SELECT COUNT(*) AS total_articles FROM articles;" || true
	@docker exec -i $(PG_CONTAINER) psql -U $(PG_USER) -d $(PG_DB) -c "SELECT COUNT(*) AS total_riders FROM riders;" || true
	@docker exec -i $(PG_CONTAINER) psql -U $(PG_USER) -d $(PG_DB) -c "SELECT COUNT(*) AS total_chunks FROM rider_chunks;" || true
	@docker exec -i $(PG_CONTAINER) psql -U $(PG_USER) -d $(PG_DB) -c "SELECT COUNT(*) AS chunks_with_embeddings FROM rider_chunks WHERE embedding IS NOT NULL;" || true
	@echo ""

# -------------------------
# Bootstrap
# -------------------------

up: pg-up ollama-up kb-update

dev:
	@echo ""
	@echo "🚀 Starting development environment..."
	@$(MAKE) pg-up
	@$(MAKE) ollama-up
	@$(MAKE) kb-update
	@$(MAKE) kb-status
	@echo ""
	@echo "✅ Dev environment ready."
	@echo ""

# -------------------------
# App
# -------------------------

api:
	@echo ""
	@echo "⚡ [api] Starting FastAPI on port $(API_PORT)..."
	@$(UV) uvicorn src.baikpacking.api.main:app --reload --port $(API_PORT)

ui: check-env
	@echo ""
	@echo "🎨 [ui] Starting Reflex UI..."
	@echo "    Frontend: http://localhost:$(REFLEX_FRONTEND_PORT)"
	@echo "    Reflex backend: http://localhost:$(REFLEX_BACKEND_PORT)"
	@echo "    API_BASE_URL=$(API_BASE_URL)"
	@cd apps/reflex_ui && API_BASE_URL=$(API_BASE_URL) $(UV) reflex run

app: check-env
	@echo ""
	@echo "🚀 Starting FastAPI + Reflex UI..."
	@echo "    FastAPI:        http://127.0.0.1:$(API_PORT)"
	@echo "    Reflex UI:      http://localhost:$(REFLEX_FRONTEND_PORT)"
	@echo "    Reflex backend: http://localhost:$(REFLEX_BACKEND_PORT)"
	@trap 'kill 0' INT TERM EXIT; \
	($(UV) uvicorn src.baikpacking.api.main:app --reload --port $(API_PORT)) & \
	(cd apps/reflex_ui && API_BASE_URL=$(API_BASE_URL) $(UV) reflex run) & \
	wait

stop-app:
	@echo ""
	@echo "🛑 Stopping app ports..."
	-@kill -9 $$(lsof -ti:$(REFLEX_FRONTEND_PORT)) 2>/dev/null || true
	-@kill -9 $$(lsof -ti:$(REFLEX_BACKEND_PORT)) 2>/dev/null || true
	-@kill -9 $$(lsof -ti:$(API_PORT)) 2>/dev/null || true
	@echo "✅ Stopped ports $(REFLEX_FRONTEND_PORT), $(REFLEX_BACKEND_PORT), $(API_PORT)"
	@echo ""

stop-ports: stop-app




