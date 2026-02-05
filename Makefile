SHELL := /bin/bash

UV ?= uv run
PY ?= python

-include .env
export


PG_PORT ?= 5433
PG_CONTAINER ?= baikpacking-postgres
PG_USER ?= baikpacking
PG_DB ?= baikpacking
PG_VOLUME ?= pg_data

# -------------------------
# Docker (Postgres / infra)
# -------------------------

.PHONY: docker-up docker-check pg-up pg-check pg-reset

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
# Postgres (pgvector) via docker compose
# Assumes your docker-compose.yml defines a `postgres` service
# and (optionally) a named volume like `pg_data`.
# -------------------------

pg-up: docker-up
	@echo ""
	@echo "🐘 [pg] Starting Postgres (docker compose up -d postgres)..."
	@docker compose up -d postgres
	@echo "⏳ [pg] Waiting for Postgres health/ready..."
	@until docker exec -i baikpacking-postgres pg_isready -U baikpacking -d baikpacking >/dev/null 2>&1; do \
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
	@docker exec -i $(PG_CONTAINER) pg_isready -U $(PG_USER) -d $(PG_DB) >/dev/null 2>&1 && \
		echo "✅ Postgres responds to pg_isready" || \
		( echo "❌ Postgres not ready"; exit 1 )
	@echo ""

# WARNING: This will destroy your DB data if you use a named volume.
# It stops containers and removes the Postgres volume.
pg-reset: docker-up
	@echo ""
	@echo "⚠️  [pg] RESET requested: this will DELETE the Postgres data volume."
	@docker compose down --remove-orphans
	@docker volume rm -f $(PG_VOLUME) >/dev/null 2>&1 || true
	@echo "✅ [pg] Reset complete"
	@echo "👉 Run: make pg-up"
	@echo ""



# -------------------------
# Ollama (embeddings)
# -------------------------

.PHONY: ollama-up ollama-check

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
# KB incremental
# -------------------------

DATA_DIR := data
SNAP_RAW_DIR := $(DATA_DIR)/snapshots/raw
SNAP_CLEAN_DIR := $(DATA_DIR)/snapshots/clean
KB_LOAD_MARKER := $(DATA_DIR)/snapshots/.last_db_load

RAW_SNAP_GLOB := dotwatcher_bikes_raw_new_*.jsonl
CLEAN_SNAP_GLOB := dotwatcher_bikes_cleaned_new_*.json



.PHONY: help kb-update kb-scrape kb-clean kb-load kb-check

help:
	@echo ""
	@echo "KB incremental:"
	@echo "  make kb-update   scrape -> (if new) clean -> (if new) load"
	@echo "  make kb-scrape   run incremental scraper (raw new-only snapshots)"
	@echo "  make kb-clean    clean latest raw new-only snapshot + merge latest cleaned"
	@echo "  make kb-load     load latest cleaned new-only snapshot into DB"
	@echo "  make kb-check    show latest snapshot files"
	@echo ""

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

kb-update:
	@$(MAKE) kb-scrape
	@NEW_MARKER="$(SNAP_RAW_DIR)/.last_scrape_new"; \
	if [[ ! -f "$$NEW_MARKER" ]]; then \
		echo "No new raw snapshot created by scrape. Skipping clean/load."; \
		exit 0; \
	fi; \
	RAW_LATEST="$$(tail -n 1 "$$NEW_MARKER")"; \
	echo "New raw snapshot created: $$RAW_LATEST"; \
	$(MAKE) kb-clean; \
	CLEAN_LATEST="$$(ls -1t $(SNAP_CLEAN_DIR)/$(CLEAN_SNAP_GLOB) 2>/dev/null | head -n 1)"; \
	if [[ -z "$$CLEAN_LATEST" ]]; then \
		echo "No clean new-only snapshots found in $(SNAP_CLEAN_DIR). Skipping DB load."; \
		rm -f "$$NEW_MARKER"; \
		exit 0; \
	fi; \
	echo "Latest clean snapshot: $$CLEAN_LATEST"; \
	$(UV) $(PY) -m baikpacking.db.data_loader --input "$$CLEAN_LATEST"; \
	rm -f "$$NEW_MARKER"

kb-check:
	@echo ""
	@echo "Latest raw new-only snapshot:"
	@ls -1t $(SNAP_RAW_DIR)/$(RAW_SNAP_GLOB) 2>/dev/null | head -n 3 || true
	@echo ""
	@echo "Latest clean new-only snapshot:"
	@ls -1t $(SNAP_CLEAN_DIR)/$(CLEAN_SNAP_GLOB) 2>/dev/null | head -n 3 || true
	@echo ""

.PHONY: kb-backfill
kb-backfill:
	$(UV) $(PY) -m baikpacking.db.data_loader --input data/dotwatcher_bikes_cleaned.json

.PHONY: kb-load-file
kb-load-file:
	@if [[ -z "$(FILE)" ]]; then echo "Usage: make kb-load-file FILE=path/to/snapshot.json"; exit 1; fi
	$(UV) $(PY) -m baikpacking.db.data_loader --input "$(FILE)"