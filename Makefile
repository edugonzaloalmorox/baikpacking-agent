SHELL := /bin/bash

UV ?= uv run
PY ?= python

DATA_DIR := data
SNAP_RAW_DIR := $(DATA_DIR)/snapshots/raw
SNAP_CLEAN_DIR := $(DATA_DIR)/snapshots/clean

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
	$(UV) $(PY) -m baikpacking.db.data_loader

# Incremental KB update:
# - Run scraper
# - If scraper produced a newer raw snapshot than the last cleaned snapshot -> run clean + load
# - Else skip
kb-update: kb-scrape
	@RAW_LATEST="$$(ls -1t $(SNAP_RAW_DIR)/$(RAW_SNAP_GLOB) 2>/dev/null | head -n 1)"; \
	CLEAN_LATEST="$$(ls -1t $(SNAP_CLEAN_DIR)/$(CLEAN_SNAP_GLOB) 2>/dev/null | head -n 1)"; \
	if [[ -z "$$RAW_LATEST" ]]; then \
		echo "No raw new-only snapshots found in $(SNAP_RAW_DIR). Nothing to do."; \
		exit 0; \
	fi; \
	if [[ -z "$$CLEAN_LATEST" ]]; then \
		echo "No clean new-only snapshots found yet. Running clean + load..."; \
		$(MAKE) kb-clean; \
		$(MAKE) kb-load; \
		exit 0; \
	fi; \
	if [[ "$$RAW_LATEST" -nt "$$CLEAN_LATEST" ]]; then \
		echo "New raw snapshot detected: $$RAW_LATEST"; \
		$(MAKE) kb-clean; \
		$(MAKE) kb-load; \
	else \
		echo "No new raw snapshot since last clean snapshot. Skipping clean/load."; \
	fi

kb-check:
	@echo ""
	@echo "Latest raw new-only snapshot:"
	@ls -1t $(SNAP_RAW_DIR)/$(RAW_SNAP_GLOB) 2>/dev/null | head -n 3 || true
	@echo ""
	@echo "Latest clean new-only snapshot:"
	@ls -1t $(SNAP_CLEAN_DIR)/$(CLEAN_SNAP_GLOB) 2>/dev/null | head -n 3 || true
	@echo ""
