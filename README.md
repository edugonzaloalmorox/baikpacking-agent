# 🚴‍♂️ bAIpacking Agent

An AI agent for exploring bikepacking setups using structured data, retrieval, and multi-tool reasoning

## 📌 Overview

This project develops an LLM-powered agent capable of answering questions about bikepacking race setups (bikes, gear, riders, events). The agent uses a curated knowledge base extracted from DotWatcher.cc (“Bikes of…” articles), stored in a Postgres database as well as vector embeddings in Qdrant, and enhanced with tools for search, similarity queries, structured reasoning, and evaluation.


## 🎯 Problem Description

Bikepacking racers share detailed equipment lists online. This information is spread across dozens of DotWatcher articles and cannot be queried efficiently.

### The problem:

Cyclists cannot easily explore setups from previous riders to decide what bike, components, bags, or electronics to use for an upcoming race- specially if they are novice into this world.

**The bAIpacking Agent solves this by**:

- Scraping and cleaning _“Bikes of…”_ articles

- Parsing riders into a normalized Pydantic schema

- Storing riders in Postgres + embedding them in Qdrant

- Providing an LLM agent with multiple tools for:
  
  - DB search (SQL)

  - Vector search (semantic similarity)

  - Filtering and reasoning

  - Model-based evaluation of results

Example questions the agent can answer:

_“Recommend a setup for a lightweight hardtail rider doing the GranGuanche Trail.”_

_“I'm riding Transiberica, what wheels should I use”_


This is a real-world application combining structured and unstructured data, RAG, evaluation, and monitoring.

# 🧠 Knowledge Base

The core of the bAIpacking Agent is a domain-specific knowledge base (KB) built from historical bikepacking race data.

What the knowledge base contains

The KB is derived from DotWatcher.cc “Bikes of …” articles and represents individual riders and their setups, normalized into a structured schema. Each rider record includes (when available):

- Rider metadata (name, age, location)

- Event context (race name, year, article source)

- Bike and frame details (bike model, frame type, material)

- Components (drivetrain, wheels, tyres, electronics)

- Gear and kit (bags, sleep system, navigation, power)

- Free-text descriptions preserved for semantic retrieval

This combination allows both structured querying (e.g. filters) and semantic similarity search.

## Architecture of the Knowledge Base

The knowledge base is stored across two complementary systems:

### 1. Postgres (structured storage)

- Acts as the authoritative source of truth

- Stores cleaned and normalized rider records

- Enables deterministic queries (filters by tyre width, frame type, electronic shifting, year, etc.)

Typical use cases:

- Counting and filtering riders

- Aggregations (e.g. component frequency)

- Debugging and traceability

### 2. Qdrant (vector database)

- Stores vector embeddings of rider records

- Each vector is linked to the corresponding Postgres record via IDs

- Payload includes key metadata (event, bike type, components) for hybrid reasoning

Typical use cases:

- Semantic similarity search (“riders similar to X”)

- Retrieval-augmented generation (RAG)

- Candidate selection before agent reasoning

This dual-storage approach enables hybrid retrieval: precise filtering + semantic recall.

## Ingestion Pipeline

The ingestion process is designed as a reproducible, modular pipeline:

**Scraping**

- DotWatcher.cc articles are scraped and stored as raw JSON

- Each article is kept intact for traceability

**Parsing & Cleaning**

- Articles are parsed into individual rider sections

- Noise (navigation text, headers, duplicates) is removed

- Fields are normalized (e.g. tyre width, wheel size, electronics)

**Schema Validation**

- Each rider is validated against a Pydantic model

- Missing or ambiguous fields are allowed but explicit

**Persistence**

- Clean rider records are inserted into Postgres

- Textual representations are embedded and upserted into Qdrant

**Linking**

- Postgres IDs and Qdrant vector IDs are cross-referenced

- Enables explainability and result inspection

This separation ensures:

- Clear responsibility between scraping, cleaning, storage, and retrieval

- Easy re-ingestion when models or schemas change

## How the Agent Uses the Knowledge Base

The LLM agent does not answer questions from parametric knowledge alone.

Instead, it:

- Retrieves relevant riders from Postgres and/or Qdrant

- Inspects retrieved evidence (structured + text)

- Reasons over the retrieved setups

- Produces a grounded, explainable answer

This ensures that recommendations are based on real historical setups, not hallucinated configurations.

## Evaluation

### Retrieval evaluation 

This module aims to evaluate and improve **retrieval quality** for a bikepacking / ultracycling setup recommender. This evaluation is done **before** evaluating recommendation text quality.

The logic is based on **retrieval-first**: given a natural-language query, retrieve relevant rider setups (from DotWatcher-style articles) as early as possible in the ranked list.

---

#### Data & Index

- **Source**: Parsed rider setup articles (DotWatcher, etc.)
- **Storage**: Qdrant vector database
- **Indexing unit**: Text chunks

#### Payload includes
- `rider_id` (stable document ID)
- `text` (compact setup description)
- Structured fields:
  - `event_key`
  - `frame_type`
  - `tyre_width`
  - `electronic_shifting`
  - …

**Important**: Since a single rider may have multiple chunks there is deduplication by `rider_id` 

---

#### Queries & Ground Truth

There are three datasets used for the evaluation of the retriever.

- `queries.jsonl`
  - Natural-language queries
  - Mix of constraint-heavy and exploratory queries

- `qrels.jsonl`
  - Manually labeled relevant `rider_id`

#### Relevance
- **Binary relevance** used for retrieval metrics
- **Graded relevance** reserved for later evaluation (e.g. nDCG)

---

####  Pipeline (per query)

**1. Dense Retrieval**
- Embedding-based search via Qdrant
- Oversampling enabled when reranking is active (e.g. `k × 5`)
- Retrieval latency measured separately

**2. Deduplication**
- Deduplicate by `rider_id`
- Keep the chunk with the **best dense score**
- Track diagnostics: `dedupe_in → dedupe_out`

**3. Baseline Ranking**
- Rank deduplicated hits by **dense similarity only**
- Serves as the **safety baseline**

**4. Deterministic Reranking**
- Rule-based, payload-aware reranker
- Soft additive boosts (e.g. tyres, gearing, shifting, event match)
- **Global clamp** ensures dense similarity always dominates
- Rerank latency measured separately (≈ < 1 ms)

**5. Safe Fallback Logic**
Applied **only on labeled queries**:
- If reranking **hurts hitrate@k** → fallback to baseline
- If hitrate ties but **MRR@k decreases** → fallback to baseline

This guarantees **no metric regression** during tuning.

**6. Final Ranking**
- Either reranked or baseline ranking (after fallback)
- Top-k results used for metric computation


