# 🚴‍♂️ bAIpacking Agent

An AI agent for exploring bikepacking setups using structured data, retrieval, and multi-tool reasoning.

---

## 📌 Overview

This project develops an LLM-powered agent capable of answering questions about bikepacking race setups (bikes, gear, riders, events).

The agent is built on a **domain-specific knowledge base extracted from DotWatcher.cc (“Bikes of…” articles)** and stored entirely in **PostgreSQL**, using:

- **Relational tables** for structured rider data  
- **pgvector** for semantic embeddings and similarity search  

The system combines **structured querying, vector retrieval, and agentic reasoning** to produce grounded, explainable recommendations.

---

## 🎯 Problem Description

Bikepacking racers share detailed equipment lists online. This information is spread across dozens of DotWatcher articles and cannot be queried efficiently.

### The problem

Cyclists cannot easily explore setups from previous riders to decide what bike, components, bags, or electronics to use for an upcoming race — especially if they are new to the bikepacking world.

### The bAIpacking Agent solves this by

- Scraping and cleaning *“Bikes of…”* articles  
- Parsing riders into a normalized Pydantic schema  
- Storing riders and embeddings in **Postgres + pgvector**  
- Providing an LLM agent with multiple tools for:
  - SQL-based structured search
  - Vector similarity search
  - Filtering and reasoning
  - Model-based evaluation of retrieval quality

Example questions the agent can answer:

> *“Recommend a setup for a lightweight hardtail rider doing the GranGuanche Trail.”*  
> *“I'm riding Transiberica — what wheels should I use?”*

This is a real-world application combining **structured + unstructured data**, **RAG**, **evaluation**, and **monitoring**.


# 🧠 Knowledge Base

The core of the bAIpacking Agent is a **domain-specific knowledge base (KB)** built from historical bikepacking race data.

## What the Knowledge Base Contains

The KB is derived from DotWatcher.cc *“Bikes of …”* articles and represents individual riders and their setups, normalized into a structured schema.

Each rider record includes (when available):

- Rider metadata (name, age, location)
- Event context (race name, year, article source)
- Bike and frame details (bike model, frame type, material)
- Components (drivetrain, wheels, tyres, electronics)
- Gear and kit (bags, sleep system, navigation, power)
- Free-text descriptions preserved for semantic retrieval

This design supports both **deterministic filtering** and **semantic similarity search**.

---

## Architecture of the Knowledge Base

The knowledge base is stored **entirely in PostgreSQL**, using two tightly coupled layers:

### 1. PostgreSQL (structured storage)

- Acts as the **single source of truth**
- Stores cleaned and normalized rider records
- Enables deterministic queries:
  - Filters (tyre width, frame type, electronic shifting, year, event)
  - Aggregations (component frequency, distributions)
  - Debugging and traceability

### 2. pgvector (semantic layer)

- Stores vector embeddings of rider records inside Postgres
- Each embedding row is linked to a rider record via a stable ID
- Embeddings are generated from compact textual representations of setups

Typical use cases:

- Semantic similarity search (“riders similar to X”)
- Retrieval-augmented generation (RAG)
- Candidate selection before agent reasoning

This **single-database architecture** avoids operational complexity while enabling **hybrid retrieval** (SQL + vectors).

---

## Ingestion Pipeline

The ingestion process is designed as a **reproducible, modular pipeline**.

### Scraping

- DotWatcher.cc articles are scraped and stored as raw JSON
- Raw articles are preserved for traceability

### Parsing & Cleaning

- Articles are parsed into individual rider sections
- Noise (navigation text, headers, duplicates) is removed
- Fields are normalized (tyre width, wheel size, electronics, etc.)

### Schema Validation

- Each rider is validated against a Pydantic model
- Missing or ambiguous fields are allowed but explicit

### Persistence

- Clean rider records are inserted into PostgreSQL
- Textual representations are embedded
- Embeddings are stored in Postgres using **pgvector**

### Linking

- Rider table IDs and embedding table IDs are explicitly linked
- Enables explainability and result inspection

This separation ensures:

- Clear responsibility between scraping, cleaning, storage, and retrieval
- Easy re-ingestion when schemas or embedding models change

---

## How the Agent Uses the Knowledge Base

The LLM agent does **not** answer questions from parametric knowledge alone.

Instead, it:

1. Retrieves candidate riders using:
   - SQL filters
   - Vector similarity search (pgvector)
2. Inspects retrieved evidence (structured + text)
3. Reasons over real historical setups
4. Produces a grounded, explainable answer

This prevents hallucinated configurations and keeps recommendations traceable.

---

## Evaluation

### Retrieval Evaluation

This module evaluates and improves **retrieval quality** for a bikepacking / ultracycling setup recommender.

Evaluation is **retrieval-first**: given a natural-language query, relevant rider setups should appear as early as possible in the ranked list.

---

### Data & Index

- **Source**: Parsed rider setup articles (DotWatcher)
- **Storage**: PostgreSQL + pgvector
- **Indexing unit**: Text chunks derived from rider setups

Stored fields include:

- `rider_id` (stable document ID)
- `text` (compact setup description)
- Structured attributes:
  - `event_key`
  - `frame_type`
  - `tyre_width`
  - `electronic_shifting`
  - …

**Important**: A single rider may produce multiple chunks. Deduplication is always done by `rider_id`.

---

### Queries & Ground Truth

- `queries.jsonl`: Natural-language queries
- `qrels.jsonl`: Manually labeled relevant `rider_id`s

Relevance:

- **Binary relevance** for retrieval metrics
- **Graded relevance** reserved for later (e.g. nDCG)

---

### Retrieval Pipeline (per query)

1. **Dense Retrieval**
   - Embedding-based search via pgvector
   - Oversampling when reranking is enabled
   - Latency measured independently

2. **Deduplication**
   - Deduplicate by `rider_id`
   - Keep the chunk with the best similarity score

3. **Baseline Ranking**
   - Ranked purely by vector similarity
   - Acts as the safety baseline

4. **Deterministic Reranking**
   - Rule-based, payload-aware reranker
   - Soft boosts (tyres, gearing, shifting, event match)
   - Global clamp ensures similarity dominates

5. **Safe Fallback Logic**
   - If reranking hurts hitrate@k → fallback
   - If hitrate ties but MRR@k decreases → fallback

This guarantees **no metric regression** during tuning.

---

## How to launch the system

The project is designed to be run via a **Makefile**, ensuring reproducibility and minimal setup friction.

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- `make`
- An OpenAI-compatible API key
