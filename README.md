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

- - Gear and kit (bags, sleep system, navigation, power)

Free-text descriptions preserved for semantic retrieval

- This combination allows both structured querying (e.g. filters) and semantic similarity search.

## Architecture of the Knowledge Base

The knowledge base is stored across two complementary systems:

### 1. Postgres (structured storage)

- Acts as the authoritative source of truth

- Stores cleaned and normalized rider records

- Enables deterministic queries (filters by tyre width, frame type, electronic shifting, year, etc.)

Schema is derived from Pydantic models used throughout the project

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