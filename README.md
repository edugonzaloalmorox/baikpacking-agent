🚴‍♂️ bAIpacking Agent

An AI agent for exploring bikepacking setups using structured data, retrieval, and multi-tool reasoning

📌 Overview

This project develops an LLM-powered agent capable of answering questions about bikepacking race setups (bikes, gear, riders, events). The agent uses a curated knowledge base extracted from DotWatcher.cc (“Bikes of…” articles), stored in a Postgres database as well as vector embeddings in Qdrant, and enhanced with tools for search, similarity queries, structured reasoning, and evaluation.


🎯 Problem Description

Bikepacking racers share detailed equipment lists online. This information is spread across dozens of DotWatcher articles and cannot be queried efficiently.

The problem:
👉 Cyclists cannot easily explore setups from previous riders to decide what bike, components, bags, or electronics to use for an upcoming race.

The bAIpacking Agent solves this by:

- Scraping and cleaning “Bikes of…” articles

- Parsing riders into a normalized Pydantic schema

- Storing riders in Postgres + embedding them in Qdrant

- Providing an LLM agent with multiple tools for:

- DB search (SQL)

- Vector search (semantic similarity)

- Filtering and reasoning

- Model-based evaluation of results

Example questions the agent can answer:

`“Recommend a setup for a lightweight hardtail rider doing the GranGuanche Trail.”`

`“Show similar bikes to this Focus Raven with 50mm tyres.”`

`“Which riders used electronic shifting in 2023?”`

This is a real-world application combining structured and unstructured data, RAG, evaluation, and monitoring.