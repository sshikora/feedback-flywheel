# 0001: Python pipeline, generated TypeScript client

**Status:** Accepted, 2026-10-31

**Context.** The first draft split the project into a TypeScript monorepo plus a Python eval package. Metrics, training SDKs and dataset tooling are Python-native, and the split would double the tooling and CI for no benefit to the pipeline. The consuming app is TypeScript, so integration still has to be easy.

**Decision.** The pipeline is Python. Pydantic models are the single source of truth. JSON Schema is generated from them and checked in, and CI fails if it goes stale. A TypeScript client comes later, generated from that schema and from a small HTTP service's OpenAPI spec.

**Consequences.** One language, one test suite. TypeScript types cannot drift from the Python models. Integration before the client exists happens through file exports and an OpenAI-compatible endpoint.
