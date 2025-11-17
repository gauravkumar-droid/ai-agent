# API Call RAG Toolkit

This project seeds a Retrieval-Augmented Generation (RAG) workflow that captures
all of the details needed to make downstream API calls successfully. The
knowledge base keeps authentication requirements (Basic auth, bearer tokens,
API keys, etc.), connection prerequisites, and end-to-end request/response
schemas so delivery teams can quickly look up what is required for any
integration.

## What's Included

- `data/api_specs/*.yaml` — source of truth for each upstream API. Provide auth
  instructions, headers, schema fields, sample requests, and sample responses.
- Includes directory-focused specs (e.g., `user_accounts.yaml`) that document
  cursor/page-based pagination flows plus retry policies so you can script
  full exports safely.
- Ships with a support-focused integration (`zendesk_support.yaml`) to
  illustrate how application-specific content stays isolated (queries about
  “Zendesk” won’t leak payments or inventory details).
- `src/rag` — lightweight Python package that:
  - loads the YAML specs,
  - expands them into semantically rich text chunks tagged by application name,
  - indexes the corpus with a TF‑IDF vector store, and
  - answers natural-language questions with the most relevant context.
- `artifacts/vector_store.joblib` — persisted embedding store generated during
  ingestion (can be regenerated at any time).
- `requirements.txt` — minimal runtime dependencies (`pyyaml`, `scikit-learn`,
  `joblib`).

## Quickstart

```bash
cd /workspace
pip install -r requirements.txt
PYTHONPATH=src python3 -m rag.cli ingest
PYTHONPATH=src python3 -m rag.cli query "How do I authenticate for the Zendesk API?"
PYTHONPATH=src python3 -m rag.cli query --api-name "Payments Service" "What retries should I use?"
```

Flags:

- `--spec-dir`: where YAML specs live (default `data/api_specs`)
- `--store-path`: where to write/read the vector store (default
  `artifacts/vector_store.joblib`)
- `--k`: number of contexts to retrieve when querying
- `--json`: emit structured output for automation pipelines
- `--api-name`: force retrieval to one integration (otherwise the engine will
  auto-detect when the question names a known application like “Zendesk”)

## Adding Additional APIs

1. Drop a new `*.yaml` file into `data/api_specs/` following the existing
   examples (auth, operations, request schema, response schema, samples).
   Always set:
   - `application`: human-friendly product/system name.
   - `aliases`: list of keywords (e.g., `["zendesk", "zendesk support"]`)
     so the RAG engine can auto-scope queries mentioning that app.
2. Re-run `python3 -m rag.cli ingest` to rebuild the store.
3. Run queries to validate the new knowledge is discoverable.

The ingestion flow automatically creates dedicated chunks for:

- Overall service description and base URL
- Authentication requirements (including secrets source)
- Each operation's request contract (headers, params, body schema, pagination
  knobs, retry expectations)
- Each operation's response schema and status codes (including paging tokens)

This ensures downstream consumers can retrieve both connection requirements and
payload structure in one go.

## Design Notes

- **Vector store**: TF‑IDF + cosine similarity (`scikit-learn`). Lightweight,
  deterministic, and easy to run locally without GPU dependencies.
- **Document builder**: Guarantees that every spec emits consistent labeled
  text, including pagination + retry instructions, making the retrieved
  excerpts easy to reformat into prompts or runbooks.
- **Per-app segmentation**: Documents carry `application` and `aliases`
  metadata. Retrieval either auto-detects the app mentioned in the question or
  honors `--api-name`, so Zendesk queries never mix with Payments, Inventory,
  etc.
- **RAG answer synthesis**: Produces a templated summary plus the raw retrieved
  chunks so you can forward them to an LLM, display them in tooling, or export
  as JSON.

## Next Steps

- Plug retrieved contexts into your preferred LLM for final response generation.
- Extend the YAML schema with SLA notes, rate limits, or environment-specific
  overrides.
- Replace TF‑IDF with a neural embedding model (e.g., `sentence-transformers`)
  if you need fuzzier semantic recall.
