# API Identity Agent

This project provides a lightweight agent that inspects an OpenAPI specification and highlights
identity-related endpoints (users, accounts, roles, organisations), authentication/authorisation
flows, and pagination patterns.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

The installation uses `requests` for fetching remote specifications and `PyYAML` for parsing YAML files.

## Usage

Run the agent as a module, supplying either a remote URL or a local file path to an OpenAPI JSON/YAML document:

```bash
python3 -m src.api_spec_agent https://example.com/openapi.yaml
```

Local files are supported directly:

```bash
python3 -m src.api_spec_agent examples/sample_identity_api.yaml --output json
```

### Output formats

- Pretty text (default) summarises security schemes and relevant operations.
- JSON (`--output json`) returns a machine-friendly representation, including matched keywords and pagination hints.

Optional `--verbose` enables debug logs (helpful for troubleshooting downloads and parsing).

## Sample data

`examples/sample_identity_api.yaml` provides a minimal specification you can use to exercise the agent without external dependencies.
