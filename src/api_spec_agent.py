from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

import requests
import yaml


LOGGER = logging.getLogger(__name__)


DEFAULT_KEYWORDS = {
    "user",
    "account",
    "profile",
    "identity",
    "role",
    "permission",
    "group",
    "organisation",
    "organization",
    "tenant",
    "team",
    "member",
    "auth",
    "authentication",
    "authorization",
    "authorisation",
    "login",
    "logout",
    "register",
    "signup",
    "password",
    "token",
}


AUTH_OPERATION_KEYWORDS = {
    "auth",
    "authentication",
    "authorization",
    "authorisation",
    "login",
    "logout",
    "token",
    "password",
    "mfa",
    "2fa",
    "refresh",
}


PAGINATION_PARAM_NAMES = {
    "page",
    "per_page",
    "perPage",
    "limit",
    "offset",
    "cursor",
    "next",
    "prev",
    "previous",
    "pageSize",
    "pageNumber",
    "start",
    "count",
}


PAGINATION_RESPONSE_FIELDS = {
    "next",
    "nextPage",
    "next_page",
    "links",
    "paging",
    "pagination",
    "cursor",
    "hasMore",
    "has_more",
    "total",
    "count",
    "offset",
}


@dataclass
class PaginationInfo:
    parameters: List[str] = field(default_factory=list)
    headers: List[str] = field(default_factory=list)
    response_fields: List[str] = field(default_factory=list)

    def is_present(self) -> bool:
        return bool(self.parameters or self.headers or self.response_fields)


@dataclass
class OperationSummary:
    path: str
    method: str
    summary: Optional[str]
    description: Optional[str]
    keywords_matched: List[str]
    tags: Sequence[str]
    security: Sequence[Dict[str, Any]]
    pagination: PaginationInfo


class ApiSpecAgent:
    """Agent to analyse API specifications for identity-related endpoints."""

    def __init__(self, *, keywords: Optional[Iterable[str]] = None, timeout: int = 30) -> None:
        self.keywords = {kw.lower() for kw in (keywords or DEFAULT_KEYWORDS)}
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, url: str) -> Dict[str, Any]:
        """Fetch, parse, and analyse an API specification."""

        spec = self._load_spec_from_url(url)
        relevant_operations = self._extract_relevant_operations(spec)
        security_schemes = self._extract_security_schemes(spec)

        return {
            "source_url": url,
            "spec_info": spec.get("info", {}),
            "relevant_operations": [self._operation_to_dict(op) for op in relevant_operations],
            "security_schemes": security_schemes,
            "global_security": spec.get("security", []),
        }

    # ------------------------------------------------------------------
    # Specification loading helpers
    # ------------------------------------------------------------------
    def _load_spec_from_url(self, url: str) -> Dict[str, Any]:
        parsed = urlparse(url)

        if parsed.scheme in {"http", "https"}:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
            raw_text = response.text

            LOGGER.debug("Fetched spec from %s with content-type %s", url, content_type)

            if content_type in {"application/json", "application/vnd.oai.openapi+json"}:
                return self._parse_json(raw_text, url)

            if content_type in {"application/x-yaml", "application/yaml", "text/yaml", "application/vnd.oai.openapi"}:
                return self._parse_yaml(raw_text, url)

            # Content type not definitive; try JSON then YAML
            try:
                return self._parse_json(raw_text, url)
            except ValueError:
                return self._parse_yaml(raw_text, url)

        if parsed.scheme in {"", "file"}:
            path_str = parsed.path if parsed.scheme == "file" else url
            path = Path(path_str).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Specification file not found: {path}")

            raw_text = path.read_text(encoding="utf-8")
            content_type, _ = mimetypes.guess_type(path.name)
            content_type = (content_type or "").lower()

            LOGGER.debug("Loaded spec from %s with guessed content-type %s", path, content_type)

            if path.suffix.lower() in {".json"} or content_type in {"application/json", "application/vnd.oai.openapi+json"}:
                return self._parse_json(raw_text, str(path))

            return self._parse_yaml(raw_text, str(path))

        raise ValueError(f"Unsupported URL scheme for specification: {url}")

    @staticmethod
    def _parse_json(raw_text: str, url: str) -> Dict[str, Any]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - informative exception path
            raise ValueError(f"Failed to parse JSON from {url}: {exc}") from exc

    @staticmethod
    def _parse_yaml(raw_text: str, url: str) -> Dict[str, Any]:
        try:
            parsed = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:  # pragma: no cover - informative exception path
            raise ValueError(f"Failed to parse YAML from {url}: {exc}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"YAML spec at {url} did not produce a dictionary")

        return parsed

    # ------------------------------------------------------------------
    # Extraction / analysis helpers
    # ------------------------------------------------------------------
    def _extract_relevant_operations(self, spec: Dict[str, Any]) -> List[OperationSummary]:
        operations: List[OperationSummary] = []
        paths = spec.get("paths", {})

        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue

            for method, operation in methods.items():
                if not isinstance(operation, dict):
                    continue

                if method.lower() not in {"get", "post", "put", "patch", "delete", "options", "head"}:
                    continue

                keywords_matched = self._match_operation_keywords(path, method, operation, spec)
                if not keywords_matched:
                    continue

                pagination_info = self._inspect_pagination(operation, spec)
                security = operation.get("security", [])

                operations.append(
                    OperationSummary(
                        path=path,
                        method=method.upper(),
                        summary=operation.get("summary"),
                        description=operation.get("description"),
                        keywords_matched=sorted(set(keywords_matched)),
                        tags=operation.get("tags", []),
                        security=security,
                        pagination=pagination_info,
                    )
                )

        return operations

    def _match_operation_keywords(self, path: str, method: str, operation: Dict[str, Any], spec: Dict[str, Any]) -> List[str]:
        haystacks = [path, method, operation.get("summary", ""), operation.get("description", ""), operation.get("operationId", "")]

        for key in ("tags", "x-tags"):
            value = operation.get(key)
            if isinstance(value, list):
                haystacks.extend(str(item) for item in value)

        matched: List[str] = []
        for haystack in haystacks:
            if not haystack:
                continue

            haystack_lower = str(haystack).lower()
            for keyword in self.keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", haystack_lower):
                    matched.append(keyword)

        parameters = operation.get("parameters", [])
        for param in parameters:
            resolved = self._resolve_parameter(param, spec)
            if resolved and isinstance(resolved, dict):
                name = resolved.get("name", "")
                if name and name.lower() in self.keywords:
                    matched.append(name.lower())

        return matched

    def _inspect_pagination(self, operation: Dict[str, Any], spec: Dict[str, Any]) -> PaginationInfo:
        info = PaginationInfo()

        parameters = operation.get("parameters", [])
        for param in parameters:
            resolved = self._resolve_parameter(param, spec)
            if not isinstance(resolved, dict):
                continue

            name = resolved.get("name")
            if name and name in PAGINATION_PARAM_NAMES:
                info.parameters.append(name)

        responses = operation.get("responses", {})
        for response_code, response in responses.items():
            resolved_response = self._resolve_reference(response, spec)
            if not isinstance(resolved_response, dict):
                continue

            headers = resolved_response.get("headers", {})
            for header_name in headers.keys():
                if header_name.lower() in {"link", "x-next-page", "x-next-cursor"}:
                    info.headers.append(header_name)

            content = resolved_response.get("content", {})
            for media_type, media in content.items():
                schema = media.get("schema")
                resolved_schema = self._resolve_schema(schema, spec)
                self._collect_pagination_fields(resolved_schema, spec, info)

        return info

    def _collect_pagination_fields(self, schema: Optional[Dict[str, Any]], spec: Dict[str, Any], info: PaginationInfo) -> None:
        if not isinstance(schema, dict):
            return

        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            for prop_name, prop_schema in properties.items():
                if prop_name in PAGINATION_RESPONSE_FIELDS:
                    info.response_fields.append(prop_name)
                resolved = self._resolve_schema(prop_schema, spec)
                self._collect_pagination_fields(resolved, spec, info)

        if schema.get("type") == "array":
            items = schema.get("items")
            resolved_items = self._resolve_schema(items, spec)
            self._collect_pagination_fields(resolved_items, spec, info)

    def _extract_security_schemes(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        security_schemes = spec.get("components", {}).get("securitySchemes", {})
        summaries = []
        for name, scheme in security_schemes.items():
            resolved = self._resolve_reference(scheme, spec)
            if not isinstance(resolved, dict):
                continue

            summaries.append(
                {
                    "name": name,
                    "type": resolved.get("type"),
                    "scheme": resolved.get("scheme"),
                    "bearerFormat": resolved.get("bearerFormat"),
                    "flows": resolved.get("flows"),
                    "description": resolved.get("description"),
                    "in": resolved.get("in"),
                    "name_field": resolved.get("name"),
                }
            )

        return summaries

    # ------------------------------------------------------------------
    # Reference resolution helpers
    # ------------------------------------------------------------------
    def _resolve_reference(self, node: Any, spec: Dict[str, Any]) -> Any:
        if not isinstance(node, dict) or "$ref" not in node:
            return node

        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            return node

        parts = ref.lstrip("#/").split("/")
        target: Any = spec
        try:
            for part in parts:
                target = target[part]
        except (KeyError, TypeError):
            LOGGER.debug("Failed to resolve reference %s", ref)
            return node

        # Support nested $ref
        return self._resolve_reference(target, spec)

    def _resolve_schema(self, schema: Any, spec: Dict[str, Any]) -> Any:
        return self._resolve_reference(schema, spec)

    def _resolve_parameter(self, param: Any, spec: Dict[str, Any]) -> Any:
        if not isinstance(param, dict):
            return None
        return self._resolve_reference(param, spec)

    def _operation_to_dict(self, operation: OperationSummary) -> Dict[str, Any]:
        data = {
            "path": operation.path,
            "method": operation.method,
            "summary": operation.summary,
            "description": operation.description,
            "tags": list(operation.tags),
            "keywords_matched": operation.keywords_matched,
            "security": operation.security,
        }

        if operation.pagination.is_present():
            data["pagination"] = {
                "parameters": operation.pagination.parameters,
                "headers": operation.pagination.headers,
                "response_fields": operation.pagination.response_fields,
            }

        return data


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s - %(message)s")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse an API specification for identity-related endpoints")
    parser.add_argument("url", help="URL to the API specification (OpenAPI JSON or YAML)")
    parser.add_argument("--output", choices={"json", "pretty"}, default="pretty", help="Output format")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    agent = ApiSpecAgent()
    try:
        result = agent.run(args.url)
    except Exception as exc:  # pragma: no cover - CLI convenience path
        LOGGER.error("Failed to analyse specification: %s", exc)
        return 1

    if args.output == "json":
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0

    _render_pretty(result)
    return 0


def _render_pretty(result: Dict[str, Any]) -> None:
    info = result.get("spec_info", {})
    title = info.get("title", "Unnamed API")
    version = info.get("version", "?")
    print(f"Specification: {title} (version {version})")
    print(f"Source URL: {result.get('source_url')}")
    print()

    security_schemes = result.get("security_schemes", [])
    if security_schemes:
        print("Security Schemes:")
        for scheme in security_schemes:
            description = scheme.get("description") or ""
            desc_suffix = f" ? {description}" if description else ""
            scheme_line = f"  - {scheme.get('name')} ({scheme.get('type')})"
            if scheme.get("scheme"):
                scheme_line += f" scheme={scheme['scheme']}"
            if scheme.get("bearerFormat"):
                scheme_line += f" bearerFormat={scheme['bearerFormat']}"
            print(scheme_line + desc_suffix)
        print()

    operations = result.get("relevant_operations", [])
    if not operations:
        print("No identity-related operations detected.")
        return

    print("Relevant Operations:")
    for op in operations:
        print(f"  - {op['method']} {op['path']}")
        if op.get("summary"):
            print(f"      summary: {op['summary']}")
        if op.get("tags"):
            print(f"      tags: {', '.join(op['tags'])}")
        print(f"      keywords: {', '.join(op['keywords_matched'])}")
        if op.get("security"):
            print(f"      security: {op['security']}")
        pagination = op.get("pagination")
        if pagination:
            print("      pagination:")
            for key, values in pagination.items():
                if values:
                    joined = ", ".join(values)
                    print(f"        {key}: {joined}")
        print()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
