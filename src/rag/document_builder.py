from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .documents import Document


class DocumentBuilder:
    """
    Converts structured API specs into retrievable text chunks.
    """

    def build_corpus(self, specs: Iterable[Dict[str, Any]]) -> List[Document]:
        corpus: List[Document] = []
        for spec in specs:
            corpus.extend(self._build_docs_for_spec(spec))
        return corpus

    def _build_docs_for_spec(self, spec: Dict[str, Any]) -> List[Document]:
        docs: List[Document] = []
        docs.append(self._overview_doc(spec))
        docs.append(self._auth_doc(spec))

        for operation in spec.get("operations", []):
            docs.append(self._operation_request_doc(spec, operation))
            docs.append(self._operation_response_doc(spec, operation))

        return [doc for doc in docs if doc.text.strip()]

    def _overview_doc(self, spec: Dict[str, Any]) -> Document:
        text = "\n".join(
            [
                f"API: {spec.get('api_name')}",
                f"Description: {spec.get('service_description', 'n/a')}",
                f"Base URL: {spec.get('base_url', 'n/a')}",
                "Operations:",
                *[
                    f"- {op.get('name')} ({op.get('method')} {op.get('path')}): {op.get('summary', '').strip()}"
                    for op in spec.get("operations", [])
                ],
            ]
        )
        metadata = {"api_name": spec.get("api_name"), "section": "overview"}
        return Document(text=text, metadata=metadata)

    def _auth_doc(self, spec: Dict[str, Any]) -> Document:
        auth = spec.get("auth", {})
        header_lines = []
        for header in auth.get("headers", []):
            parts = [header.get("name")]
            if header.get("format"):
                parts.append(f"format: {header['format']}")
            if header.get("value"):
                parts.append(f"default: {header['value']}")
            header_lines.append(", ".join(parts))

        text = "\n".join(
            [
                f"API: {spec.get('api_name')}",
                "Section: Authentication",
                f"Auth type: {auth.get('type', 'unspecified')}",
                f"Instructions: {auth.get('instructions', '').strip()}",
                f"Token/Key source: {auth.get('token_source', 'n/a')}",
                "Headers:",
                *(f"- {line}" for line in header_lines or ["- none provided"]),
            ]
        )
        metadata = {"api_name": spec.get("api_name"), "section": "auth"}
        return Document(text=text, metadata=metadata)

    def _operation_request_doc(self, spec: Dict[str, Any], op: Dict[str, Any]) -> Document:
        request = op.get("request", {})
        lines = [
            f"API: {spec.get('api_name')}",
            f"Operation: {op.get('name')}",
            "Section: Request",
            f"HTTP: {op.get('method')} {op.get('path')}",
            f"Summary: {op.get('summary', '').strip()}",
        ]
        if op.get("prerequisites"):
            lines.append("Prerequisites:")
            lines.extend(f"- {item}" for item in op["prerequisites"])

        lines.extend(self._format_params("Path params", request.get("path_params")))
        lines.extend(self._format_params("Query params", request.get("query_params")))
        lines.extend(self._format_params("Headers", request.get("headers"), key_name="name"))
        lines.extend(self._format_schema("Request body", request.get("body_schema")))
        lines.extend(self._format_pagination(op.get("pagination")))
        lines.extend(self._format_retries(op.get("retries")))

        example = (request.get("example") or {}).get("curl") or request.get("example")
        if example:
            lines.append("Example request:")
            lines.append(str(example).strip())

        metadata = {
            "api_name": spec.get("api_name"),
            "section": "request",
            "operation": op.get("name"),
        }
        return Document(text="\n".join(lines), metadata=metadata)

    def _operation_response_doc(self, spec: Dict[str, Any], op: Dict[str, Any]) -> Document:
        response = op.get("response", {})
        lines = [
            f"API: {spec.get('api_name')}",
            f"Operation: {op.get('name')}",
            "Section: Response",
            f"HTTP: {op.get('method')} {op.get('path')}",
        ]

        status_codes = response.get("status_codes") or []
        if status_codes:
            lines.append("Status codes:")
            for status in status_codes:
                code = status.get("code")
                meaning = status.get("meaning", "")
                lines.append(f"- {code}: {meaning}")

        lines.extend(self._format_schema("Response body", response.get("body_schema")))

        example = (response.get("example") or {}).get("json") or response.get("example")
        if example:
            lines.append("Example response:")
            lines.append(str(example).strip())

        metadata = {
            "api_name": spec.get("api_name"),
            "section": "response",
            "operation": op.get("name"),
        }
        return Document(text="\n".join(lines), metadata=metadata)

    def _format_params(
        self, title: str, params: Iterable[Dict[str, Any]] | None, key_name: str = "name"
    ) -> List[str]:
        if not params:
            return []
        lines = [f"{title}:"]
        for param in params:
            key = param.get(key_name, "<unknown>")
            param_type = param.get("type", "string")
            required = "required" if param.get("required", False) else "optional"
            description = param.get("description") or ""
            default = f" default={param.get('default')}" if param.get("default") is not None else ""
            lines.append(f"- {key} ({param_type}, {required}{default}): {description}")
        return lines

    def _format_schema(self, title: str, schema: Iterable[Dict[str, Any]] | None) -> List[str]:
        if not schema:
            return []
        lines = [f"{title} schema:"]
        for field in schema:
            name = field.get("field") or field.get("name") or "<field>"
            field_type = field.get("type", "string")
            required = field.get("required")
            required_text = (
                "required" if required is True else "optional" if required is False else "contextual"
            )
            description = field.get("description") or ""
            default = f" default={field.get('default')}" if field.get("default") is not None else ""
            lines.append(f"- {name} ({field_type}, {required_text}{default}): {description}")
        return lines

    def _format_pagination(self, pagination: Dict[str, Any] | None) -> List[str]:
        if not pagination:
            return []

        lines = ["Pagination:"]
        if pagination.get("strategy"):
            lines.append(f"- strategy: {pagination['strategy']}")
        if pagination.get("instructions"):
            lines.append(f"- instructions: {pagination['instructions'].strip()}")
        if pagination.get("completion_condition"):
            lines.append(f"- completion condition: {pagination['completion_condition']}")
        if pagination.get("request_params"):
            lines.append("- request params:")
            lines.extend(
                f"  * {param.get('name')} (default={param.get('default', 'n/a')}): {param.get('description', '')}"
                for param in pagination["request_params"]
            )
        if pagination.get("response_fields"):
            lines.append("- response fields:")
            lines.extend(
                f"  * {field.get('field')} ({field.get('description', '')})"
                for field in pagination["response_fields"]
            )
        return lines

    def _format_retries(self, retries: Dict[str, Any] | None) -> List[str]:
        if not retries:
            return []

        lines = ["Retry strategy:"]
        if retries.get("policy"):
            lines.append(f"- policy: {retries['policy']}")
        if retries.get("max_attempts") is not None:
            lines.append(f"- max attempts: {retries['max_attempts']}")
        if retries.get("backoff_seconds"):
            lines.append(f"- backoff: {retries['backoff_seconds']}")
        if retries.get("retryable_status_codes"):
            codes = ", ".join(str(code) for code in retries["retryable_status_codes"])
            lines.append(f"- retryable status codes: {codes}")
        if retries.get("instructions"):
            lines.append(f"- instructions: {retries['instructions'].strip()}")
        return lines
