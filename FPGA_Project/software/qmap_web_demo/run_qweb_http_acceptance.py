#!/usr/bin/env python3
"""Run the strict host-side HTTP acceptance for the board-hosted QWEB demo."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit


REPORT_SCHEMA_VERSION = 1
EXPECTED_PROMPT = "The future of FPGA is"
EXPECTED_PROMPT_TOKEN_IDS = (785, 3853, 315, 89462, 374)
EXPECTED_GENERATED_TOKEN_IDS = (264, 26291)
EXPECTED_GENERATED_SCORES_Q26 = (1_296_911_292, 1_225_544_557)
EXPECTED_OUTPUT = b" a fascinating"
EXPECTED_OUTPUT_PREFIX_LENGTHS = (0, 2, len(EXPECTED_OUTPUT))
EXPECTED_STOP_REASON = "MAX_NEW"
EXPECTED_SERVICE = "qmap-web"
EXPECTED_HTTP_PORT = 80
EXPECTED_ROOT_CACHE_CONTROL = "public, max-age=60"
EXPECTED_API_CACHE_CONTROL = "no-store"
EXPECTED_ROOT_CONTENT_TYPE = "text/html; charset=utf-8"
EXPECTED_JSON_CONTENT_TYPE = "application/json; charset=utf-8"
EXPECTED_OUTPUT_CONTENT_TYPE = "application/octet-stream"
MAX_ROOT_BYTES = 64 * 1024
MAX_JSON_BYTES = 12_288
MAX_OUTPUT_BYTES = 32_768
MAX_STATUS_POLLS = 10_000

STATUS_KEYS = {
    "job_id",
    "state",
    "prompt_token_count",
    "prompt_tokens_consumed",
    "generated_count",
    "generated_token_ids",
    "generated_scores_q26",
    "output_length",
    "stop_reason",
    "error",
}
ERROR_KEYS = {"job_code", "session_code", "tokenizer_code"}
HEALTH_KEYS = {"service", "ready", "job_state"}
ACCEPTED_KEYS = {"job_id", "state"}
ACTIVE_STATES = {"queued", "running"}


class AcceptanceFailure(RuntimeError):
    """Raised when the board's HTTP behavior does not match the contract."""


@dataclass(frozen=True)
class ExpectedRootAsset:
    body_length: int
    sha256: str
    etag: str
    manifest_path: str
    manifest_sha256: str


@dataclass(frozen=True)
class HttpResponse:
    status: int
    reason: str
    version: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    elapsed_ms: float = 0.0


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
        max_body_bytes: int,
    ) -> HttpResponse: ...


class StdlibHttpTransport:
    """One-request-per-connection HTTP/1.1 transport for the raw-lwIP server."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
        max_body_bytes: int,
    ) -> HttpResponse:
        started = time.monotonic()
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            connection.request(method, path, body=body or None, headers=dict(headers))
            response = connection.getresponse()
            response_body = response.read(max_body_bytes + 1)
            if len(response_body) > max_body_bytes:
                raise AcceptanceFailure(
                    f"{method} {path} response exceeds {max_body_bytes} bytes"
                )
            return HttpResponse(
                status=response.status,
                reason=response.reason,
                version=response.version,
                headers=tuple(response.getheaders()),
                body=response_body,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
            )
        except (OSError, socket.timeout, http.client.HTTPException) as error:
            raise AcceptanceFailure(f"{method} {path} transport failure: {error}") from error
        finally:
            connection.close()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AcceptanceFailure(f"JSON contains duplicate member {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise AcceptanceFailure(f"JSON contains non-finite number {value}")


def _parse_json_bytes(body: bytes, context: str) -> dict[str, Any]:
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise AcceptanceFailure(f"{context} is not strict UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except AcceptanceFailure:
        raise
    except (json.JSONDecodeError, ValueError) as error:
        raise AcceptanceFailure(f"{context} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise AcceptanceFailure(f"{context} must be one JSON object")
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AcceptanceFailure(
            f"{context} member set mismatch: missing={missing} extra={extra}"
        )


def _header_values(response: HttpResponse) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for name, value in response.headers:
        lowered = name.lower()
        result.setdefault(lowered, []).append(value.strip())
    return result


def _require_single_header(
    headers: Mapping[str, list[str]], name: str, expected: str | None = None
) -> str:
    values = headers.get(name.lower(), [])
    if len(values) != 1:
        raise AcceptanceFailure(
            f"response must contain exactly one {name} header, got {len(values)}"
        )
    value = values[0]
    if expected is not None and value != expected:
        raise AcceptanceFailure(
            f"unexpected {name} header: {value!r} != {expected!r}"
        )
    return value


def _validate_response(
    response: HttpResponse,
    *,
    expected_status: int,
    expected_reason: str,
    expected_content_type: str,
    expected_cache_control: str,
    expected_etag: str | None = None,
) -> None:
    if response.version != 11:
        raise AcceptanceFailure(f"response is not HTTP/1.1: version={response.version}")
    if response.status != expected_status or response.reason != expected_reason:
        raise AcceptanceFailure(
            f"unexpected HTTP result: {response.status} {response.reason!r}; "
            f"expected {expected_status} {expected_reason!r}"
        )
    headers = _header_values(response)
    expected_names = {
        "content-type",
        "content-length",
        "cache-control",
        "connection",
    }
    if expected_etag is not None:
        expected_names.add("etag")
    if set(headers) != expected_names:
        raise AcceptanceFailure(
            "HTTP response header set mismatch: "
            f"actual={sorted(headers)} expected={sorted(expected_names)}"
        )
    _require_single_header(headers, "Content-Type", expected_content_type)
    length_text = _require_single_header(headers, "Content-Length")
    if not length_text.isascii() or not length_text.isdigit():
        raise AcceptanceFailure(f"invalid Content-Length header: {length_text!r}")
    if int(length_text, 10) != len(response.body):
        raise AcceptanceFailure(
            f"Content-Length {length_text} does not match {len(response.body)} body bytes"
        )
    _require_single_header(headers, "Cache-Control", expected_cache_control)
    _require_single_header(headers, "Connection", "close")
    if expected_etag is not None:
        _require_single_header(headers, "ETag", expected_etag)


def _response_record(response: HttpResponse) -> dict[str, object]:
    return {
        "status": response.status,
        "reason": response.reason,
        "http_version": "1.1" if response.version == 11 else str(response.version),
        "headers": [{"name": name, "value": value} for name, value in response.headers],
        "body_bytes": len(response.body),
        "body_sha256": _sha256_bytes(response.body),
        "elapsed_ms": round(response.elapsed_ms, 3),
    }


class AcceptanceRunner:
    def __init__(
        self,
        transport: HttpTransport,
        expected_root: ExpectedRootAsset,
        *,
        request_timeout: float,
        job_timeout: float,
        poll_interval: float,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.transport = transport
        self.expected_root = expected_root
        self.request_timeout = request_timeout
        self.job_timeout = job_timeout
        self.poll_interval = poll_interval
        self.monotonic = monotonic
        self.sleep = sleep
        self.transactions: list[dict[str, object]] = []
        self.status_snapshots: list[dict[str, Any]] = []

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes = b"",
        content_type: str | None = None,
        max_body_bytes: int,
    ) -> HttpResponse:
        headers = {
            "Accept": (
                "application/octet-stream"
                if path.endswith("/output")
                else "application/json"
                if path.startswith("/api/")
                else "text/html"
            ),
            "Connection": "close",
        }
        if content_type is not None:
            headers["Content-Type"] = content_type
        response = self.transport.request(
            method,
            path,
            headers=headers,
            body=body,
            timeout=self.request_timeout,
            max_body_bytes=max_body_bytes,
        )
        request_record: dict[str, object] = {
            "method": method,
            "path": path,
            "headers": dict(headers),
            "body_bytes": len(body),
            "body_sha256": _sha256_bytes(body),
        }
        if body:
            try:
                request_record["body_json"] = _parse_json_bytes(body, "request body")
            except AcceptanceFailure:
                request_record["body_base64"] = base64.b64encode(body).decode("ascii")
        self.transactions.append(
            {
                "sequence": len(self.transactions) + 1,
                "request": request_record,
                "response": _response_record(response),
            }
        )
        return response

    def _json_response(
        self,
        method: str,
        path: str,
        *,
        expected_status: int,
        expected_reason: str,
        body: bytes = b"",
        content_type: str | None = None,
    ) -> dict[str, Any]:
        response = self._request(
            method,
            path,
            body=body,
            content_type=content_type,
            max_body_bytes=MAX_JSON_BYTES,
        )
        _validate_response(
            response,
            expected_status=expected_status,
            expected_reason=expected_reason,
            expected_content_type=EXPECTED_JSON_CONTENT_TYPE,
            expected_cache_control=EXPECTED_API_CACHE_CONTROL,
        )
        payload = _parse_json_bytes(response.body, f"{method} {path} response")
        response_record = self.transactions[-1]["response"]
        assert isinstance(response_record, dict)
        response_record["body_json"] = payload
        return payload

    @staticmethod
    def _validate_health(payload: Mapping[str, Any], expected_state: str | None) -> None:
        _expect_keys(payload, HEALTH_KEYS, "health response")
        if payload["service"] != EXPECTED_SERVICE:
            raise AcceptanceFailure(f"unexpected health service: {payload['service']!r}")
        if payload["ready"] is not True:
            raise AcceptanceFailure(f"board health is not ready: {payload['ready']!r}")
        if not isinstance(payload["job_state"], str):
            raise AcceptanceFailure("health job_state must be a string")
        if expected_state is None:
            if payload["job_state"] in ACTIVE_STATES:
                raise AcceptanceFailure(
                    f"board already has an active job: {payload['job_state']}"
                )
            if payload["job_state"] not in {"idle", "done", "error"}:
                raise AcceptanceFailure(
                    f"unexpected initial health job_state: {payload['job_state']!r}"
                )
        elif payload["job_state"] != expected_state:
            raise AcceptanceFailure(
                f"final health job_state {payload['job_state']!r} != {expected_state!r}"
            )

    @staticmethod
    def _validate_accepted(payload: Mapping[str, Any]) -> int:
        _expect_keys(payload, ACCEPTED_KEYS, "generation acceptance")
        if (
            not _is_int(payload["job_id"])
            or payload["job_id"] <= 0
            or payload["job_id"] > 0xFFFF_FFFF
        ):
            raise AcceptanceFailure("accepted job_id must be a positive uint32")
        if payload["state"] != "queued":
            raise AcceptanceFailure(
                f"accepted generation state must be 'queued', got {payload['state']!r}"
            )
        return int(payload["job_id"])

    @staticmethod
    def _validate_status(
        payload: Mapping[str, Any],
        job_id: int,
        previous: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        _expect_keys(payload, STATUS_KEYS, "job status")
        integer_fields = (
            "job_id",
            "prompt_token_count",
            "prompt_tokens_consumed",
            "generated_count",
            "output_length",
        )
        for name in integer_fields:
            if not _is_int(payload[name]) or payload[name] < 0:
                raise AcceptanceFailure(f"job status {name} must be a nonnegative integer")
        if payload["job_id"] != job_id:
            raise AcceptanceFailure(
                f"status job_id {payload['job_id']} does not match accepted {job_id}"
            )
        state = payload["state"]
        if not isinstance(state, str):
            raise AcceptanceFailure("job status state must be a string")
        if state not in {"queued", "running", "done", "error"}:
            raise AcceptanceFailure(f"unexpected job state: {state!r}")
        if payload["prompt_token_count"] != len(EXPECTED_PROMPT_TOKEN_IDS):
            raise AcceptanceFailure(
                "prompt token count mismatch: "
                f"{payload['prompt_token_count']} != {len(EXPECTED_PROMPT_TOKEN_IDS)}"
            )
        consumed = int(payload["prompt_tokens_consumed"])
        if consumed > len(EXPECTED_PROMPT_TOKEN_IDS):
            raise AcceptanceFailure("prompt_tokens_consumed exceeds prompt_token_count")
        generated_count = int(payload["generated_count"])
        if generated_count > len(EXPECTED_GENERATED_TOKEN_IDS):
            raise AcceptanceFailure("generated_count exceeds requested max_new_tokens")
        token_ids = payload["generated_token_ids"]
        scores = payload["generated_scores_q26"]
        if not isinstance(token_ids, list) or not isinstance(scores, list):
            raise AcceptanceFailure("generated token IDs and scores must be arrays")
        if len(token_ids) != generated_count or len(scores) != generated_count:
            raise AcceptanceFailure("generated arrays do not match generated_count")
        if any(not _is_int(value) for value in token_ids + scores):
            raise AcceptanceFailure("generated token IDs and scores must contain integers")
        expected_ids = list(EXPECTED_GENERATED_TOKEN_IDS[:generated_count])
        expected_scores = list(EXPECTED_GENERATED_SCORES_Q26[:generated_count])
        if token_ids != expected_ids:
            raise AcceptanceFailure(
                f"generated token IDs mismatch: {token_ids} != {expected_ids}"
            )
        if scores != expected_scores:
            raise AcceptanceFailure(
                f"generated Q26 scores mismatch: {scores} != {expected_scores}"
            )
        output_length = int(payload["output_length"])
        expected_output_length = EXPECTED_OUTPUT_PREFIX_LENGTHS[generated_count]
        if output_length != expected_output_length:
            raise AcceptanceFailure(
                f"output_length mismatch at generated_count={generated_count}: "
                f"{output_length} != {expected_output_length}"
            )
        if generated_count != 0 and consumed != len(EXPECTED_PROMPT_TOKEN_IDS):
            raise AcceptanceFailure("generation began before all prompt tokens were consumed")
        if not isinstance(payload["stop_reason"], str):
            raise AcceptanceFailure("stop_reason must be a string")
        expected_stop = EXPECTED_STOP_REASON if state == "done" else "NONE"
        if payload["stop_reason"] != expected_stop:
            raise AcceptanceFailure(
                f"stop_reason {payload['stop_reason']!r} != {expected_stop!r}"
            )
        error = payload["error"]
        if not isinstance(error, dict):
            raise AcceptanceFailure("job status error must be an object")
        _expect_keys(error, ERROR_KEYS, "job status error")
        if any(not _is_int(error[name]) for name in ERROR_KEYS):
            raise AcceptanceFailure("job status error codes must be integers")
        if state == "error":
            raise AcceptanceFailure(f"generation entered error state: {error}")
        if any(error[name] != 0 for name in ERROR_KEYS):
            raise AcceptanceFailure(f"nonzero error codes in successful job status: {error}")
        if state == "done":
            if consumed != len(EXPECTED_PROMPT_TOKEN_IDS):
                raise AcceptanceFailure("done status did not consume the complete prompt")
            if generated_count != len(EXPECTED_GENERATED_TOKEN_IDS):
                raise AcceptanceFailure("done status did not produce two tokens")
        if previous is not None:
            for name in (
                "prompt_tokens_consumed",
                "generated_count",
                "output_length",
            ):
                if payload[name] < previous[name]:
                    raise AcceptanceFailure(f"job status counter regressed: {name}")
            previous_state = previous["state"]
            if previous_state == "running" and state == "queued":
                raise AcceptanceFailure("job state regressed from running to queued")
            if previous_state == "done" and state != "done":
                raise AcceptanceFailure("job state changed after done")
        return dict(payload)

    def run(self) -> dict[str, object]:
        root = self._request("GET", "/", max_body_bytes=MAX_ROOT_BYTES)
        _validate_response(
            root,
            expected_status=200,
            expected_reason="OK",
            expected_content_type=EXPECTED_ROOT_CONTENT_TYPE,
            expected_cache_control=EXPECTED_ROOT_CACHE_CONTROL,
            expected_etag=self.expected_root.etag,
        )
        if len(root.body) != self.expected_root.body_length:
            raise AcceptanceFailure(
                f"root body length {len(root.body)} != {self.expected_root.body_length}"
            )
        root_hash = _sha256_bytes(root.body)
        if root_hash != self.expected_root.sha256:
            raise AcceptanceFailure(
                f"root body SHA-256 {root_hash} != {self.expected_root.sha256}"
            )

        health = self._json_response(
            "GET", "/api/health", expected_status=200, expected_reason="OK"
        )
        self._validate_health(health, expected_state=None)

        request_payload = {
            "prompt": EXPECTED_PROMPT,
            "max_new_tokens": len(EXPECTED_GENERATED_TOKEN_IDS),
        }
        request_body = json.dumps(
            request_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        accepted = self._json_response(
            "POST",
            "/api/generate",
            expected_status=202,
            expected_reason="Accepted",
            body=request_body,
            content_type="application/json",
        )
        job_id = self._validate_accepted(accepted)

        deadline = self.monotonic() + self.job_timeout
        previous: dict[str, Any] | None = None
        final_status: dict[str, Any] | None = None
        active_status_observed = False
        for _ in range(MAX_STATUS_POLLS):
            if self.monotonic() > deadline:
                raise AcceptanceFailure(
                    f"generation job {job_id} did not finish within {self.job_timeout:.1f}s"
                )
            payload = self._json_response(
                "GET",
                f"/api/generate/{job_id}",
                expected_status=200,
                expected_reason="OK",
            )
            current = self._validate_status(payload, job_id, previous)
            self.status_snapshots.append(current)
            if current["state"] in ACTIVE_STATES:
                active_status_observed = True
            if current["state"] == "done":
                if not active_status_observed:
                    raise AcceptanceFailure(
                        "job reached done before any queued/running status response; "
                        "network responsiveness during inference was not observed"
                    )
                final_status = current
                break
            previous = current
            remaining = deadline - self.monotonic()
            if remaining <= 0.0:
                continue
            self.sleep(min(self.poll_interval, remaining))
        if final_status is None:
            raise AcceptanceFailure(
                f"generation job {job_id} exceeded {MAX_STATUS_POLLS} status polls"
            )

        output_response = self._request(
            "GET",
            f"/api/generate/{job_id}/output",
            max_body_bytes=MAX_OUTPUT_BYTES,
        )
        _validate_response(
            output_response,
            expected_status=200,
            expected_reason="OK",
            expected_content_type=EXPECTED_OUTPUT_CONTENT_TYPE,
            expected_cache_control=EXPECTED_API_CACHE_CONTROL,
        )
        if output_response.body != EXPECTED_OUTPUT:
            raise AcceptanceFailure(
                "detokenized output mismatch: "
                f"{output_response.body!r} != {EXPECTED_OUTPUT!r}"
            )
        try:
            output_text = output_response.body.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise AcceptanceFailure("final output is not strict UTF-8") from error

        final_health = self._json_response(
            "GET", "/api/health", expected_status=200, expected_reason="OK"
        )
        self._validate_health(final_health, expected_state="done")
        return {
            "job_id": job_id,
            "active_status_observed": active_status_observed,
            "final_status": final_status,
            "output_bytes": output_response.body,
            "output_text": output_text,
            "root_sha256": root_hash,
            "initial_health": health,
            "final_health": final_health,
        }


def _load_expected_root_asset(path: Path) -> ExpectedRootAsset:
    resolved = path.resolve(strict=True)
    payload = _parse_json_bytes(resolved.read_bytes(), "Web asset manifest")
    if payload.get("schema_version") != 1 or not isinstance(payload.get("assets"), list):
        raise AcceptanceFailure("Web asset manifest has an unexpected schema")
    matches = [
        item
        for item in payload["assets"]
        if isinstance(item, dict)
        and isinstance(item.get("routes"), list)
        and "/" in item["routes"]
    ]
    if len(matches) != 1:
        raise AcceptanceFailure("Web asset manifest must contain one root asset")
    item = matches[0]
    required = ("body_length", "sha256", "etag", "mime_type")
    if any(name not in item for name in required):
        raise AcceptanceFailure("root asset manifest entry is incomplete")
    if not _is_int(item["body_length"]) or not (0 < item["body_length"] <= MAX_ROOT_BYTES):
        raise AcceptanceFailure("root asset body_length is invalid")
    sha256 = item["sha256"]
    etag = item["etag"]
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(value not in "0123456789abcdef" for value in sha256)
    ):
        raise AcceptanceFailure("root asset SHA-256 is invalid")
    if etag != f'"{sha256}"':
        raise AcceptanceFailure("root asset ETag is not its quoted SHA-256")
    if item["mime_type"] != EXPECTED_ROOT_CONTENT_TYPE:
        raise AcceptanceFailure("root asset MIME type is unexpected")
    return ExpectedRootAsset(
        body_length=int(item["body_length"]),
        sha256=sha256,
        etag=etag,
        manifest_path=str(resolved),
        manifest_sha256=_sha256_file(resolved),
    )


def _normalize_base_url(value: str) -> tuple[str, str, int]:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise AcceptanceFailure(f"invalid base URL: {error}") from error
    if parsed.scheme != "http":
        raise AcceptanceFailure("base URL scheme must be http")
    if parsed.username is not None or parsed.password is not None:
        raise AcceptanceFailure("base URL must not contain user information")
    if parsed.hostname is None:
        raise AcceptanceFailure("base URL is missing its host")
    try:
        address = ipaddress.IPv4Address(parsed.hostname)
    except ipaddress.AddressValueError as error:
        raise AcceptanceFailure("base URL host must be a literal IPv4 address") from error
    if address.is_unspecified or address.is_multicast or address.is_loopback:
        raise AcceptanceFailure(f"base URL host is not a usable board address: {address}")
    selected_port = EXPECTED_HTTP_PORT if port is None else port
    if selected_port != EXPECTED_HTTP_PORT:
        raise AcceptanceFailure(
            f"base URL port {selected_port} is not the QWEB port {EXPECTED_HTTP_PORT}"
        )
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise AcceptanceFailure("base URL must not contain a path, query, or fragment")
    if parsed.netloc not in (str(address), f"{address}:{selected_port}"):
        raise AcceptanceFailure("base URL authority is not in canonical IPv4 form")
    normalized = f"http://{address}:{selected_port}/"
    return normalized, str(address), selected_port


def _load_startup_binding(path: Path) -> tuple[str, dict[str, object]]:
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    payload = _parse_json_bytes(raw, "QWEB startup report")
    if payload.get("schema_version") != 1:
        raise AcceptanceFailure("QWEB startup report schema is unexpected")
    if (
        payload.get("tool") != "capture_qweb_uart.py"
        or payload.get("mode") != "live"
        or payload.get("passed") is not True
    ):
        raise AcceptanceFailure("QWEB startup report is not a passing live capture")
    network = payload.get("network")
    phy = payload.get("phy")
    artifacts = payload.get("artifacts")
    if (
        not isinstance(network, dict)
        or not isinstance(phy, dict)
        or not isinstance(artifacts, dict)
    ):
        raise AcceptanceFailure("QWEB startup report lacks network/PHY/artifact evidence")
    if (
        phy.get("address") != 7
        or phy.get("id1") != 0
        or phy.get("id2") != 0x011A
        or phy.get("duplex") != "full"
        or phy.get("speed_mbps") not in (10, 100, 1000)
        or not _is_int(phy.get("bmsr"))
        or (int(phy["bmsr"]) & 0x0024) != 0x0024
        or not _is_int(phy.get("status"))
        or (int(phy["status"]) & 0x0C00) != 0x0C00
    ):
        raise AcceptanceFailure("QWEB startup report does not identify the expected YT8521")
    tokenizer = payload.get("tokenizer")
    if (
        payload.get("qot_baseaddr") != 0xA0040000
        or payload.get("ddr_status") != 0x5
        or payload.get("missing_milestones") != []
        or not isinstance(tokenizer, dict)
        or tokenizer
        != {"tokens": 151669, "vocab": 151936, "eos": 151643, "bytes": 3629566}
    ):
        raise AcceptanceFailure("QWEB startup report lacks the exact runtime milestones")
    url = network.get("url")
    if not isinstance(url, str):
        raise AcceptanceFailure("QWEB startup report lacks its READY URL")
    normalized, host, port = _normalize_base_url(url)
    if (
        network.get("board_ip") != host
        or network.get("ip") != host
        or network.get("port") != port
        or network.get("context") != 256
        or network.get("vocab") != 151936
    ):
        raise AcceptanceFailure("QWEB startup report URL is inconsistent with Board IP")
    raw_path_value = artifacts.get("uart_raw")
    raw_sha = artifacts.get("uart_raw_sha256")
    if not isinstance(raw_path_value, str) or not isinstance(raw_sha, str):
        raise AcceptanceFailure("QWEB startup report lacks raw UART provenance")
    raw_path = Path(raw_path_value)
    if not raw_path.is_absolute():
        raw_path = resolved.parent / raw_path
    raw_path = raw_path.resolve(strict=True)
    actual_raw_sha = _sha256_file(raw_path)
    if actual_raw_sha != raw_sha:
        raise AcceptanceFailure(
            f"startup raw UART SHA-256 mismatch: {actual_raw_sha} != {raw_sha}"
        )
    binding = {
        "report": str(resolved),
        "report_sha256": _sha256_bytes(raw),
        "uart_raw": str(raw_path),
        "uart_raw_sha256": raw_sha,
        "url": normalized,
    }
    return normalized, binding


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(tempfile.gettempdir()) / "qweb_http_acceptance" / stamp


def _prepare_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved.mkdir()
    except FileExistsError as error:
        raise AcceptanceFailure(
            f"refusing to overwrite existing evidence directory: {resolved}"
        ) from error
    return resolved


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise AcceptanceFailure(f"refusing to overwrite evidence artifact: {path}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    if path.exists():
        raise AcceptanceFailure(f"refusing to overwrite evidence artifact: {path}")
    os.replace(temporary, path)


def _write_bytes_exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--startup-json",
        required=True,
        type=Path,
        help="passing startup.json from capture_qweb_uart.py",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="optional explicit URL; it must equal the UART READY URL",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        default=Path(__file__).resolve().with_name("web_assets_manifest.json"),
    )
    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument("--job-timeout", type=float, default=3600.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (0.0 < args.request_timeout <= 60.0):
        print("ERROR: --request-timeout must be > 0 and <= 60 seconds", file=sys.stderr)
        return 2
    if not (0.0 < args.job_timeout <= 7200.0):
        print("ERROR: --job-timeout must be > 0 and <= 7200 seconds", file=sys.stderr)
        return 2
    if not (0.05 <= args.poll_interval <= 10.0):
        print("ERROR: --poll-interval must be between 0.05 and 10 seconds", file=sys.stderr)
        return 2

    output_dir: Path | None = None
    report_path: Path | None = None
    report: dict[str, object] | None = None
    try:
        startup_url, startup_binding = _load_startup_binding(args.startup_json)
        if args.base_url is not None:
            requested_url, _, _ = _normalize_base_url(args.base_url)
            if requested_url != startup_url:
                raise AcceptanceFailure(
                    f"--base-url {requested_url} does not match UART READY {startup_url}"
                )
        base_url, host, port = _normalize_base_url(startup_url)
        expected_root = _load_expected_root_asset(args.asset_manifest)
        output_dir = _prepare_output_dir(args.output_dir or _default_output_dir())
        report_path = output_dir / "acceptance.json"
        started_utc = _utc_now()
        runner = AcceptanceRunner(
            StdlibHttpTransport(host, port),
            expected_root,
            request_timeout=args.request_timeout,
            job_timeout=args.job_timeout,
            poll_interval=args.poll_interval,
        )
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "tool": Path(__file__).name,
            "passed": False,
            "failure": None,
            "started_utc": started_utc,
            "finished_utc": None,
            "base_url": base_url,
            "startup_binding": startup_binding,
            "asset_binding": {
                "manifest": expected_root.manifest_path,
                "manifest_sha256": expected_root.manifest_sha256,
                "root_body_bytes": expected_root.body_length,
                "root_body_sha256": expected_root.sha256,
                "root_etag": expected_root.etag,
            },
            "request": {
                "prompt": EXPECTED_PROMPT,
                "expected_prompt_token_ids": list(EXPECTED_PROMPT_TOKEN_IDS),
                "max_new_tokens": len(EXPECTED_GENERATED_TOKEN_IDS),
            },
            "expected": {
                "generated_token_ids": list(EXPECTED_GENERATED_TOKEN_IDS),
                "generated_scores_q26": list(EXPECTED_GENERATED_SCORES_Q26),
                "output_utf8": EXPECTED_OUTPUT.decode("utf-8"),
                "output_base64": base64.b64encode(EXPECTED_OUTPUT).decode("ascii"),
                "stop_reason": EXPECTED_STOP_REASON,
            },
            "transactions": runner.transactions,
            "status_snapshots": runner.status_snapshots,
            "result": None,
            "artifacts": {"report": str(report_path)},
        }
        try:
            result = runner.run()
            output_bytes = result.pop("output_bytes")
            assert isinstance(output_bytes, bytes)
            output_path = output_dir / "output.bin"
            _write_bytes_exclusive(output_path, output_bytes)
            report["result"] = result
            report["artifacts"] = {
                "report": str(report_path),
                "output": str(output_path),
                "output_bytes": len(output_bytes),
                "output_sha256": _sha256_bytes(output_bytes),
            }
            report["passed"] = True
        except AcceptanceFailure as error:
            report["failure"] = str(error)
        report["finished_utc"] = _utc_now()
        _write_json_exclusive(report_path, report)
    except (AcceptanceFailure, OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    if report["passed"]:
        result = report["result"]
        assert isinstance(result, dict)
        print(
            "PASS QWEB HTTP prompt-to-text acceptance: "
            f"job={result['job_id']} output={result['output_text']!r}"
        )
        print(f"report: {report_path}")
        return 0
    print(f"FAIL: {report['failure']}", file=sys.stderr)
    print(f"report: {report_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
