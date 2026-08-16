#!/usr/bin/env python3
"""Offline mock-HTTP tests for the strict QWEB host acceptance client."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import run_qweb_http_acceptance as acceptance


ROOT_BODY = b"<html><title>offline QWEB fixture</title></html>"
ROOT_HASH = hashlib.sha256(ROOT_BODY).hexdigest()
ROOT_ETAG = f'"{ROOT_HASH}"'


def expected_root() -> acceptance.ExpectedRootAsset:
    return acceptance.ExpectedRootAsset(
        body_length=len(ROOT_BODY),
        sha256=ROOT_HASH,
        etag=ROOT_ETAG,
        manifest_path="offline-manifest.json",
        manifest_sha256="0" * 64,
    )


def json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def response(
    status: int,
    body: bytes | dict[str, object],
    *,
    content_type: str,
    etag: str | None = None,
    cache_control: str = acceptance.EXPECTED_API_CACHE_CONTROL,
    headers_override: tuple[tuple[str, str], ...] | None = None,
) -> acceptance.HttpResponse:
    raw = json_bytes(body) if isinstance(body, dict) else body
    reason = {200: "OK", 202: "Accepted"}[status]
    headers: list[tuple[str, str]] = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(raw))),
    ]
    if etag is not None:
        headers.append(("ETag", etag))
    headers.extend(
        [
            ("Cache-Control", cache_control),
            ("Connection", "close"),
        ]
    )
    return acceptance.HttpResponse(
        status=status,
        reason=reason,
        version=11,
        headers=tuple(headers) if headers_override is None else headers_override,
        body=raw,
        elapsed_ms=1.25,
    )


def health(state: str = "idle") -> dict[str, object]:
    return {"service": "qmap-web", "ready": True, "job_state": state}


def status(
    state: str,
    *,
    consumed: int,
    generated: int,
    job_id: int = 1,
) -> dict[str, object]:
    return {
        "job_id": job_id,
        "state": state,
        "prompt_token_count": 5,
        "prompt_tokens_consumed": consumed,
        "generated_count": generated,
        "generated_token_ids": list(
            acceptance.EXPECTED_GENERATED_TOKEN_IDS[:generated]
        ),
        "generated_scores_q26": list(
            acceptance.EXPECTED_GENERATED_SCORES_Q26[:generated]
        ),
        "output_length": acceptance.EXPECTED_OUTPUT_PREFIX_LENGTHS[generated],
        "stop_reason": "MAX_NEW" if state == "done" else "NONE",
        "error": {"job_code": 0, "session_code": 0, "tokenizer_code": 0},
    }


def normal_responses() -> list[acceptance.HttpResponse]:
    return [
        response(
            200,
            ROOT_BODY,
            content_type=acceptance.EXPECTED_ROOT_CONTENT_TYPE,
            etag=ROOT_ETAG,
            cache_control=acceptance.EXPECTED_ROOT_CACHE_CONTROL,
        ),
        response(200, health(), content_type=acceptance.EXPECTED_JSON_CONTENT_TYPE),
        response(
            202,
            {"job_id": 1, "state": "queued"},
            content_type=acceptance.EXPECTED_JSON_CONTENT_TYPE,
        ),
        response(
            200,
            status("queued", consumed=0, generated=0),
            content_type=acceptance.EXPECTED_JSON_CONTENT_TYPE,
        ),
        response(
            200,
            status("running", consumed=3, generated=0),
            content_type=acceptance.EXPECTED_JSON_CONTENT_TYPE,
        ),
        response(
            200,
            status("running", consumed=5, generated=1),
            content_type=acceptance.EXPECTED_JSON_CONTENT_TYPE,
        ),
        response(
            200,
            status("done", consumed=5, generated=2),
            content_type=acceptance.EXPECTED_JSON_CONTENT_TYPE,
        ),
        response(
            200,
            acceptance.EXPECTED_OUTPUT,
            content_type=acceptance.EXPECTED_OUTPUT_CONTENT_TYPE,
        ),
        response(
            200,
            health("done"),
            content_type=acceptance.EXPECTED_JSON_CONTENT_TYPE,
        ),
    ]


class FakeTransport:
    def __init__(self, responses: list[acceptance.HttpResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        body: bytes,
        timeout: float,
        max_body_bytes: int,
    ) -> acceptance.HttpResponse:
        if not self.responses:
            raise AssertionError(f"unexpected request {method} {path}")
        self.calls.append(
            {
                "method": method,
                "path": path,
                "headers": dict(headers),
                "body": body,
                "timeout": timeout,
                "max_body_bytes": max_body_bytes,
            }
        )
        item = self.responses.pop(0)
        if len(item.body) > max_body_bytes:
            raise acceptance.AcceptanceFailure("mock body exceeded client bound")
        return item


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += duration


def make_runner(
    transport: FakeTransport,
) -> tuple[acceptance.AcceptanceRunner, FakeClock]:
    clock = FakeClock()
    runner = acceptance.AcceptanceRunner(
        transport,
        expected_root(),
        request_timeout=10.0,
        job_timeout=30.0,
        poll_interval=0.25,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    return runner, clock


class AcceptanceFlowTests(unittest.TestCase):
    def test_complete_prompt_to_text_flow(self) -> None:
        transport = FakeTransport(normal_responses())
        runner, _ = make_runner(transport)
        result = runner.run()
        self.assertEqual(result["job_id"], 1)
        self.assertEqual(result["output_bytes"], b" a fascinating")
        self.assertEqual(result["output_text"], " a fascinating")
        self.assertEqual(
            result["final_status"]["generated_token_ids"], [264, 26291]
        )
        self.assertEqual(len(runner.status_snapshots), 4)
        self.assertEqual(len(runner.transactions), 9)
        self.assertFalse(transport.responses)
        self.assertEqual(
            [(call["method"], call["path"]) for call in transport.calls],
            [
                ("GET", "/"),
                ("GET", "/api/health"),
                ("POST", "/api/generate"),
                ("GET", "/api/generate/1"),
                ("GET", "/api/generate/1"),
                ("GET", "/api/generate/1"),
                ("GET", "/api/generate/1"),
                ("GET", "/api/generate/1/output"),
                ("GET", "/api/health"),
            ],
        )
        submitted = json.loads(transport.calls[2]["body"].decode("utf-8"))
        self.assertEqual(
            submitted,
            {"prompt": "The future of FPGA is", "max_new_tokens": 2},
        )

    def test_wrong_generated_token_is_rejected(self) -> None:
        items = normal_responses()
        wrong = status("done", consumed=5, generated=2)
        wrong["generated_token_ids"] = [264, 26292]
        items[6] = response(
            200, wrong, content_type=acceptance.EXPECTED_JSON_CONTENT_TYPE
        )
        runner, _ = make_runner(FakeTransport(items))
        with self.assertRaisesRegex(
            acceptance.AcceptanceFailure, "generated token IDs mismatch"
        ):
            runner.run()

    def test_final_only_status_cannot_prove_inference_network_pumping(self) -> None:
        items = normal_responses()
        final_only = items[:3] + [items[6]]
        runner, _ = make_runner(FakeTransport(final_only))
        with self.assertRaisesRegex(
            acceptance.AcceptanceFailure,
            "before any queued/running status response",
        ):
            runner.run()

    def test_wrong_score_is_rejected(self) -> None:
        items = normal_responses()
        wrong = status("running", consumed=5, generated=1)
        wrong["generated_scores_q26"] = [1]
        items[5] = response(
            200, wrong, content_type=acceptance.EXPECTED_JSON_CONTENT_TYPE
        )
        runner, _ = make_runner(FakeTransport(items))
        with self.assertRaisesRegex(
            acceptance.AcceptanceFailure, "generated Q26 scores mismatch"
        ):
            runner.run()

    def test_wrong_output_bytes_are_rejected(self) -> None:
        items = normal_responses()
        items[7] = response(
            200,
            b" a merely interesting",
            content_type=acceptance.EXPECTED_OUTPUT_CONTENT_TYPE,
        )
        runner, _ = make_runner(FakeTransport(items))
        with self.assertRaisesRegex(
            acceptance.AcceptanceFailure, "detokenized output mismatch"
        ):
            runner.run()

    def test_status_counter_regression_is_rejected(self) -> None:
        items = normal_responses()
        items[4] = response(
            200,
            status("running", consumed=4, generated=0),
            content_type=acceptance.EXPECTED_JSON_CONTENT_TYPE,
        )
        items[5] = response(
            200,
            status("running", consumed=3, generated=0),
            content_type=acceptance.EXPECTED_JSON_CONTENT_TYPE,
        )
        runner, _ = make_runner(FakeTransport(items))
        with self.assertRaisesRegex(
            acceptance.AcceptanceFailure, "counter regressed"
        ):
            runner.run()

    def test_duplicate_json_member_is_rejected(self) -> None:
        items = normal_responses()
        duplicate = b'{"service":"qmap-web","ready":true,"ready":true,"job_state":"idle"}'
        items[1] = response(
            200, duplicate, content_type=acceptance.EXPECTED_JSON_CONTENT_TYPE
        )
        runner, _ = make_runner(FakeTransport(items))
        with self.assertRaisesRegex(
            acceptance.AcceptanceFailure, "duplicate member 'ready'"
        ):
            runner.run()

    def test_unknown_status_member_is_rejected(self) -> None:
        items = normal_responses()
        wrong = status("queued", consumed=0, generated=0)
        wrong["prompt_token_ids"] = [785, 3853, 315, 89462, 374]
        items[3] = response(
            200, wrong, content_type=acceptance.EXPECTED_JSON_CONTENT_TYPE
        )
        runner, _ = make_runner(FakeTransport(items))
        with self.assertRaisesRegex(acceptance.AcceptanceFailure, "member set mismatch"):
            runner.run()

    def test_non_string_status_state_is_rejected_cleanly(self) -> None:
        items = normal_responses()
        wrong = status("queued", consumed=0, generated=0)
        wrong["state"] = ["queued"]
        items[3] = response(
            200, wrong, content_type=acceptance.EXPECTED_JSON_CONTENT_TYPE
        )
        runner, _ = make_runner(FakeTransport(items))
        with self.assertRaisesRegex(
            acceptance.AcceptanceFailure, "state must be a string"
        ):
            runner.run()

    def test_response_header_set_is_exact(self) -> None:
        items = normal_responses()
        root = items[0]
        items[0] = acceptance.HttpResponse(
            status=root.status,
            reason=root.reason,
            version=root.version,
            headers=root.headers + (("Server", "mock"),),
            body=root.body,
        )
        runner, _ = make_runner(FakeTransport(items))
        with self.assertRaisesRegex(acceptance.AcceptanceFailure, "header set mismatch"):
            runner.run()

    def test_root_hash_is_bound(self) -> None:
        items = normal_responses()
        items[0] = response(
            200,
            b"x" * len(ROOT_BODY),
            content_type=acceptance.EXPECTED_ROOT_CONTENT_TYPE,
            etag=ROOT_ETAG,
            cache_control=acceptance.EXPECTED_ROOT_CACHE_CONTROL,
        )
        runner, _ = make_runner(FakeTransport(items))
        with self.assertRaisesRegex(acceptance.AcceptanceFailure, "root body SHA-256"):
            runner.run()

    def test_initial_busy_health_is_rejected(self) -> None:
        items = normal_responses()
        items[1] = response(
            200,
            health("running"),
            content_type=acceptance.EXPECTED_JSON_CONTENT_TYPE,
        )
        runner, _ = make_runner(FakeTransport(items))
        with self.assertRaisesRegex(acceptance.AcceptanceFailure, "active job"):
            runner.run()


class ParsingAndEvidenceTests(unittest.TestCase):
    def test_base_url_requires_literal_board_ipv4_and_port_80(self) -> None:
        self.assertEqual(
            acceptance._normalize_base_url("http://192.168.1.10"),
            ("http://192.168.1.10:80/", "192.168.1.10", 80),
        )
        for invalid in (
            "https://192.168.1.10/",
            "http://example.com/",
            "http://127.0.0.1/",
            "http://192.168.1.10:8080/",
            "http://192.168.1.10/api/health",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(acceptance.AcceptanceFailure):
                    acceptance._normalize_base_url(invalid)

    def test_real_asset_manifest_parses(self) -> None:
        manifest = Path(acceptance.__file__).with_name("web_assets_manifest.json")
        root = acceptance._load_expected_root_asset(manifest)
        self.assertEqual(root.body_length, 43_776)
        self.assertEqual(
            root.sha256,
            "39b0b58717f9c0c6fb7c4b73c9bb76d6e65fdd029e3e80e675afaf57824744e7",
        )
        self.assertEqual(root.etag, f'"{root.sha256}"')

    def test_startup_binding_verifies_uart_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "uart_raw.bin"
            raw_path.write_bytes(b"strict-uart-evidence")
            report_path = root / "startup.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "tool": "capture_qweb_uart.py",
                        "mode": "live",
                        "passed": True,
                        "qot_baseaddr": 0xA0040000,
                        "ddr_status": 0x5,
                        "missing_milestones": [],
                        "network": {
                            "board_ip": "192.168.1.10",
                            "ip": "192.168.1.10",
                            "port": 80,
                            "context": 256,
                            "vocab": 151936,
                            "url": "http://192.168.1.10:80/",
                        },
                        "phy": {
                            "address": 7,
                            "id1": 0,
                            "id2": 0x011A,
                            "bmsr": 0x0024,
                            "status": 0xAC00,
                            "duplex": "full",
                            "speed_mbps": 1000,
                        },
                        "tokenizer": {
                            "tokens": 151669,
                            "vocab": 151936,
                            "eos": 151643,
                            "bytes": 3629566,
                        },
                        "artifacts": {
                            "uart_raw": str(raw_path),
                            "uart_raw_sha256": hashlib.sha256(
                                raw_path.read_bytes()
                            ).hexdigest(),
                        },
                    }
                ),
                encoding="utf-8",
            )
            url, binding = acceptance._load_startup_binding(report_path)
            self.assertEqual(url, "http://192.168.1.10:80/")
            self.assertEqual(binding["uart_raw"], str(raw_path.resolve()))
            raw_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(
                acceptance.AcceptanceFailure, "raw UART SHA-256 mismatch"
            ):
                acceptance._load_startup_binding(report_path)

    def test_startup_binding_rejects_transcript_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "startup.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "tool": "capture_qweb_uart.py",
                        "mode": "transcript_replay",
                        "passed": True,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                acceptance.AcceptanceFailure, "passing live capture"
            ):
                acceptance._load_startup_binding(report_path)

    def test_existing_evidence_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "evidence"
            existing.mkdir()
            with self.assertRaisesRegex(
                acceptance.AcceptanceFailure, "refusing to overwrite"
            ):
                acceptance._prepare_output_dir(existing)


if __name__ == "__main__":
    unittest.main()
