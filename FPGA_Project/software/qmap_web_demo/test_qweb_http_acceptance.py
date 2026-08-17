#!/usr/bin/env python3
"""Offline mock-HTTP tests for the strict QWEB host acceptance client."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_qweb_http_acceptance as acceptance


HERE = Path(__file__).resolve().parent
WRAPPER = HERE / "run_qweb_board.ps1"
TCL_LAUNCHER = HERE / "launch_qweb_board.tcl"
VITIS_XSDB = Path(
    r"D:\Applications\Vivado_2025.1.1\2025.1.1\Vitis\bin\xsdb.bat"
)
VITIS_LOADER = VITIS_XSDB.with_name("loader.bat")
VITIS_SETUP_ENV = VITIS_XSDB.with_name("setupEnv.bat")
VITIS_RDI_ARGS = VITIS_XSDB.with_name("rdiArgs.bat")
VITIS_ROOT = VITIS_XSDB.parent.parent
VITIS_XSDB_EXE = VITIS_ROOT / "bin" / "unwrapped" / "win64.o" / "xsdb.exe"
VITIS_XSDB_MANIFEST = VITIS_XSDB_EXE.with_suffix(".exe.manifest")
PINNED_VITIS_FILES = (
    VITIS_XSDB,
    VITIS_LOADER,
    VITIS_SETUP_ENV,
    VITIS_RDI_ARGS,
    VITIS_XSDB_EXE,
    VITIS_XSDB_MANIFEST,
)

ROOT_BODY = b"<html><title>offline QWEB fixture</title></html>"
ROOT_HASH = hashlib.sha256(ROOT_BODY).hexdigest()
ROOT_ETAG = f'"{ROOT_HASH}"'


def valid_startup_transcript(ip: str = "192.168.1.10") -> bytes:
    lines = (
        "Qwen3-0.6B full28 PS Web demo",
        "QOT_BASEADDR=0x00000000a0040000",
        "Detected Motorcomm YT8521 at PHY address 7 (id1=0x0000 id2=0x011a)",
        "YT8521 link resolved: bmsr=0x0024 status=0xac00 duplex=full",
        f"Board IP: {ip}",
        "DDR4 status=0x00000005",
        "TOKENIZER tokens=151669 model_vocab=151936 eos=151643 bytes=3629566",
        f"QWEB READY http://{ip}:80/ context=256 vocab=151936",
    )
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


def write_live_startup_fixture(
    root: Path,
    *,
    transcript: bytes | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    if not all(path.is_file() for path in PINNED_VITIS_FILES):
        raise unittest.SkipTest("pinned Vitis 2025.1 XSDB toolchain is not installed")
    raw = valid_startup_transcript() if transcript is None else transcript
    (root / "xsdb_profile").mkdir()
    raw_path = root / "uart_raw.bin"
    raw_path.write_bytes(raw)
    parsed = acceptance.capture_qweb_uart.verify_transcript_bytes(raw)
    if parsed.get("passed") is not True:
        raise AssertionError(f"invalid test startup transcript: {parsed}")
    report_path = root / "startup.json"
    now = datetime.now(timezone.utc)
    startup_started = now - timedelta(seconds=3)
    launch_started = now - timedelta(seconds=2)
    launch_finished = now - timedelta(seconds=1)
    uart_ready = now - timedelta(milliseconds=500)
    run_id = "12345678-1234-4234-8234-123456789abc"
    claim_path = root / "launch.claim.json"
    claim_payload = {
        "schema_version": 1,
        "tool": "run_qweb_board.ps1",
        "state": "claimed",
        "run_id": run_id,
        "started_utc": launch_started.isoformat(),
        "evidence_directory": str(root.resolve()),
    }
    claim_path.write_text(json.dumps(claim_payload), encoding="utf-8")
    log_path = root / "xsdb.log"
    log_path.write_text(
        "\n".join(acceptance.capture_qweb_uart.REQUIRED_XSDB_MARKERS) + "\n",
        encoding="utf-8",
    )
    launch_path = root / "launch.json"
    launch_payload = {
        "schema_version": 1,
        "tool": "run_qweb_board.ps1",
        "run_id": run_id,
        "passed": True,
        "failure": None,
        "started_utc": launch_started.isoformat(),
        "finished_utc": launch_finished.isoformat(),
        "evidence_directory": str(root.resolve()),
        "xsdb_exit_code": 0,
        "claim": {
            "path": "launch.claim.json",
            "sha256": hashlib.sha256(claim_path.read_bytes()).hexdigest(),
        },
        "audited": acceptance.capture_qweb_uart.EXPECTED_LAUNCH_AUDIT,
        "launcher": {
            "wrapper": str(WRAPPER),
            "wrapper_sha256": hashlib.sha256(WRAPPER.read_bytes()).hexdigest(),
            "tcl": str(TCL_LAUNCHER),
            "tcl_sha256": hashlib.sha256(TCL_LAUNCHER.read_bytes()).hexdigest(),
            "xsdb": str(VITIS_XSDB),
            "xsdb_sha256": hashlib.sha256(VITIS_XSDB.read_bytes()).hexdigest(),
            "loader": str(VITIS_LOADER),
            "loader_sha256": hashlib.sha256(VITIS_LOADER.read_bytes()).hexdigest(),
            "setup_env": str(VITIS_SETUP_ENV),
            "setup_env_sha256": hashlib.sha256(
                VITIS_SETUP_ENV.read_bytes()
            ).hexdigest(),
            "rdi_args": str(VITIS_RDI_ARGS),
            "rdi_args_sha256": hashlib.sha256(
                VITIS_RDI_ARGS.read_bytes()
            ).hexdigest(),
            "xsdb_exe": str(VITIS_XSDB_EXE),
            "xsdb_exe_sha256": hashlib.sha256(
                VITIS_XSDB_EXE.read_bytes()
            ).hexdigest(),
            "xsdb_manifest": str(VITIS_XSDB_MANIFEST),
            "xsdb_manifest_sha256": hashlib.sha256(
                VITIS_XSDB_MANIFEST.read_bytes()
            ).hexdigest(),
            "command": str(VITIS_XSDB),
            "arguments": ["-no-ini", str(TCL_LAUNCHER)],
            "working_directory": "xsdb_profile",
            "isolated_home": "xsdb_profile",
            "profile_initially_empty": True,
            "sanitized_environment": list(
                acceptance.capture_qweb_uart.EXPECTED_SANITIZED_XSDB_ENVIRONMENT
            ),
            "required_output_markers": list(
                acceptance.capture_qweb_uart.REQUIRED_XSDB_MARKERS
            ),
            "output_log": "xsdb.log",
            "output_log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        },
    }
    launch_path.write_text(json.dumps(launch_payload), encoding="utf-8")
    launch_raw = launch_path.read_bytes()
    payload: dict[str, object] = {
        "schema_version": 1,
        "tool": "capture_qweb_uart.py",
        "mode": "live",
        "started_utc": startup_started.isoformat(),
        "finished_utc": now.isoformat(),
        **parsed,
        "uart_ready_utc": uart_ready.isoformat(),
        "launch_binding": {
            "report": "launch.json",
            "report_sha256": hashlib.sha256(launch_raw).hexdigest(),
            "run_id": run_id,
            "started_utc": launch_started.isoformat(),
            "finished_utc": launch_finished.isoformat(),
            "audited": acceptance.capture_qweb_uart.EXPECTED_LAUNCH_AUDIT,
            "claim_sha256": hashlib.sha256(claim_path.read_bytes()).hexdigest(),
            "xsdb_log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        },
        "artifacts": {
            "uart_raw": str(raw_path),
            "uart_raw_sha256": hashlib.sha256(raw).hexdigest(),
        },
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return report_path, raw_path, payload


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
        self.assertIs(result["running_status_observed"], True)
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
            "before any running status response",
        ):
            runner.run()

    def test_queued_status_does_not_prove_inference_network_pumping(self) -> None:
        items = normal_responses()
        queued_then_done = items[:4] + [items[6]]
        runner, _ = make_runner(FakeTransport(queued_then_done))
        with self.assertRaisesRegex(
            acceptance.AcceptanceFailure,
            "before any running status response",
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
        self.assertEqual(root.body_length, 44_729)
        self.assertEqual(
            root.sha256,
            "50eac58df6365f2b41864f3ae194891bf47a4e455668a3ae47aa1aaf728ad605",
        )
        self.assertEqual(root.etag, f'"{root.sha256}"')

    def test_startup_binding_verifies_uart_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path, raw_path, _ = write_live_startup_fixture(root)
            url, binding = acceptance._load_startup_binding(report_path)
            self.assertEqual(url, "http://192.168.1.10:80/")
            self.assertEqual(binding["uart_raw"], str(raw_path.resolve()))
            raw_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(
                acceptance.AcceptanceFailure, "raw UART SHA-256 mismatch"
            ):
                acceptance._load_startup_binding(report_path)

    def test_startup_binding_rejects_arbitrary_raw_with_matching_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "arbitrary"
            root.mkdir()
            report_path, raw_path, payload = write_live_startup_fixture(root)
            arbitrary = b"arbitrary bytes that are not a QWEB startup\r\n"
            raw_path.write_bytes(arbitrary)
            artifacts = payload["artifacts"]
            assert isinstance(artifacts, dict)
            artifacts["uart_raw_sha256"] = hashlib.sha256(arbitrary).hexdigest()
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                acceptance.AcceptanceFailure,
                "raw UART transcript verification failed",
            ):
                acceptance._load_startup_binding(report_path)

            external_root = parent / "external"
            external_root.mkdir()
            report_path, raw_path, payload = write_live_startup_fixture(
                external_root
            )
            outside_raw = parent / "outside_uart.bin"
            outside_raw.write_bytes(raw_path.read_bytes())
            artifacts = payload["artifacts"]
            assert isinstance(artifacts, dict)
            artifacts["uart_raw"] = str(outside_raw)
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                acceptance.AcceptanceFailure,
                "outside the startup evidence run",
            ):
                acceptance._load_startup_binding(report_path)

    def test_startup_binding_rejects_mutated_launch_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path, _, _ = write_live_startup_fixture(root)
            launch_path = root / "launch.json"
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            launch["audited"]["qweb_elf_sha256"] = "0" * 64
            launch_path.write_text(json.dumps(launch), encoding="utf-8")
            with self.assertRaisesRegex(
                acceptance.AcceptanceFailure,
                "launch report SHA-256 mismatch",
            ):
                acceptance._load_startup_binding(report_path)

    def test_startup_binding_rejects_fields_inconsistent_with_raw(self) -> None:
        mutations = (
            ("network", "board_ip", "192.168.1.11"),
            ("phy", "bmsr", 0x0025),
            ("tokenizer", "tokens", 151668),
            ("ddr_status", None, 0x4),
        )
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            for field, member, replacement in mutations:
                with self.subTest(field=field):
                    root = parent / field
                    root.mkdir()
                    report_path, _, payload = write_live_startup_fixture(root)
                    if member is None:
                        payload[field] = replacement
                    else:
                        nested = payload[field]
                        assert isinstance(nested, dict)
                        nested[member] = replacement
                    report_path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(
                        acceptance.AcceptanceFailure,
                        rf"reparsed raw UART field '{field}'",
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

            race_path = Path(temporary) / "race.json"
            original_link = acceptance.os.link

            def create_competing_target(source: object, destination: object) -> None:
                Path(destination).write_bytes(b"competing-writer")
                original_link(source, destination)

            with mock.patch.object(
                acceptance.os,
                "link",
                side_effect=create_competing_target,
            ):
                with self.assertRaisesRegex(
                    acceptance.AcceptanceFailure,
                    "refusing to overwrite",
                ):
                    acceptance._write_json_exclusive(
                        race_path,
                        {"winner": False},
                    )
            self.assertEqual(race_path.read_bytes(), b"competing-writer")


if __name__ == "__main__":
    unittest.main()
