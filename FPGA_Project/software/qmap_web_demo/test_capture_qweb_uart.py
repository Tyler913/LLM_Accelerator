#!/usr/bin/env python3
"""Offline tests for the strict QWEB UART startup capture."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import uuid

import capture_qweb_uart as capture


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


def passing_transcript(ip: str = "192.168.1.10") -> bytes:
    records = [
        "Qwen3-0.6B full28 PS Web demo",
        "QOT_BASEADDR=0x00000000a0040000",
        "Detected Motorcomm YT8521 at PHY address 7 (id1=0x0000 id2=0x011a)",
        "YT8521 phy_addr=7 chip_config=0x0000 mode=0 phy_type=rgmii",
        "Start YT8521 UTP autonegotiation",
        "YT8521 link resolved: bmsr=0x0024 status=0xac00 duplex=full",
        f"Board IP: {ip}",
        "Netmask : 255.255.255.0",
        "Gateway : 192.168.1.1",
        "DDR4 status=0x00000005",
        "TOKENIZER tokens=151669 model_vocab=151936 eos=151643 bytes=3629566",
        f"QWEB READY http://{ip}:80/ context=256 vocab=151936",
    ]
    # Exercise the AMD template's mixed LF/CR record boundaries.
    return ("\r\n" + "\n\r".join(records) + "\r\n").encode("utf-8")


class LineFramerTests(unittest.TestCase):
    def test_fragmented_mixed_newlines(self) -> None:
        framer = capture.LineFramer()
        lines: list[str] = []
        for chunk in (b"one\r", b"\ntw", b"o\n\rthr", b"ee"):
            lines.extend(framer.feed(chunk))
        lines.extend(framer.finish())
        self.assertEqual(lines, ["one", "two", "three"])

    def test_line_bound_is_fail_closed(self) -> None:
        framer = capture.LineFramer()
        with self.assertRaisesRegex(capture.StartupFailure, "line exceeds"):
            framer.feed(b"x" * (capture.MAX_LINE_BYTES + 1))


class StartupParserTests(unittest.TestCase):
    def test_exact_passing_transcript(self) -> None:
        report = capture.verify_transcript_bytes(passing_transcript())
        self.assertTrue(report["passed"])
        self.assertIsNone(report["failure"])
        self.assertEqual(
            [item["name"] for item in report["milestones"]],
            list(capture.MILESTONE_ORDER),
        )
        self.assertEqual(report["qot_baseaddr"], 0xA0040000)
        self.assertEqual(report["ddr_status"], 0x5)
        self.assertEqual(report["phy"]["address"], 7)
        self.assertEqual(report["phy"]["id2"], 0x011A)
        self.assertEqual(report["phy"]["speed_mbps"], 1000)
        self.assertEqual(report["network"]["board_ip"], "192.168.1.10")
        self.assertEqual(report["network"]["url"], "http://192.168.1.10:80/")

    def test_missing_phy_detection_is_rejected(self) -> None:
        data = passing_transcript().replace(
            b"Detected Motorcomm YT8521 at PHY address 7 (id1=0x0000 id2=0x011a)\n\r",
            b"",
        )
        report = capture.verify_transcript_bytes(data)
        self.assertFalse(report["passed"])
        self.assertIn("before required yt8521_detected", report["failure"])

    def test_wrong_phy_identity_is_rejected(self) -> None:
        data = passing_transcript().replace(b"id2=0x011a", b"id2=0x011b")
        report = capture.verify_transcript_bytes(data)
        self.assertFalse(report["passed"])
        self.assertIn("unexpected YT8521 identity", report["failure"])

    def test_half_duplex_is_rejected(self) -> None:
        data = passing_transcript().replace(b"duplex=full", b"duplex=half")
        report = capture.verify_transcript_bytes(data)
        self.assertFalse(report["passed"])
        self.assertIn("half duplex", report["failure"])

    def test_ready_ip_must_equal_board_ip(self) -> None:
        data = passing_transcript().replace(
            b"QWEB READY http://192.168.1.10:80/",
            b"QWEB READY http://192.168.1.11:80/",
        )
        report = capture.verify_transcript_bytes(data)
        self.assertFalse(report["passed"])
        self.assertIn("does not match Board IP", report["failure"])

    def test_network_failure_record_wins(self) -> None:
        prefix = passing_transcript().split(b"Start YT8521 UTP autonegotiation")[0]
        report = capture.verify_transcript_bytes(
            prefix + b"Start YT8521 UTP autonegotiation\r\nAuto negotiation error\r\n"
        )
        self.assertFalse(report["passed"])
        self.assertEqual(
            report["failure"], "UART failure record: Auto negotiation error"
        )

    def test_malformed_ready_is_rejected(self) -> None:
        data = passing_transcript().replace(
            b" context=256 vocab=151936", b" context=255 vocab=151936"
        )
        report = capture.verify_transcript_bytes(data)
        self.assertFalse(report["passed"])
        self.assertIn("unexpected QWEB limits", report["failure"])

    def test_duplicate_milestone_is_rejected(self) -> None:
        data = passing_transcript().replace(
            b"QOT_BASEADDR=0x00000000a0040000\n\r",
            b"QOT_BASEADDR=0x00000000a0040000\n\r"
            b"QOT_BASEADDR=0x00000000a0040000\n\r",
        )
        report = capture.verify_transcript_bytes(data)
        self.assertFalse(report["passed"])
        self.assertIn("duplicate startup milestone", report["failure"])


class CaptureCliReplayTests(unittest.TestCase):
    def test_launch_binding_pins_current_run_and_artifact_lineage(self) -> None:
        if not all(path.is_file() for path in PINNED_VITIS_FILES):
            self.skipTest("pinned Vitis 2025.1 XSDB toolchain is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary).resolve()
            (evidence / "xsdb_profile").mkdir()
            now = datetime.now(timezone.utc)
            run_id = str(uuid.uuid4())
            claim_payload = {
                "schema_version": 1,
                "tool": "run_qweb_board.ps1",
                "state": "claimed",
                "run_id": run_id,
                "started_utc": now.isoformat(),
                "evidence_directory": str(evidence),
            }
            claim_path = evidence / "launch.claim.json"
            claim_path.write_text(json.dumps(claim_payload), encoding="utf-8")
            log_path = evidence / "xsdb.log"
            log_path.write_text(
                "\n".join(capture.REQUIRED_XSDB_MARKERS) + "\n",
                encoding="utf-8",
            )
            payload = {
                "schema_version": 1,
                "tool": "run_qweb_board.ps1",
                "run_id": run_id,
                "passed": True,
                "failure": None,
                "started_utc": now.isoformat(),
                "finished_utc": (now + timedelta(milliseconds=1)).isoformat(),
                "evidence_directory": str(evidence),
                "xsdb_exit_code": 0,
                "claim": {
                    "path": "launch.claim.json",
                    "sha256": hashlib.sha256(claim_path.read_bytes()).hexdigest(),
                },
                "audited": capture.EXPECTED_LAUNCH_AUDIT,
                "launcher": {
                    "wrapper": str(WRAPPER),
                    "wrapper_sha256": hashlib.sha256(WRAPPER.read_bytes()).hexdigest(),
                    "tcl": str(TCL_LAUNCHER),
                    "tcl_sha256": hashlib.sha256(
                        TCL_LAUNCHER.read_bytes()
                    ).hexdigest(),
                    "xsdb": str(VITIS_XSDB),
                    "xsdb_sha256": hashlib.sha256(
                        VITIS_XSDB.read_bytes()
                    ).hexdigest(),
                    "loader": str(VITIS_LOADER),
                    "loader_sha256": hashlib.sha256(
                        VITIS_LOADER.read_bytes()
                    ).hexdigest(),
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
                        capture.EXPECTED_SANITIZED_XSDB_ENVIRONMENT
                    ),
                    "required_output_markers": list(capture.REQUIRED_XSDB_MARKERS),
                    "output_log": "xsdb.log",
                    "output_log_sha256": hashlib.sha256(
                        log_path.read_bytes()
                    ).hexdigest(),
                },
            }
            launch_path = evidence / "launch.json"
            launch_path.write_text(json.dumps(payload), encoding="utf-8")
            binding = capture._load_launch_binding(
                launch_path,
                output_dir=evidence,
                capture_started_utc=(now - timedelta(seconds=1)).isoformat(),
                uart_ready_utc=(now + timedelta(milliseconds=2)).isoformat(),
            )
            self.assertEqual(binding["run_id"], run_id)
            self.assertEqual(binding["audited"], capture.EXPECTED_LAUNCH_AUDIT)

            with self.assertRaisesRegex(
                capture.StartupFailure,
                "UART READY did not occur after this launch started",
            ):
                capture._load_launch_binding(
                    launch_path,
                    output_dir=evidence,
                    capture_started_utc=(now - timedelta(seconds=3)).isoformat(),
                    uart_ready_utc=(now - timedelta(seconds=2)).isoformat(),
                )

            if sys.platform == "win32":
                alias = str(evidence).lower()
                claim_payload["evidence_directory"] = alias
                claim_path.write_text(json.dumps(claim_payload), encoding="utf-8")
                payload["evidence_directory"] = alias
                payload["claim"]["sha256"] = hashlib.sha256(
                    claim_path.read_bytes()
                ).hexdigest()
                launch_path.write_text(json.dumps(payload), encoding="utf-8")
                case_binding = capture._load_launch_binding(
                    launch_path,
                    output_dir=evidence,
                    capture_started_utc=(now - timedelta(seconds=1)).isoformat(),
                    uart_ready_utc=(now + timedelta(milliseconds=2)).isoformat(),
                )
                self.assertEqual(case_binding["run_id"], run_id)

            log_path.write_text(
                "\n".join(reversed(capture.REQUIRED_XSDB_MARKERS)) + "\n",
                encoding="utf-8",
            )
            payload["launcher"]["output_log_sha256"] = hashlib.sha256(
                log_path.read_bytes()
            ).hexdigest()
            launch_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                capture.StartupFailure,
                "ordered PASS marker",
            ):
                capture._load_launch_binding(
                    launch_path,
                    output_dir=evidence,
                    capture_started_utc=(now - timedelta(seconds=1)).isoformat(),
                    uart_ready_utc=(now + timedelta(milliseconds=2)).isoformat(),
                )

            log_path.write_text(
                "\n".join(capture.REQUIRED_XSDB_MARKERS) + "\n",
                encoding="utf-8",
            )
            payload["launcher"]["output_log_sha256"] = hashlib.sha256(
                log_path.read_bytes()
            ).hexdigest()

            payload["audited"] = {
                **capture.EXPECTED_LAUNCH_AUDIT,
                "qweb_elf_sha256": "0" * 64,
            }
            launch_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                capture.StartupFailure,
                "pinned artifact lineage",
            ):
                capture._load_launch_binding(
                    launch_path,
                    output_dir=evidence,
                    capture_started_utc=(now - timedelta(seconds=1)).isoformat(),
                    uart_ready_utc=(now + timedelta(milliseconds=2)).isoformat(),
                )

    def test_live_timeout_default_covers_cold_jtag_load(self) -> None:
        args = capture.build_parser().parse_args(["--port", "COM230"])
        self.assertEqual(args.timeout, 3600.0)

    def test_replay_writes_bound_artifacts_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript = root / "input.bin"
            transcript.write_bytes(passing_transcript())
            evidence = root / "evidence"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = capture.main(
                    [
                        "--verify-transcript",
                        str(transcript),
                        "--output-dir",
                        str(evidence),
                    ]
                )
            self.assertEqual(result, 0, stderr.getvalue())
            raw = evidence / "uart_raw.bin"
            report_path = evidence / "startup.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(report["mode"], "transcript_replay")
            self.assertEqual(raw.read_bytes(), passing_transcript())
            self.assertEqual(
                report["artifacts"]["uart_raw_sha256"],
                hashlib.sha256(raw.read_bytes()).hexdigest(),
            )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                second = capture.main(
                    [
                        "--verify-transcript",
                        str(transcript),
                        "--output-dir",
                        str(evidence),
                    ]
                )
            self.assertEqual(second, 2)

            race_path = root / "race.json"
            original_link = capture.os.link

            def create_competing_target(source: object, destination: object) -> None:
                Path(destination).write_bytes(b"competing-writer")
                original_link(source, destination)

            with mock.patch.object(
                capture.os,
                "link",
                side_effect=create_competing_target,
            ):
                with self.assertRaisesRegex(
                    capture.StartupFailure,
                    "refusing to overwrite",
                ):
                    capture._write_json_exclusive(race_path, {"winner": False})
            self.assertEqual(race_path.read_bytes(), b"competing-writer")

    def test_failed_replay_still_writes_json_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript = root / "failed.bin"
            transcript.write_bytes(
                b"Qwen3-0.6B full28 PS Web demo\r\n"
                b"QOT_BASEADDR=0x00000000a0040000\r\n"
                b"Auto negotiation error\r\n"
            )
            evidence = root / "evidence"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = capture.main(
                    [
                        "--verify-transcript",
                        str(transcript),
                        "--output-dir",
                        str(evidence),
                    ]
                )
            self.assertEqual(result, 1)
            report = json.loads((evidence / "startup.json").read_text("utf-8"))
            self.assertFalse(report["passed"])
            self.assertIn("Auto negotiation error", report["failure"])

    def test_invalid_live_serial_writes_atomic_empty_failure_evidence(self) -> None:
        class FakeSerialException(Exception):
            pass

        serial_module = types.ModuleType("serial")
        serial_module.EIGHTBITS = 8
        serial_module.PARITY_NONE = "N"
        serial_module.STOPBITS_ONE = 1
        serial_module.SerialException = FakeSerialException

        def reject_serial(**_kwargs: object) -> object:
            raise FakeSerialException("could not open port COM404")

        serial_module.Serial = reject_serial

        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            with mock.patch.dict(sys.modules, {"serial": serial_module}):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    result = capture.main(
                        [
                            "--port",
                            "COM404",
                            "--output-dir",
                            str(evidence),
                            "--launch-json",
                            str(evidence / "launch.json"),
                        ]
                    )

            self.assertEqual(result, 2)
            raw = evidence / "uart_raw.bin"
            report_path = evidence / "startup.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["mode"], "live")
            self.assertFalse(report["passed"])
            self.assertEqual(report["error"]["kind"], "capture_exception")
            self.assertEqual(report["error"]["type"], "StartupFailure")
            self.assertIn("could not open port COM404", report["error"]["reason"])
            self.assertEqual(report["artifacts"]["uart_raw"], "uart_raw.bin")
            self.assertEqual(report["artifacts"]["uart_raw_bytes"], 0)
            self.assertEqual(
                report["artifacts"]["uart_raw_sha256"],
                hashlib.sha256(b"").hexdigest(),
            )
            self.assertEqual(raw.read_bytes(), b"")
            self.assertFalse((evidence / "startup.json.tmp").exists())

    def test_unexpected_capture_exception_preserves_partial_raw_evidence(self) -> None:
        partial = b"partial UART before disconnect\r\n"

        def crash_capture(
            _port: str,
            raw_stream: object,
            **_kwargs: object,
        ) -> dict[str, object]:
            raw_stream.write(partial)
            raw_stream.flush()
            raise RuntimeError("simulated capture_live crash")

        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            with mock.patch.object(capture, "capture_live", side_effect=crash_capture):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    result = capture.main(
                        [
                            "--port",
                            "COM230",
                            "--output-dir",
                            str(evidence),
                            "--launch-json",
                            str(evidence / "launch.json"),
                        ]
                    )

            self.assertEqual(result, 2)
            raw = evidence / "uart_raw.bin"
            report = json.loads(
                (evidence / "startup.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["mode"], "live")
            self.assertFalse(report["passed"])
            self.assertEqual(report["error"]["kind"], "capture_exception")
            self.assertEqual(report["error"]["type"], "RuntimeError")
            self.assertEqual(
                report["error"]["reason"], "simulated capture_live crash"
            )
            self.assertEqual(raw.read_bytes(), partial)
            self.assertEqual(report["artifacts"]["uart_raw_bytes"], len(partial))
            self.assertEqual(
                report["artifacts"]["uart_raw_sha256"],
                hashlib.sha256(partial).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
