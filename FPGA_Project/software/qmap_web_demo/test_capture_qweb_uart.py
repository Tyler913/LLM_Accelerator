#!/usr/bin/env python3
"""Offline tests for the strict QWEB UART startup capture."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import capture_qweb_uart as capture


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


if __name__ == "__main__":
    unittest.main()
