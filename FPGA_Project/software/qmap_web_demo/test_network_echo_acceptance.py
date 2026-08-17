from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import run_network_echo_acceptance as gate1


HERE = Path(__file__).resolve().parent
FORMAL_WORKSPACE = Path(r"F:\vwk")
WRAPPER = HERE / "run_network_echo_board.ps1"


PASSING_TRANSCRIPT = (
    b"Zynq MP First Stage Boot Loader \r\n"
    b"\r-----lwIP TCP echo server ------\n\r"
    b"TCP packets sent to port 6001 will be echoed back\n\r"
    b"Detected Motorcomm YT8521 at PHY address 7 (id1=0x0000 id2=0x011A)\r\n"
    b"YT8521 phy_addr=7 chip_config=0x8160 mode=0 phy_type=rgmii-id\r\n"
    b"Start YT8521 UTP autonegotiation\r\n"
    b"YT8521 link resolved: bmsr=0x796D status=0xAC00 duplex=full\r\n"
    b"link speed for phy address 7: 1000\r\n"
    b"Board IP: 192.168.1.10\n\r"
    b"Netmask : 255.255.255.0\n\r"
    b"Gateway : 192.168.1.1\n\r"
    b"TCP echo server started @ port 7\n\r"
)


def run_wrapper(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WRAPPER),
            *arguments,
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )


class FakeProcess:
    pid = 12345
    returncode = 0

    def poll(self) -> int:
        return 0


class FakeUart:
    def __init__(self, chunks: list[bytes], events: list[str]) -> None:
        self.chunks = list(chunks)
        self.events = events

    def reset_input_buffer(self) -> None:
        self.events.append("reset")

    def read(self, _size: int) -> bytes:
        if self.chunks:
            return self.chunks.pop(0)
        return b""


class FakeSocket:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)
        self.sent = b""
        self.timeout: float | None = None

    def __enter__(self) -> "FakeSocket":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def settimeout(self, value: float) -> None:
        self.timeout = value

    def sendall(self, data: bytes) -> None:
        self.sent = data
        self.chunks = [data[:7], data[7:]]

    def recv(self, size: int) -> bytes:
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if len(chunk) <= size:
            return chunk
        self.chunks.insert(0, chunk[size:])
        return chunk[:size]


class NetworkEchoAcceptanceTests(unittest.TestCase):
    def test_passing_transcript_is_exactly_parsed(self) -> None:
        result = gate1.verify_startup_transcript(PASSING_TRANSCRIPT)
        self.assertEqual(
            [item["name"] for item in result["milestones"]],
            list(gate1.EchoStartupParser.MILESTONES),
        )
        self.assertEqual(result["phy"], {"address": 7, "id1": 0, "id2": 0x011A})
        self.assertEqual(result["link"]["speed_mbps"], 1000)
        self.assertEqual(result["link"]["duplex"], "full")
        self.assertEqual(result["network"], {"ip": "192.168.1.10", "port": 7})

    def test_parser_accepts_fragmented_lines(self) -> None:
        framer = gate1.LineFramer()
        parser = gate1.EchoStartupParser()
        for index in range(0, len(PASSING_TRANSCRIPT), 7):
            for line in framer.feed(PASSING_TRANSCRIPT[index : index + 7]):
                parser.process_line(line, "2026-08-16T00:00:00Z")
        self.assertIn(bytes(framer.pending), (b"", b"\r"))
        self.assertTrue(parser.complete)

    def test_parser_rejects_out_of_order_or_bad_link_records(self) -> None:
        cases = {
            "out of order": PASSING_TRANSCRIPT.replace(
                b"Detected Motorcomm YT8521 at PHY address 7 (id1=0x0000 id2=0x011A)\r\n",
                b"",
            ),
            "wrong PHY": PASSING_TRANSCRIPT.replace(
                b"PHY address 7", b"PHY address 1", 1
            ),
            "half duplex": PASSING_TRANSCRIPT.replace(b"duplex=full", b"duplex=half"),
            "100 Mbps": PASSING_TRANSCRIPT.replace(b": 1000\r\n", b": 100\r\n"),
            "bad IP": PASSING_TRANSCRIPT.replace(
                b"Board IP: 192.168.1.10", b"Board IP: 0.0.0.0"
            ),
        }
        for label, transcript in cases.items():
            with self.subTest(label=label), self.assertRaises(gate1.Gate1Failure):
                gate1.verify_startup_transcript(transcript)

    def test_parser_rejects_hardware_failure_and_unterminated_ready(self) -> None:
        failed = PASSING_TRANSCRIPT.split(
            b"YT8521 link resolved", maxsplit=1
        )[0] + b"YT8521 autonegotiation timeout: status=0x0000\r\n"
        with self.assertRaisesRegex(gate1.Gate1Failure, "UART reported failure"):
            gate1.verify_startup_transcript(failed)
        with self.assertRaisesRegex(gate1.Gate1Failure, "unterminated"):
            gate1.verify_startup_transcript(PASSING_TRANSCRIPT.rstrip(b"\r\n"))

    def test_serial_is_reset_before_and_immediately_after_launcher_start(self) -> None:
        events: list[str] = []
        uart = FakeUart([PASSING_TRANSCRIPT[:100], PASSING_TRANSCRIPT[100:]], events)
        raw = io.BytesIO()

        def start_launcher() -> FakeProcess:
            events.append("launch")
            return FakeProcess()

        with contextlib.redirect_stdout(io.StringIO()):
            process, startup, _started = gate1.capture_startup_after_launch(
                uart, raw, start_launcher, timeout=2.0
            )
        self.assertIsInstance(process, FakeProcess)
        self.assertEqual(events[:3], ["reset", "launch", "reset"])
        self.assertEqual(raw.getvalue(), PASSING_TRANSCRIPT)
        self.assertEqual(startup["network"]["ip"], "192.168.1.10")

    def test_launch_markers_must_be_complete_and_ordered(self) -> None:
        text = "\n".join(gate1.REQUIRED_LAUNCH_MARKERS)
        gate1._validate_ordered_markers(text, gate1.REQUIRED_LAUNCH_MARKERS)
        with self.assertRaises(gate1.Gate1Failure):
            gate1._validate_ordered_markers(
                "\n".join(reversed(gate1.REQUIRED_LAUNCH_MARKERS)),
                gate1.REQUIRED_LAUNCH_MARKERS,
            )

    def test_ping_requires_ten_address_bound_ttl_replies(self) -> None:
        ip = "192.168.1.10"
        success = "\r\n".join(
            f"Reply from {ip}: bytes=32 time<1ms TTL=64" for _ in range(10)
        ).encode("ascii")

        def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess([], 0, success, b"")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = gate1._run_ping(
                ip,
                "192.168.1.20",
                2000,
                root / "out.txt",
                root / "err.txt",
                runner=runner,
            )
            self.assertEqual(record["ttl_replies"], 10)
            self.assertEqual(record["source_ip"], "192.168.1.20")
            self.assertEqual(record["command"][2:4], ["-S", "192.168.1.20"])
            self.assertEqual((root / "out.txt").read_bytes(), success)

            def partial(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
                return subprocess.CompletedProcess([], 0, b"\n".join(success.splitlines()[:9]), b"")

            with self.assertRaisesRegex(gate1.Gate1Failure, "9/10"):
                gate1._run_ping(
                    ip,
                    "192.168.1.20",
                    2000,
                    root / "partial.txt",
                    root / "partial_err.txt",
                    runner=partial,
                )
        self.assertEqual(
            gate1._select_source_ip(ip, "192.168.1.20"), "192.168.1.20"
        )
        with self.assertRaisesRegex(gate1.Gate1Failure, "VPN/proxy tunnel"):
            gate1._select_source_ip(ip, "198.18.0.1")

    def test_tcp_echo_uses_unique_payload_and_requires_exact_bytes(self) -> None:
        fake = FakeSocket([])
        connector_arguments: dict[str, object] = {}

        def connector(*_args: object, **kwargs: object) -> FakeSocket:
            connector_arguments.update(kwargs)
            return fake

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = gate1._run_tcp_echo(
                "192.168.1.10",
                "192.168.1.20",
                "00000000-0000-0000-0000-000000000001",
                2.0,
                root / "tx.bin",
                root / "rx.bin",
                connector=connector,
            )
            self.assertTrue(record["exact_match"])
            self.assertEqual((root / "tx.bin").read_bytes(), (root / "rx.bin").read_bytes())
            self.assertIn(b"00000000-0000-0000-0000-000000000001", fake.sent)
            self.assertEqual(
                connector_arguments["source_address"], ("192.168.1.20", 0)
            )

    def test_child_environment_scrubs_xsdb_override_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary)
            environment = gate1._child_environment(profile)
        self.assertEqual(environment["PATH"], "")
        self.assertEqual(environment["HOME"], str(profile))
        self.assertEqual(environment["USERPROFILE"], str(profile))
        for name in ("RDI_SETUP_ENV_FUNCTION", "_RDI_BASELINE", "TCLLIBPATH", "QWEB_DEVICE_FILTER"):
            self.assertNotIn(name, environment)

    def test_failed_preflight_still_commits_structured_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "evidence"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                code = gate1.main(
                    [
                        "--port",
                        "COM_TEST",
                        "--workspace",
                        str(root / "not_vwc"),
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(code, 1)
            report = json.loads((output / "acceptance.json").read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertEqual(report["failure"]["stage"], "input_audit")
            self.assertTrue((output / "uart_raw.bin").is_file())
            self.assertTrue((output / "launcher_stdout.log").is_file())
            self.assertTrue((output / "launcher_stderr.log").is_file())

    def test_wrapper_audit_only_returns_before_xsdb_and_uses_no_ini(self) -> None:
        text = WRAPPER.read_text(encoding="utf-8")
        audit_return = text.index('Write-Host "PASS audit-only mode; XSDB was not invoked"')
        invocation = text.index("& $xsdbCommand @xsdbArguments")
        self.assertLess(audit_return, invocation)
        self.assertIn('$xsdbArguments = @("-no-ini", $launcher)', text)

    @unittest.skipUnless(
        (FORMAL_WORKSPACE / "network_workspace_manifest.json").is_file(),
        "formal F:\\vwc workspace is not present",
    )
    def test_formal_inputs_and_wrapper_audit_only_pass(self) -> None:
        audited = gate1.audit_formal_inputs(FORMAL_WORKSPACE)
        self.assertEqual(
            audited["network_manifest"]["sha256"],
            gate1.EXPECTED_INPUT_HASHES["network_manifest"],
        )
        result = run_wrapper("-Workspace", str(FORMAL_WORKSPACE), "-AuditOnly")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS audited network echo launch inputs", result.stdout)
        self.assertIn("PASS audit-only mode; XSDB was not invoked", result.stdout)
        self.assertNotIn("attempting to launch hw_server", result.stdout)


if __name__ == "__main__":
    unittest.main()
