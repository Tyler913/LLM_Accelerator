#!/usr/bin/env python3
"""Capture and verify the bounded UART startup of the board-hosted QWEB demo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import BinaryIO, Callable, Iterable, Sequence


REPORT_SCHEMA_VERSION = 1
BAUD_RATE = 115_200
MAX_CAPTURE_BYTES = 4 * 1024 * 1024
MAX_CAPTURE_LINES = 4096
MAX_LINE_BYTES = 4096
EXPECTED_PHY_ADDRESS = 7
EXPECTED_PHY_ID1 = 0x0000
EXPECTED_PHY_ID2 = 0x011A
EXPECTED_QOT_BASEADDR = 0xA004_0000
EXPECTED_DDR_STATUS = 0x0000_0005
EXPECTED_TOKEN_COUNT = 151_669
EXPECTED_MODEL_VOCAB = 151_936
EXPECTED_EOS_TOKEN = 151_643
EXPECTED_TOKENIZER_BYTES = 3_629_566
EXPECTED_CONTEXT = 256
EXPECTED_HTTP_PORT = 80

APP_HEADER = "Qwen3-0.6B full28 PS Web demo"

PHY_RE = re.compile(
    r"^Detected Motorcomm YT8521 at PHY address (?P<address>[0-9]+) "
    r"\(id1=0x(?P<id1>[0-9a-fA-F]{4}) id2=0x(?P<id2>[0-9a-fA-F]{4})\)$"
)
LINK_RE = re.compile(
    r"^YT8521 link resolved: bmsr=0x(?P<bmsr>[0-9a-fA-F]{4}) "
    r"status=0x(?P<status>[0-9a-fA-F]{4}) duplex=(?P<duplex>full|half)$"
)
BOARD_IP_RE = re.compile(r"^Board IP:\s*(?P<ip>[0-9]+(?:\.[0-9]+){3})$")
QOT_BASE_RE = re.compile(r"^QOT_BASEADDR=0x(?P<address>[0-9a-fA-F]{16})$")
DDR_RE = re.compile(r"^DDR4 status=0x(?P<status>[0-9a-fA-F]{8})$")
TOKENIZER_RE = re.compile(
    r"^TOKENIZER tokens=(?P<tokens>[0-9]+) model_vocab=(?P<vocab>[0-9]+) "
    r"eos=(?P<eos>[0-9]+) bytes=(?P<bytes>[0-9]+)$"
)
READY_RE = re.compile(
    r"^QWEB READY http://(?P<ip>[0-9]+(?:\.[0-9]+){3}):(?P<port>[0-9]+)/ "
    r"context=(?P<context>[0-9]+) vocab=(?P<vocab>[0-9]+)$"
)

FAILURE_PREFIXES = (
    "Error ",
    "Error adding N/W interface",
    "Error creating PCB",
    "Unable to bind to port",
    "Out of memory while tcp_listen",
    "Auto negotiation error",
    "Phy setup error",
    "Phy setup failure",
    "Ethernet Link down",
    "YT8521 software reset timeout",
    "YT8521 extended-register initialization failed",
    "YT8521 autonegotiation timeout",
    "YT8521 reported invalid speed status",
    "Unsupported PHY initialization",
    "WARNING: Unsupported Ethernet PHY",
    "Link error",
    "Clock Divisors incorrect",
    "ERROR ",
    "FAIL",
)

MILESTONE_ORDER = (
    "app_header",
    "qot_baseaddr",
    "yt8521_detected",
    "yt8521_link_resolved",
    "board_ip",
    "ddr_ready",
    "tokenizer_ready",
    "http_ready",
)


class StartupFailure(RuntimeError):
    """Raised when UART startup evidence is incomplete or inconsistent."""


@dataclass
class LineFramer:
    """Frame UART text on CR, LF, CRLF, or LFCR without losing raw bytes."""

    pending: bytearray = field(default_factory=bytearray)

    def feed(self, data: bytes) -> list[str]:
        lines: list[str] = []
        for value in data:
            if value in (0x0A, 0x0D):
                if self.pending:
                    lines.append(self._take_line())
                continue
            if len(self.pending) >= MAX_LINE_BYTES:
                raise StartupFailure(
                    f"UART line exceeds the {MAX_LINE_BYTES}-byte bound"
                )
            self.pending.append(value)
        return lines

    def finish(self) -> list[str]:
        return [self._take_line()] if self.pending else []

    def _take_line(self) -> str:
        raw = bytes(self.pending)
        self.pending.clear()
        return raw.decode("utf-8", errors="replace").lstrip("\ufeff")


@dataclass
class QwebStartupParser:
    """Recognize one exact, ordered QWEB startup transcript."""

    lines: list[str] = field(default_factory=list)
    milestones: list[dict[str, object]] = field(default_factory=list)
    next_milestone: int = 0
    failure: str | None = None
    phy: dict[str, object] | None = None
    network: dict[str, object] | None = None
    qot_baseaddr: int | None = None
    ddr_status: int | None = None
    tokenizer: dict[str, int] | None = None

    @property
    def ready(self) -> bool:
        return self.next_milestone == len(MILESTONE_ORDER) and self.failure is None

    def observe(self, line: str) -> None:
        normalized = line.strip()
        if not normalized:
            return
        if len(self.lines) >= MAX_CAPTURE_LINES:
            self._fail(f"UART transcript exceeds {MAX_CAPTURE_LINES} lines")
            return
        self.lines.append(normalized)
        line_number = len(self.lines)

        failure = self._failure_record(normalized)
        if failure is not None:
            self._fail(f"UART failure record: {failure}")
            return

        milestone = self._parse_milestone(normalized)
        if milestone is None:
            return
        name, details = milestone
        if self.ready:
            self._fail(f"duplicate startup milestone after READY: {name}")
            return
        expected = MILESTONE_ORDER[self.next_milestone]
        if name != expected:
            if name in MILESTONE_ORDER[: self.next_milestone]:
                self._fail(f"duplicate startup milestone: {name}")
            else:
                self._fail(
                    f"startup milestone {name} arrived before required {expected}"
                )
            return
        self.milestones.append(
            {"name": name, "line_number": line_number, "line": normalized}
        )
        self.next_milestone += 1
        self._commit(name, details)

    def finish(self) -> None:
        if self.failure is not None or self.ready:
            return
        missing = ", ".join(MILESTONE_ORDER[self.next_milestone :])
        self._fail(f"startup ended before required milestones: {missing}")

    def _fail(self, message: str) -> None:
        if self.failure is None:
            self.failure = message

    @staticmethod
    def _failure_record(line: str) -> str | None:
        for prefix in FAILURE_PREFIXES:
            if line.startswith(prefix):
                return line
        return None

    def _parse_milestone(
        self, line: str
    ) -> tuple[str, dict[str, object]] | None:
        if line == APP_HEADER:
            return "app_header", {}
        if line.startswith("QOT_BASEADDR="):
            match = QOT_BASE_RE.fullmatch(line)
            if match is None:
                self._fail(f"malformed QOT_BASEADDR record: {line}")
                return None
            address = int(match.group("address"), 16)
            if address != EXPECTED_QOT_BASEADDR:
                self._fail(
                    "unexpected QOT base address: "
                    f"0x{address:016x} != 0x{EXPECTED_QOT_BASEADDR:016x}"
                )
                return None
            return "qot_baseaddr", {"address": address}
        if line.startswith("Detected Motorcomm YT8521"):
            match = PHY_RE.fullmatch(line)
            if match is None:
                self._fail(f"malformed YT8521 detection record: {line}")
                return None
            details = {
                "address": int(match.group("address"), 10),
                "id1": int(match.group("id1"), 16),
                "id2": int(match.group("id2"), 16),
            }
            if (
                details["address"] != EXPECTED_PHY_ADDRESS
                or details["id1"] != EXPECTED_PHY_ID1
                or details["id2"] != EXPECTED_PHY_ID2
            ):
                self._fail(
                    "unexpected YT8521 identity: "
                    f"address={details['address']} id1=0x{details['id1']:04x} "
                    f"id2=0x{details['id2']:04x}"
                )
                return None
            return "yt8521_detected", details
        if line.startswith("YT8521 link resolved:"):
            match = LINK_RE.fullmatch(line)
            if match is None:
                self._fail(f"malformed YT8521 link record: {line}")
                return None
            bmsr = int(match.group("bmsr"), 16)
            status = int(match.group("status"), 16)
            duplex = match.group("duplex")
            if (bmsr & 0x0024) != 0x0024:
                self._fail(
                    f"YT8521 BMSR lacks link/autoneg-complete bits: 0x{bmsr:04x}"
                )
                return None
            if (status & 0x0C00) != 0x0C00:
                self._fail(
                    f"YT8521 status lacks link/resolved bits: 0x{status:04x}"
                )
                return None
            if duplex != "full":
                self._fail("YT8521 negotiated half duplex")
                return None
            speed_mbps = {0x0000: 10, 0x4000: 100, 0x8000: 1000}.get(
                status & 0xC000
            )
            if speed_mbps is None:
                self._fail(f"YT8521 reported invalid speed bits: 0x{status:04x}")
                return None
            return "yt8521_link_resolved", {
                "bmsr": bmsr,
                "status": status,
                "duplex": duplex,
                "speed_mbps": speed_mbps,
            }
        if line.startswith("Board IP:"):
            match = BOARD_IP_RE.fullmatch(line)
            if match is None:
                self._fail(f"malformed Board IP record: {line}")
                return None
            try:
                address = ipaddress.IPv4Address(match.group("ip"))
            except ipaddress.AddressValueError:
                self._fail(f"invalid Board IP address: {match.group('ip')}")
                return None
            if address.is_unspecified or address.is_multicast or address.is_loopback:
                self._fail(f"Board IP is not a usable unicast address: {address}")
                return None
            return "board_ip", {"ip": str(address)}
        if line.startswith("DDR4 status="):
            match = DDR_RE.fullmatch(line)
            if match is None:
                self._fail(f"malformed DDR4 status record: {line}")
                return None
            status = int(match.group("status"), 16)
            if status != EXPECTED_DDR_STATUS:
                self._fail(
                    f"DDR4 status is 0x{status:08x}, expected 0x{EXPECTED_DDR_STATUS:08x}"
                )
                return None
            return "ddr_ready", {"status": status}
        if line.startswith("TOKENIZER tokens="):
            match = TOKENIZER_RE.fullmatch(line)
            if match is None:
                self._fail(f"malformed TOKENIZER record: {line}")
                return None
            details = {name: int(match.group(name), 10) for name in (
                "tokens", "vocab", "eos", "bytes"
            )}
            expected = {
                "tokens": EXPECTED_TOKEN_COUNT,
                "vocab": EXPECTED_MODEL_VOCAB,
                "eos": EXPECTED_EOS_TOKEN,
                "bytes": EXPECTED_TOKENIZER_BYTES,
            }
            if details != expected:
                self._fail(f"unexpected tokenizer metadata: {details} != {expected}")
                return None
            return "tokenizer_ready", details
        if line.startswith("QWEB READY"):
            match = READY_RE.fullmatch(line)
            if match is None:
                self._fail(f"malformed QWEB READY record: {line}")
                return None
            try:
                address = ipaddress.IPv4Address(match.group("ip"))
            except ipaddress.AddressValueError:
                self._fail(f"invalid READY IP address: {match.group('ip')}")
                return None
            port = int(match.group("port"), 10)
            context = int(match.group("context"), 10)
            vocab = int(match.group("vocab"), 10)
            board_ip = None if self.network is None else self.network.get("board_ip")
            if str(address) != board_ip:
                self._fail(
                    f"QWEB READY IP {address} does not match Board IP {board_ip}"
                )
                return None
            if port != EXPECTED_HTTP_PORT:
                self._fail(f"QWEB READY port {port} is not {EXPECTED_HTTP_PORT}")
                return None
            if context != EXPECTED_CONTEXT or vocab != EXPECTED_MODEL_VOCAB:
                self._fail(
                    "unexpected QWEB limits: "
                    f"context={context} vocab={vocab}"
                )
                return None
            return "http_ready", {
                "ip": str(address),
                "port": port,
                "context": context,
                "vocab": vocab,
                "url": f"http://{address}:{port}/",
            }
        return None

    def _commit(self, name: str, details: dict[str, object]) -> None:
        if name == "qot_baseaddr":
            self.qot_baseaddr = int(details["address"])
        elif name == "yt8521_detected":
            self.phy = dict(details)
        elif name == "yt8521_link_resolved":
            if self.phy is None:
                self._fail("internal parser error: link before PHY")
            else:
                self.phy.update(details)
        elif name == "board_ip":
            self.network = {"board_ip": details["ip"]}
        elif name == "ddr_ready":
            self.ddr_status = int(details["status"])
        elif name == "tokenizer_ready":
            self.tokenizer = {key: int(value) for key, value in details.items()}
        elif name == "http_ready":
            if self.network is None:
                self._fail("internal parser error: READY before Board IP")
            else:
                self.network.update(details)

    def result(self) -> dict[str, object]:
        return {
            "passed": self.ready and self.failure is None,
            "failure": self.failure,
            "lines": list(self.lines),
            "milestones": list(self.milestones),
            "missing_milestones": list(MILESTONE_ORDER[self.next_milestone :]),
            "qot_baseaddr": self.qot_baseaddr,
            "ddr_status": self.ddr_status,
            "phy": self.phy,
            "tokenizer": self.tokenizer,
            "network": self.network,
        }


def verify_transcript_bytes(data: bytes) -> dict[str, object]:
    """Parse a captured UART transcript without accessing a serial port."""

    if len(data) > MAX_CAPTURE_BYTES:
        raise StartupFailure(
            f"UART transcript exceeds the {MAX_CAPTURE_BYTES}-byte bound"
        )
    parser = QwebStartupParser()
    framer = LineFramer()
    for line in framer.feed(data):
        parser.observe(line)
        if parser.failure is not None:
            break
    if parser.failure is None:
        for line in framer.finish():
            parser.observe(line)
    parser.finish()
    return parser.result()


def capture_live(
    port: str,
    raw_stream: BinaryIO,
    *,
    timeout: float,
    linger: float,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Open a UART and capture one startup. Imported lazily for host tests."""

    try:
        import serial  # type: ignore[import-not-found]
    except ImportError as error:
        raise StartupFailure("pyserial is required for live UART capture") from error

    parser = QwebStartupParser()
    framer = LineFramer()
    deadline = monotonic() + timeout
    ready_at: float | None = None
    captured_bytes = 0
    try:
        with serial.Serial(
            port=port,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        ) as uart:
            uart.reset_input_buffer()
            while monotonic() < deadline:
                data = uart.read(4096)
                if data:
                    captured_bytes += len(data)
                    if captured_bytes > MAX_CAPTURE_BYTES:
                        parser._fail(
                            f"UART capture exceeds the {MAX_CAPTURE_BYTES}-byte bound"
                        )
                        break
                    raw_stream.write(data)
                    raw_stream.flush()
                    for line in framer.feed(data):
                        print(line, flush=True)
                        parser.observe(line)
                        if parser.failure is not None:
                            break
                    if parser.ready and ready_at is None:
                        ready_at = monotonic()
                if parser.failure is not None:
                    break
                if ready_at is not None and monotonic() - ready_at >= linger:
                    break
    except (OSError, serial.SerialException) as error:
        raise StartupFailure(f"UART capture failed: {error}") from error

    if parser.failure is None:
        for line in framer.finish():
            print(line, flush=True)
            parser.observe(line)
    parser.finish()
    return parser.result()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(tempfile.gettempdir()) / "qweb_uart_startup" / stamp


def _prepare_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved.mkdir()
    except FileExistsError as error:
        raise StartupFailure(
            f"refusing to overwrite existing evidence directory: {resolved}"
        ) from error
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise StartupFailure(f"refusing to overwrite evidence artifact: {path}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    if path.exists():
        raise StartupFailure(f"refusing to overwrite evidence artifact: {path}")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--port", help="live UART port, for example COM230")
    source.add_argument(
        "--verify-transcript",
        type=Path,
        help="verify existing raw UART bytes without opening a serial port",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="live startup timeout including cold JTAG load (default/max 3600s)",
    )
    parser.add_argument(
        "--linger",
        type=float,
        default=2.0,
        help="capture duration after READY in seconds (default 2, maximum 10)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (0.0 < args.timeout <= 3600.0):
        print("ERROR: --timeout must be > 0 and <= 3600 seconds", file=sys.stderr)
        return 2
    if not (0.0 <= args.linger <= 10.0):
        print("ERROR: --linger must be between 0 and 10 seconds", file=sys.stderr)
        return 2

    output_dir: Path | None = None
    report_path: Path | None = None
    try:
        transcript_data: bytes | None = None
        transcript_path: Path | None = None
        if args.verify_transcript is not None:
            transcript_path = args.verify_transcript.resolve(strict=True)
            transcript_data = transcript_path.read_bytes()
        output_dir = _prepare_output_dir(args.output_dir or _default_output_dir())
        raw_path = output_dir / "uart_raw.bin"
        report_path = output_dir / "startup.json"
        started_utc = _utc_now()

        if transcript_data is not None:
            with raw_path.open("xb") as stream:
                stream.write(transcript_data)
            result = verify_transcript_bytes(transcript_data)
            mode = "transcript_replay"
        else:
            with raw_path.open("xb") as stream:
                result = capture_live(
                    args.port,
                    stream,
                    timeout=args.timeout,
                    linger=args.linger,
                )
            mode = "live"

        report: dict[str, object] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "tool": Path(__file__).name,
            "mode": mode,
            "started_utc": started_utc,
            "finished_utc": _utc_now(),
            **result,
            "serial": {
                "port": args.port if mode == "live" else None,
                "baud_rate": BAUD_RATE,
            },
            "input_transcript": (
                str(transcript_path) if transcript_path is not None else None
            ),
            "artifacts": {
                "uart_raw": str(raw_path),
                "uart_raw_bytes": raw_path.stat().st_size,
                "uart_raw_sha256": _sha256_file(raw_path),
                "report": str(report_path),
            },
        }
        _write_json_exclusive(report_path, report)
    except (StartupFailure, OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    if report["passed"]:
        network = report["network"]
        assert isinstance(network, dict)
        print(f"PASS QWEB UART startup: {network['url']}")
        print(f"report: {report_path}")
        return 0
    print(f"FAIL: {report['failure']}", file=sys.stderr)
    print(f"report: {report_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
