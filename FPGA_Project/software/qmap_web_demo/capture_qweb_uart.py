#!/usr/bin/env python3
"""Capture and verify the bounded UART startup of the board-hosted QWEB demo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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
import uuid


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
LAUNCH_REPORT_WAIT_SECONDS = 30.0
EXPECTED_VITIS_ROOT = Path(
    r"D:\Applications\Vivado_2025.1.1\2025.1.1\Vitis"
)
EXPECTED_WRAPPER_SHA256 = (
    "49ecb6801b7b08f8072710d468dfa1732cd2ded368229db114d0b560e94b8930"
)
EXPECTED_TCL_LAUNCHER_SHA256 = (
    "3ef8fb39f94204bc5bf4645df853665e0d11d3059ab96f6b0e508f10bae01931"
)
EXPECTED_LAUNCH_AUDIT: dict[str, object] = {
    "network_manifest_sha256": "33ca1a825aff72dd7a59c9938c0b0e838062c5c6e18ee1826432341e7a85e401",
    "network_bit_sha256": "c926b3db8021e976e5ea6cc2f71da3c44899ea5c3df61da573475ce6d8c21239",
    "network_xsa_sha256": "0af1257442a68a6beb31d94811713f2ae8e6af63e0a85dd497146405e37406cf",
    "network_fsbl_sha256": "a7695cd19bec264ca0ef1d951527587cc02489a82711ab63f4534c5f3579f5cb",
    "qweb_elf_sha256": "38a772f093ce3996177640863888b6700f1ac26f7bcb82e5f2159c0ef46f89da",
    "workbench_manifest_sha256": "f9d523ab59926c7583f76e6faefc941d019b296ed5f51d00f49cf6627bc0eda7",
    "runtime_manifest_sha256": "fa8981e71101def29970135df5e863da5634274dd9fb64905666f0cf1d47d3f2",
    "runtime_loader_sha256": "0b698f28165f05efe7dbc2a8c64bfa7379cbf10cc8dcb4b1ced9660eeabbecb1",
    "runtime_segments": 61,
    "runtime_bytes": 394_547_200,
    "tokenizer_sha256": "c20242603ef4144e3f3f2ec4ba97c0e9c315aadd41f1bd2c5740e2a7ffa03a7d",
    "tokenizer_bytes": 3_629_566,
    "xsdb_sha256": "2cd1d63586a95fbb00b2badb6652cae485bdd4cbdab612c8a5de0afd7e874f59",
    "loader_sha256": "13fac52e8f0c81ea0539a0edddf8ad839d5c8a49d9fe37531abffbab8decfb38",
    "setup_env_sha256": "0f8108b67a75b3957b075c3f87d094fcc63eb7251a27281e7d456024898d881f",
    "rdi_args_sha256": "44f7765ce290ea0546f441044a5303316f7fe6c9434136ba013e8f72c5e1f544",
    "xsdb_exe_sha256": "9ef5fe040728b508a1a0efd1dfd83dbb3df02d2d9b59310ea0e7b403cd429483",
    "xsdb_manifest_sha256": "77596430e6ea9d0a8d7b686d75f0b554564fdc0a931d1f806cf16c0e100a99e3",
    "vitis_root": str(EXPECTED_VITIS_ROOT),
}
REQUIRED_XSDB_MARKERS = (
    "PASS programmed audited network bitstream",
    "PASS FSBL initialization",
    "PASS PL DDR4 ready status=0x00000005",
    "PASS all 281 QMAP packet headers",
    "PASS a_qweb downloaded and running after audited runtime load",
)
EXPECTED_SANITIZED_XSDB_ENVIRONMENT = (
    "PATH",
    "PYTHONPATH",
    "HOME",
    "HOMEDRIVE",
    "HOMEPATH",
    "HOMESHARE",
    "USERPROFILE",
    "MYXILINX",
    "MYVIVADO",
    "XILINX",
    "XILINX_PATH",
    "XIL_NO_OVERRIDE",
    "XIL_PA_NO_DEFAULT_OVERRIDE",
    "XIL_PA_NO_XILINX_OVERRIDE",
    "XIL_PA_NO_XILINX_SDK_OVERRIDE",
    "XIL_PA_NO_XILINX_PATH_OVERRIDE",
    "XILINX_VITIS",
    "XILINX_VIVADO",
    "XILINX_SDK",
    "XILINX_HLS",
    "XILINX_VCXX",
    "XILINX_COMMON_TOOLS",
    "XIL_TPS_ROOT",
    "RDI_PATCHROOT",
    "_RDI_BASELINE",
    "_RDI_NEEDS_PYTHON",
    "RDI_BASELINE",
    "RDI_PREPEND_PATH",
    "RDI_MIXED_EXT",
    "RDI_DEPENDENCY",
    "RDI_BYPASS_ARGS",
    "RDI_PROG",
    "RDI_ARGS",
    "RDI_ARGS_FUNCTION",
    "RDI_SETUP_ENV_FUNCTION",
    "RDI_JAVALAUNCH",
    "RDI_VBSLAUNCH",
    "RDI_EXECCLASS",
    "RDI_CLASSPATH",
    "RDI_JAVAARGS",
    "RDI_JAVAFXROOT",
    "RDI_JAVACEFROOT",
    "RDI_APPROOT",
    "RDI_BINROOT",
    "RDI_INSTALLROOT",
    "RDI_PLATFORM",
    "RDI_OPT_EXT",
    "_RDI_SETENV_RUN",
    "QWEB_HW_SERVER_URL",
    "QWEB_DEVICE_FILTER",
    "TCLLIBPATH",
    "TCL_LIBRARY",
    "TK_LIBRARY",
)

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
            "uart_ready_utc": None,
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
    ready_utc: str | None = None
    last_data_utc: str | None = None
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
                    last_data_utc = _utc_now()
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
                        ready_utc = last_data_utc
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
    if parser.ready and ready_utc is None:
        ready_utc = last_data_utc
    result = parser.result()
    result["uart_ready_utc"] = ready_utc
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise StartupFailure(f"{label} is not a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise StartupFailure(f"{label} is not an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise StartupFailure(f"{label} must include an explicit UTC offset")
    return parsed


def _load_launch_binding(
    path: Path,
    *,
    output_dir: Path,
    capture_started_utc: str,
    uart_ready_utc: object,
) -> dict[str, object]:
    """Bind this UART capture to one immutable, audited physical launch report."""

    expected_path = (output_dir / "launch.json").resolve()
    requested_path = path.resolve()
    if requested_path != expected_path:
        raise StartupFailure(
            "--launch-json must be the launch.json inside --output-dir"
        )
    deadline = time.monotonic() + LAUNCH_REPORT_WAIT_SECONDS
    while not requested_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not requested_path.is_file():
        raise StartupFailure(
            f"audited QWEB launch report did not appear within "
            f"{LAUNCH_REPORT_WAIT_SECONDS:.0f}s: {requested_path}"
        )
    raw = requested_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StartupFailure(f"QWEB launch report is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise StartupFailure("QWEB launch report must be one JSON object")
    run_id = payload.get("run_id")
    try:
        canonical_run_id = str(uuid.UUID(run_id)) if isinstance(run_id, str) else ""
    except ValueError as error:
        raise StartupFailure("QWEB launch report run_id is not a UUID") from error
    if canonical_run_id != run_id:
        raise StartupFailure("QWEB launch report run_id is not canonical")
    audited = payload.get("audited")
    evidence_value = payload.get("evidence_directory")
    evidence_matches = (
        isinstance(evidence_value, str)
        and Path(evidence_value).resolve() == output_dir.resolve()
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("tool") != "run_qweb_board.ps1"
        or payload.get("passed") is not True
        or payload.get("failure") is not None
        or payload.get("xsdb_exit_code") != 0
        or not evidence_matches
        or audited != EXPECTED_LAUNCH_AUDIT
    ):
        raise StartupFailure(
            "QWEB launch report is not a passing launch of the pinned artifact lineage"
        )
    claim_record = payload.get("claim")
    launcher = payload.get("launcher")
    if not isinstance(claim_record, dict) or not isinstance(launcher, dict):
        raise StartupFailure("QWEB launch report lacks claim/tool provenance")
    claim_path = (output_dir / "launch.claim.json").resolve(strict=True)
    if claim_record.get("path") != "launch.claim.json":
        raise StartupFailure("QWEB launch report claim path is not canonical")
    claim_raw = claim_path.read_bytes()
    if claim_record.get("sha256") != hashlib.sha256(claim_raw).hexdigest():
        raise StartupFailure("QWEB launch claim SHA-256 mismatch")
    try:
        claim = json.loads(claim_raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StartupFailure(f"QWEB launch claim is not valid JSON: {error}") from error
    claim_evidence = claim.get("evidence_directory") if isinstance(claim, dict) else None
    if (
        not isinstance(claim, dict)
        or claim.get("schema_version") != 1
        or claim.get("tool") != "run_qweb_board.ps1"
        or claim.get("state") != "claimed"
        or claim.get("run_id") != run_id
        or claim.get("started_utc") != payload.get("started_utc")
        or not isinstance(claim_evidence, str)
        or Path(claim_evidence).resolve() != output_dir.resolve()
    ):
        raise StartupFailure("QWEB launch claim is inconsistent with launch.json")
    if (
        launcher.get("wrapper_sha256") != EXPECTED_WRAPPER_SHA256
        or launcher.get("tcl_sha256") != EXPECTED_TCL_LAUNCHER_SHA256
        or launcher.get("xsdb_sha256") != EXPECTED_LAUNCH_AUDIT["xsdb_sha256"]
        or launcher.get("loader_sha256")
        != EXPECTED_LAUNCH_AUDIT["loader_sha256"]
        or launcher.get("setup_env_sha256")
        != EXPECTED_LAUNCH_AUDIT["setup_env_sha256"]
        or launcher.get("rdi_args_sha256")
        != EXPECTED_LAUNCH_AUDIT["rdi_args_sha256"]
        or launcher.get("xsdb_exe_sha256")
        != EXPECTED_LAUNCH_AUDIT["xsdb_exe_sha256"]
        or launcher.get("xsdb_manifest_sha256")
        != EXPECTED_LAUNCH_AUDIT["xsdb_manifest_sha256"]
        or launcher.get("required_output_markers") != list(REQUIRED_XSDB_MARKERS)
        or launcher.get("output_log") != "xsdb.log"
    ):
        raise StartupFailure("QWEB launch report lacks pinned XSDB marker provenance")
    for path_name, hash_name in (
        ("wrapper", "wrapper_sha256"),
        ("tcl", "tcl_sha256"),
        ("xsdb", "xsdb_sha256"),
        ("loader", "loader_sha256"),
        ("setup_env", "setup_env_sha256"),
        ("rdi_args", "rdi_args_sha256"),
        ("xsdb_exe", "xsdb_exe_sha256"),
        ("xsdb_manifest", "xsdb_manifest_sha256"),
    ):
        tool_value = launcher.get(path_name)
        tool_hash = launcher.get(hash_name)
        if not isinstance(tool_value, str) or not isinstance(tool_hash, str):
            raise StartupFailure(f"QWEB launch report lacks {path_name} provenance")
        tool_path = Path(tool_value).resolve(strict=True)
        if _sha256_file(tool_path) != tool_hash:
            raise StartupFailure(f"QWEB launch {path_name} SHA-256 mismatch")
    expected_tool_paths = {
        "xsdb": EXPECTED_VITIS_ROOT / "bin" / "xsdb.bat",
        "loader": EXPECTED_VITIS_ROOT / "bin" / "loader.bat",
        "setup_env": EXPECTED_VITIS_ROOT / "bin" / "setupEnv.bat",
        "rdi_args": EXPECTED_VITIS_ROOT / "bin" / "rdiArgs.bat",
        "xsdb_exe": (
            EXPECTED_VITIS_ROOT / "bin" / "unwrapped" / "win64.o" / "xsdb.exe"
        ),
        "xsdb_manifest": (
            EXPECTED_VITIS_ROOT
            / "bin"
            / "unwrapped"
            / "win64.o"
            / "xsdb.exe.manifest"
        ),
    }
    for name, expected_path in expected_tool_paths.items():
        value = launcher.get(name)
        if not isinstance(value, str) or Path(value).resolve() != expected_path.resolve():
            raise StartupFailure(
                f"QWEB launch {name} is outside the canonical Vitis toolchain"
            )
    if (
        launcher.get("command") != launcher.get("xsdb")
        or launcher.get("arguments") != ["-no-ini", launcher.get("tcl")]
        or launcher.get("working_directory") != "xsdb_profile"
        or launcher.get("isolated_home") != "xsdb_profile"
        or launcher.get("profile_initially_empty") is not True
        or launcher.get("sanitized_environment")
        != list(EXPECTED_SANITIZED_XSDB_ENVIRONMENT)
    ):
        raise StartupFailure("QWEB launch command does not execute the audited XSDB/Tcl")
    profile_path = (output_dir / "xsdb_profile").resolve(strict=True)
    if not profile_path.is_dir():
        raise StartupFailure("QWEB isolated XSDB profile is not a directory")
    for startup_name in (".xsdbrc", "xsdb.ini"):
        if (profile_path / startup_name).exists():
            raise StartupFailure(
                f"QWEB isolated XSDB profile contains forbidden {startup_name}"
            )
    log_path = (output_dir / "xsdb.log").resolve(strict=True)
    log_raw = log_path.read_bytes()
    if launcher.get("output_log_sha256") != hashlib.sha256(log_raw).hexdigest():
        raise StartupFailure("QWEB XSDB output log SHA-256 mismatch")
    try:
        log_text = log_raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise StartupFailure("QWEB XSDB output log is not strict UTF-8") from error
    marker_position = -1
    for marker in REQUIRED_XSDB_MARKERS:
        next_position = log_text.find(marker, marker_position + 1)
        if next_position < 0:
            raise StartupFailure(
                f"QWEB XSDB output log lacks ordered PASS marker: {marker}"
            )
        marker_position = next_position
    capture_started = _parse_utc(capture_started_utc, "capture started_utc")
    uart_ready = _parse_utc(uart_ready_utc, "UART READY utc")
    launch_started = _parse_utc(payload.get("started_utc"), "launch started_utc")
    launch_finished = _parse_utc(payload.get("finished_utc"), "launch finished_utc")
    now = datetime.now(timezone.utc)
    if not capture_started <= launch_started <= uart_ready:
        raise StartupFailure("QWEB UART READY did not occur after this launch started")
    if launch_finished < launch_started:
        raise StartupFailure("QWEB launch timestamps do not fall within this capture")
    if launch_finished > now + timedelta(seconds=5):
        raise StartupFailure("QWEB launch report finished_utc is in the future")
    return {
        "report": requested_path.relative_to(output_dir).as_posix(),
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "run_id": run_id,
        "started_utc": payload["started_utc"],
        "finished_utc": payload["finished_utc"],
        "audited": audited,
        "claim_sha256": claim_record["sha256"],
        "xsdb_log_sha256": launcher["output_log_sha256"],
    }


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


def _exception_result(error: Exception) -> dict[str, object]:
    """Return a fail-closed parser-shaped result for a capture exception."""

    reason = str(error) or type(error).__name__
    parser = QwebStartupParser()
    parser._fail(reason)
    return parser.result()


def _build_report(
    *,
    mode: str,
    started_utc: str,
    result: dict[str, object],
    output_dir: Path,
    raw_path: Path,
    report_path: Path,
    port: str | None,
    transcript_path: Path | None,
    launch_binding: dict[str, object] | None,
    capture_error: Exception | None = None,
) -> dict[str, object]:
    """Build one portable report after the raw UART stream has been closed."""

    passed = result.get("passed") is True
    reason = result.get("failure")
    error_record: dict[str, object] | None = None
    if not passed:
        error_record = {
            "kind": (
                "capture_exception"
                if capture_error is not None
                else "startup_validation"
            ),
            "type": (
                type(capture_error).__name__
                if capture_error is not None
                else "StartupFailure"
            ),
            "reason": (
                str(reason)
                if reason is not None
                else "startup validation failed without a reason"
            ),
        }

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "tool": Path(__file__).name,
        "mode": mode,
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        **result,
        "error": error_record,
        "serial": {
            "port": port if mode == "live" else None,
            "baud_rate": BAUD_RATE,
        },
        "input_transcript": (
            str(transcript_path) if transcript_path is not None else None
        ),
        "launch_binding": launch_binding,
        "artifacts": {
            "uart_raw": raw_path.relative_to(output_dir).as_posix(),
            "uart_raw_bytes": raw_path.stat().st_size,
            "uart_raw_sha256": _sha256_file(raw_path),
            "report": report_path.relative_to(output_dir).as_posix(),
        },
    }


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise StartupFailure(f"refusing to overwrite evidence artifact: {path}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise StartupFailure(f"refusing to overwrite evidence artifact: {path}") from error
    except OSError as error:
        raise StartupFailure(
            f"could not atomically commit evidence artifact without overwrite: {path}: {error}"
        ) from error
    temporary.unlink()


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
        "--launch-json",
        type=Path,
        default=None,
        help=(
            "launch.json written by run_qweb_board.ps1 in --output-dir; "
            "required for live Gate2 evidence"
        ),
    )
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
    if args.port is not None and args.launch_json is None:
        print("ERROR: live Gate2 capture requires --launch-json", file=sys.stderr)
        return 2
    if args.verify_transcript is not None and args.launch_json is not None:
        print("ERROR: transcript replay cannot use --launch-json", file=sys.stderr)
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
        mode = "transcript_replay" if transcript_data is not None else "live"
        capture_error: Exception | None = None
        launch_binding: dict[str, object] | None = None

        try:
            if transcript_data is not None:
                with raw_path.open("xb") as stream:
                    stream.write(transcript_data)
                result = verify_transcript_bytes(transcript_data)
            else:
                with raw_path.open("xb") as stream:
                    result = capture_live(
                        args.port,
                        stream,
                        timeout=args.timeout,
                        linger=args.linger,
                    )
        except Exception as error:
            capture_error = error
            if not raw_path.exists():
                raw_path.touch(exist_ok=False)
            result = _exception_result(error)

        if (
            mode == "live"
            and result.get("passed") is True
            and args.launch_json is not None
        ):
            try:
                launch_binding = _load_launch_binding(
                    args.launch_json,
                    output_dir=output_dir,
                    capture_started_utc=started_utc,
                    uart_ready_utc=result.get("uart_ready_utc"),
                )
            except Exception as error:
                capture_error = error
                result = _exception_result(error)

        report = _build_report(
            mode=mode,
            started_utc=started_utc,
            result=result,
            output_dir=output_dir,
            raw_path=raw_path,
            report_path=report_path,
            port=args.port,
            transcript_path=transcript_path,
            launch_binding=launch_binding,
            capture_error=capture_error,
        )
        _write_json_exclusive(report_path, report)
    except (StartupFailure, OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    if capture_error is not None:
        print(f"FAIL: {report['failure']}", file=sys.stderr)
        print(f"report: {report_path}", file=sys.stderr)
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
