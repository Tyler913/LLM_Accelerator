#!/usr/bin/env python3
"""Run the audited PS-Ethernet echo image and close physical Gate 1."""

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
import socket
import subprocess
import sys
import time
from typing import BinaryIO, Callable, Mapping, Sequence
import uuid


REPORT_SCHEMA_VERSION = 1
BAUD_RATE = 115_200
MAX_CAPTURE_BYTES = 4 * 1024 * 1024
MAX_CAPTURE_LINES = 4096
MAX_LINE_BYTES = 4096
EXPECTED_PHY_ADDRESS = 7
EXPECTED_PHY_ID1 = 0x0000
EXPECTED_PHY_ID2 = 0x011A
EXPECTED_LINK_SPEED_MBPS = 1000
EXPECTED_ECHO_PORT = 7
PING_COUNT = 10

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
WRAPPER = HERE / "run_network_echo_board.ps1"
TCL_LAUNCHER = HERE / "launch_network_echo.tcl"
DEFAULT_WORKSPACE = Path(r"F:\vwk")
EXPECTED_VITIS_ROOT = Path(
    r"D:\Applications\Vivado_2025.1.1\2025.1.1\Vitis"
)
EXPECTED_WRAPPER_SHA256 = (
    "ab0b9ee31e30f412765e03cc34aab8aa70d2f16f8a30d158d5c11b3280ed6dbb"
)
EXPECTED_TCL_SHA256 = (
    "6c7a653dda7838ce7acbe343797cb690cb37f0ccb7be3abc21097e14cf0a326d"
)
EXPECTED_INPUT_HASHES: dict[str, str] = {
    "network_manifest": "33ca1a825aff72dd7a59c9938c0b0e838062c5c6e18ee1826432341e7a85e401",
    "network_bit": "c926b3db8021e976e5ea6cc2f71da3c44899ea5c3df61da573475ce6d8c21239",
    "network_xsa": "0af1257442a68a6beb31d94811713f2ae8e6af63e0a85dd497146405e37406cf",
    "network_fsbl": "a7695cd19bec264ca0ef1d951527587cc02489a82711ab63f4534c5f3579f5cb",
    "echo_elf": "5590959d661b4eca1ae88df7b50537b816851fc0e14ed14fd7754371b2f7838b",
    "xsdb": "2cd1d63586a95fbb00b2badb6652cae485bdd4cbdab612c8a5de0afd7e874f59",
    "loader": "13fac52e8f0c81ea0539a0edddf8ad839d5c8a49d9fe37531abffbab8decfb38",
    "setup_env": "0f8108b67a75b3957b075c3f87d094fcc63eb7251a27281e7d456024898d881f",
    "rdi_args": "44f7765ce290ea0546f441044a5303316f7fe6c9434136ba013e8f72c5e1f544",
    "xsdb_exe": "9ef5fe040728b508a1a0efd1dfd83dbb3df02d2d9b59310ea0e7b403cd429483",
    "xsdb_manifest": "77596430e6ea9d0a8d7b686d75f0b554564fdc0a931d1f806cf16c0e100a99e3",
}
REQUIRED_LAUNCH_MARKERS = (
    "PASS audited network echo launch inputs",
    "bitstream SHA-256: " + EXPECTED_INPUT_HASHES["network_bit"],
    "XSA SHA-256: " + EXPECTED_INPUT_HASHES["network_xsa"],
    "echo ELF SHA-256: " + EXPECTED_INPUT_HASHES["echo_elf"],
    "PASS programmed network bitstream",
    "PASS FSBL initialization",
    "PASS network echo application downloaded and running",
    "PASS launched audited network echo image",
)
UNSAFE_CHILD_ENVIRONMENT = (
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

HEADER_RECORD = "-----lwIP TCP echo server ------"
READY_RECORD = "TCP echo server started @ port 7"
DETECT_PATTERN = re.compile(
    r"^Detected Motorcomm YT8521 at PHY address (?P<address>\d+) "
    r"\(id1=0x(?P<id1>[0-9A-Fa-f]{4}) id2=0x(?P<id2>[0-9A-Fa-f]{4})\)$"
)
LINK_PATTERN = re.compile(
    r"^YT8521 link resolved: bmsr=0x(?P<bmsr>[0-9A-Fa-f]{4}) "
    r"status=0x(?P<status>[0-9A-Fa-f]{4}) duplex=(?P<duplex>full|half)$"
)
SPEED_PATTERN = re.compile(
    r"^link speed for phy address (?P<address>\d+): (?P<speed>\d+)$"
)
IP_PATTERN = re.compile(
    r"^Board IP:\s*(?P<ip>\d{1,3}(?:\.\d{1,3}){3})$"
)
FAILURE_PREFIXES = (
    "Error adding N/W interface",
    "Error creating PCB",
    "Unable to bind to port",
    "Out of memory while tcp_listen",
    "Auto negotiation error",
    "Phy setup error",
    "Phy setup failure",
    "Ethernet Link down",
    "Could not find PHY",
    "Unsupported Ethernet PHY",
    "Unsupported PHY initialization",
    "YT8521 software reset timeout",
    "YT8521 extended-register initialization failed",
    "YT8521 autonegotiation timeout",
    "YT8521 reported invalid speed status",
)


class Gate1Failure(RuntimeError):
    """A fail-closed Gate 1 acceptance error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(
        str(right.resolve())
    )


def _require_file_hash(path: Path, expected: str, label: str) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise Gate1Failure(f"{label} is missing: {resolved}")
    actual = _sha256_file(resolved)
    if actual != expected:
        raise Gate1Failure(
            f"{label} SHA-256 mismatch: {actual} != {expected}"
        )
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": actual,
    }


def audit_formal_inputs(workspace: Path) -> dict[str, object]:
    workspace = workspace.resolve()
    if not _same_path(workspace, DEFAULT_WORKSPACE):
        raise Gate1Failure(
            f"physical Gate 1 requires the formal workspace {DEFAULT_WORKSPACE}, "
            f"not {workspace}"
        )
    if not workspace.is_dir():
        raise Gate1Failure(f"formal network workspace is missing: {workspace}")

    wrapper = _require_file_hash(
        WRAPPER, EXPECTED_WRAPPER_SHA256, "network echo PowerShell wrapper"
    )
    launcher = _require_file_hash(
        TCL_LAUNCHER, EXPECTED_TCL_SHA256, "network echo Tcl launcher"
    )
    manifest_path = workspace / "network_workspace_manifest.json"
    manifest_record = _require_file_hash(
        manifest_path,
        EXPECTED_INPUT_HASHES["network_manifest"],
        "network workspace manifest",
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Gate1Failure(f"could not parse network workspace manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise Gate1Failure("network workspace manifest is not a JSON object")
    if manifest.get("platform_name") != "p_net" or manifest.get("app_name") != "a_net_echo":
        raise Gate1Failure("network workspace manifest component names are not p_net/a_net_echo")

    xsa_value = manifest.get("xsa_snapshot")
    if not isinstance(xsa_value, str):
        raise Gate1Failure("network workspace manifest lacks xsa_snapshot")
    xsa = _require_file_hash(
        Path(xsa_value), EXPECTED_INPUT_HASHES["network_xsa"], "network XSA"
    )
    bit_candidates = tuple(
        (workspace / "p_net" / "export" / "p_net" / "hw" / "sdt").glob("*.bit")
    )
    if len(bit_candidates) != 1:
        raise Gate1Failure(
            f"expected one formal network bitstream, found {len(bit_candidates)}"
        )
    bit = _require_file_hash(
        bit_candidates[0], EXPECTED_INPUT_HASHES["network_bit"], "network bitstream"
    )
    fsbl = _require_file_hash(
        workspace / "p_net" / "export" / "p_net" / "sw" / "boot" / "fsbl.elf",
        EXPECTED_INPUT_HASHES["network_fsbl"],
        "network FSBL",
    )
    echo = _require_file_hash(
        workspace / "a_net_echo" / "build" / "a_net_echo.elf",
        EXPECTED_INPUT_HASHES["echo_elf"],
        "network echo ELF",
    )

    tool_paths = {
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
    tools = {
        name: _require_file_hash(path, EXPECTED_INPUT_HASHES[name], f"AMD {name}")
        for name, path in tool_paths.items()
    }
    return {
        "workspace": str(workspace),
        "wrapper": wrapper,
        "tcl_launcher": launcher,
        "network_manifest": manifest_record,
        "network_bit": bit,
        "network_xsa": xsa,
        "network_fsbl": fsbl,
        "echo_elf": echo,
        "vitis_root": str(EXPECTED_VITIS_ROOT.resolve()),
        "tools": tools,
    }


@dataclass
class LineFramer:
    pending: bytearray = field(default_factory=bytearray)
    line_count: int = 0

    def feed(self, data: bytes) -> list[str]:
        self.pending.extend(data)
        if len(self.pending) > MAX_LINE_BYTES:
            raise Gate1Failure("UART line exceeds the 4096-byte bound")
        result: list[str] = []
        while True:
            try:
                index = self.pending.index(0x0A)
            except ValueError:
                break
            raw = bytes(self.pending[:index])
            del self.pending[: index + 1]
            if raw.endswith(b"\r"):
                raw = raw[:-1]
            line = raw.decode("utf-8", errors="replace").lstrip("\r")
            self.line_count += 1
            if self.line_count > MAX_CAPTURE_LINES:
                raise Gate1Failure("UART transcript exceeds the 4096-line bound")
            result.append(line)
        return result


@dataclass
class EchoStartupParser:
    next_milestone: int = 0
    milestones: list[dict[str, object]] = field(default_factory=list)
    phy: dict[str, object] | None = None
    link: dict[str, object] | None = None
    network: dict[str, object] | None = None

    MILESTONES = ("application_header", "phy_detected", "link_resolved", "link_speed", "board_ip", "echo_ready")

    def _observe(self, name: str, observed_utc: str, details: Mapping[str, object] | None = None) -> None:
        if self.next_milestone >= len(self.MILESTONES):
            return
        expected = self.MILESTONES[self.next_milestone]
        if name != expected:
            raise Gate1Failure(
                f"UART milestone out of order: observed {name}, expected {expected}"
            )
        record: dict[str, object] = {"name": name, "observed_utc": observed_utc}
        if details is not None:
            record["details"] = dict(details)
        self.milestones.append(record)
        self.next_milestone += 1

    def process_line(self, line: str, observed_utc: str) -> None:
        normalized = line.strip()
        if not normalized:
            return
        if any(normalized.startswith(prefix) for prefix in FAILURE_PREFIXES):
            raise Gate1Failure(f"UART reported failure: {normalized}")

        if normalized == HEADER_RECORD:
            self._observe("application_header", observed_utc)
            return

        match = DETECT_PATTERN.fullmatch(normalized)
        if match:
            details = {
                "address": int(match.group("address")),
                "id1": int(match.group("id1"), 16),
                "id2": int(match.group("id2"), 16),
            }
            if details != {
                "address": EXPECTED_PHY_ADDRESS,
                "id1": EXPECTED_PHY_ID1,
                "id2": EXPECTED_PHY_ID2,
            }:
                raise Gate1Failure(f"unexpected YT8521 identity: {details}")
            self.phy = details
            self._observe("phy_detected", observed_utc, details)
            return

        match = LINK_PATTERN.fullmatch(normalized)
        if match:
            bmsr = int(match.group("bmsr"), 16)
            status = int(match.group("status"), 16)
            duplex = match.group("duplex")
            if (bmsr & 0x0024) != 0x0024:
                raise Gate1Failure(
                    f"YT8521 link line lacks link/autoneg-complete bits: bmsr=0x{bmsr:04x}"
                )
            if (status & 0x0C00) != 0x0C00 or duplex != "full":
                raise Gate1Failure(
                    f"YT8521 did not resolve a full-duplex link: status=0x{status:04x} duplex={duplex}"
                )
            details = {"bmsr": bmsr, "status": status, "duplex": duplex}
            self.link = details
            self._observe("link_resolved", observed_utc, details)
            return

        match = SPEED_PATTERN.fullmatch(normalized)
        if match:
            address = int(match.group("address"))
            speed = int(match.group("speed"))
            if address != EXPECTED_PHY_ADDRESS or speed != EXPECTED_LINK_SPEED_MBPS:
                raise Gate1Failure(
                    f"unsupported resolved link: PHY {address}, speed {speed} Mbps; "
                    f"expected PHY {EXPECTED_PHY_ADDRESS} at {EXPECTED_LINK_SPEED_MBPS} Mbps"
                )
            assert self.link is not None
            self.link["speed_mbps"] = speed
            self._observe(
                "link_speed", observed_utc, {"address": address, "speed_mbps": speed}
            )
            return

        match = IP_PATTERN.fullmatch(normalized)
        if match:
            try:
                address = ipaddress.IPv4Address(match.group("ip"))
            except ipaddress.AddressValueError as error:
                raise Gate1Failure(f"invalid Board IP record: {normalized}") from error
            if address.is_unspecified or address.is_loopback or address.is_multicast or address.is_link_local:
                raise Gate1Failure(f"Board IP is not usable for Gate 1: {address}")
            self.network = {"ip": str(address), "port": EXPECTED_ECHO_PORT}
            self._observe("board_ip", observed_utc, self.network)
            return

        if normalized == READY_RECORD:
            self._observe(
                "echo_ready", observed_utc, {"port": EXPECTED_ECHO_PORT}
            )

    @property
    def complete(self) -> bool:
        return self.next_milestone == len(self.MILESTONES)

    def result(self) -> dict[str, object]:
        if not self.complete or self.phy is None or self.link is None or self.network is None:
            missing = list(self.MILESTONES[self.next_milestone :])
            raise Gate1Failure(f"UART startup is incomplete; missing {missing}")
        return {
            "milestones": self.milestones,
            "phy": self.phy,
            "link": self.link,
            "network": self.network,
        }


def verify_startup_transcript(data: bytes) -> dict[str, object]:
    if len(data) > MAX_CAPTURE_BYTES:
        raise Gate1Failure("UART transcript exceeds the 4 MiB bound")
    framer = LineFramer()
    parser = EchoStartupParser()
    observed = "1970-01-01T00:00:00Z"
    for line in framer.feed(data):
        parser.process_line(line, observed)
    if framer.pending and any(byte != 0x0D for byte in framer.pending):
        raise Gate1Failure("UART transcript ends with an unterminated line")
    return parser.result()


def _powershell_path() -> Path:
    root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    path = root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if not path.is_file():
        raise Gate1Failure(f"Windows PowerShell is missing: {path}")
    return path.resolve()


def _child_environment(profile: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in UNSAFE_CHILD_ENVIRONMENT:
        environment.pop(name, None)
    environment["PATH"] = ""
    environment["HOME"] = str(profile)
    environment["USERPROFILE"] = str(profile)
    return environment


def _launch_command(workspace: Path) -> list[str]:
    return [
        str(_powershell_path()),
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(WRAPPER.resolve()),
        "-Workspace",
        str(workspace.resolve()),
        "-Xsdb",
        str((EXPECTED_VITIS_ROOT / "bin" / "xsdb.bat").resolve()),
    ]


def _validate_ordered_markers(text: str, markers: Sequence[str]) -> None:
    position = -1
    for marker in markers:
        next_position = text.find(marker, position + 1)
        if next_position < 0:
            raise Gate1Failure(f"launcher output lacks ordered marker: {marker}")
        position = next_position


def _select_source_ip(
    board_ip: str,
    requested: str | None,
    *,
    socket_factory: Callable[..., socket.socket] = socket.socket,
) -> str:
    try:
        board = ipaddress.IPv4Address(board_ip)
        if requested is not None:
            source = ipaddress.IPv4Address(requested)
        else:
            probe = socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                probe.connect((board_ip, EXPECTED_ECHO_PORT))
                source = ipaddress.IPv4Address(probe.getsockname()[0])
            finally:
                probe.close()
    except (OSError, ipaddress.AddressValueError) as error:
        raise Gate1Failure(f"could not select the host source IPv4 address: {error}") from error
    if source.is_unspecified or source.is_loopback or source.is_multicast:
        raise Gate1Failure(f"selected host source IPv4 is unusable: {source}")
    if board == ipaddress.IPv4Address("192.168.1.10") and source not in ipaddress.IPv4Network(
        "192.168.1.0/24"
    ):
        raise Gate1Failure(
            f"board fallback IP {board} is routed from {source}; configure "
            "192.168.1.20/24 on the directly connected Ethernet adapter so a "
            "VPN/proxy tunnel cannot satisfy the network checks"
        )
    if source == board:
        raise Gate1Failure("host source IPv4 equals the board IPv4 address")
    return str(source)


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=5)
        except (OSError, subprocess.SubprocessError):
            try:
                process.kill()
            except OSError:
                pass


def capture_startup_after_launch(
    uart: object,
    raw_stream: BinaryIO,
    start_launcher: Callable[[], subprocess.Popen[bytes]],
    *,
    timeout: float,
) -> tuple[subprocess.Popen[bytes], dict[str, object], str]:
    # Two resets close the stale-banner window: the first precedes process
    # creation; the second follows it but still precedes JTAG programming.
    uart.reset_input_buffer()  # type: ignore[attr-defined]
    process = start_launcher()
    launch_started_utc = _utc_now()
    uart.reset_input_buffer()  # type: ignore[attr-defined]

    framer = LineFramer()
    parser = EchoStartupParser()
    deadline = time.monotonic() + timeout
    total_bytes = 0
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code not in (None, 0):
            raise Gate1Failure(f"network echo launcher exited with code {return_code}")
        data = uart.read(4096)  # type: ignore[attr-defined]
        if not data:
            continue
        total_bytes += len(data)
        if total_bytes > MAX_CAPTURE_BYTES:
            raise Gate1Failure("UART capture exceeds the 4 MiB bound")
        raw_stream.write(data)
        raw_stream.flush()
        observed_utc = _utc_now()
        for line in framer.feed(data):
            print(line, flush=True)
            parser.process_line(line, observed_utc)
        if parser.complete:
            return process, parser.result(), launch_started_utc
    raise Gate1Failure(
        f"UART startup did not reach {READY_RECORD!r} within {timeout:.1f}s"
    )


def _run_ping(
    ip: str,
    source_ip: str,
    timeout_ms: int,
    stdout_path: Path,
    stderr_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, object]:
    command = [
        "ping.exe",
        "-4",
        "-S",
        source_ip,
        "-n",
        str(PING_COUNT),
        "-w",
        str(timeout_ms),
        ip,
    ]
    try:
        completed = runner(
            command,
            check=False,
            capture_output=True,
            timeout=(PING_COUNT * timeout_ms / 1000.0) + 15.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise Gate1Failure(f"ping execution failed: {error}") from error
    stdout = bytes(completed.stdout or b"")
    stderr = bytes(completed.stderr or b"")
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    if len(stdout) > 1024 * 1024 or len(stderr) > 1024 * 1024:
        raise Gate1Failure("ping output exceeds the 1 MiB evidence bound")
    text = stdout.decode("utf-8", errors="replace")
    reply_lines = [
        line for line in text.splitlines() if "TTL=" in line.upper() and ip in line
    ]
    if completed.returncode != 0 or len(reply_lines) != PING_COUNT:
        raise Gate1Failure(
            f"ping acceptance failed: exit={completed.returncode}, "
            f"TTL replies={len(reply_lines)}/{PING_COUNT}"
        )
    return {
        "command": command,
        "source_ip": source_ip,
        "exit_code": completed.returncode,
        "ttl_replies": len(reply_lines),
        "required_replies": PING_COUNT,
    }


def _run_tcp_echo(
    ip: str,
    source_ip: str,
    run_id: str,
    timeout: float,
    tx_path: Path,
    rx_path: Path,
    *,
    connector: Callable[..., socket.socket] = socket.create_connection,
) -> dict[str, object]:
    payload = f"qweb-gate1:{run_id}:{_utc_now()}".encode("ascii")
    tx_path.write_bytes(payload)
    received = bytearray()
    try:
        with connector(
            (ip, EXPECTED_ECHO_PORT),
            timeout=timeout,
            source_address=(source_ip, 0),
        ) as connection:
            connection.settimeout(timeout)
            connection.sendall(payload)
            while len(received) < len(payload):
                block = connection.recv(len(payload) - len(received))
                if not block:
                    raise Gate1Failure(
                        f"TCP echo closed after {len(received)}/{len(payload)} bytes"
                    )
                received.extend(block)
    except Gate1Failure:
        raise
    except (OSError, TimeoutError) as error:
        raise Gate1Failure(f"TCP echo failed: {error}") from error
    finally:
        rx_path.write_bytes(bytes(received))
    if bytes(received) != payload:
        raise Gate1Failure("TCP port 7 response is not byte-exact")
    return {
        "host": ip,
        "source_ip": source_ip,
        "port": EXPECTED_ECHO_PORT,
        "tx_bytes": len(payload),
        "rx_bytes": len(received),
        "tx_sha256": hashlib.sha256(payload).hexdigest(),
        "rx_sha256": hashlib.sha256(received).hexdigest(),
        "exact_match": True,
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise Gate1Failure(f"refusing to overwrite evidence artifact: {path}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise Gate1Failure(f"refusing to overwrite evidence artifact: {path}") from error
    except OSError as error:
        raise Gate1Failure(
            f"could not atomically commit evidence artifact: {path}: {error}"
        ) from error
    temporary.unlink()


def _prepare_output_dir(path: Path | None) -> Path:
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = REPOSITORY_ROOT / "Temp" / f"network_echo_gate1_{stamp}"
    resolved = path.resolve()
    try:
        resolved.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise Gate1Failure(f"refusing to reuse evidence directory: {resolved}") from error
    return resolved


def _artifact_inventory(output_dir: Path) -> dict[str, object]:
    records: dict[str, object] = {}
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name not in {"acceptance.json", "acceptance.json.tmp"}:
            records[path.name] = {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="board UART, for example COM230")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--source-ip",
        help="host IPv4 source; otherwise derive it from the active route",
    )
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--launcher-exit-timeout", type=float, default=30.0)
    parser.add_argument("--ping-timeout-ms", type=int, default=2000)
    parser.add_argument("--tcp-timeout", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (1.0 <= args.startup_timeout <= 600.0):
        raise SystemExit("--startup-timeout must be between 1 and 600 seconds")
    if not (1.0 <= args.launcher_exit_timeout <= 120.0):
        raise SystemExit("--launcher-exit-timeout must be between 1 and 120 seconds")
    if not (100 <= args.ping_timeout_ms <= 10_000):
        raise SystemExit("--ping-timeout-ms must be between 100 and 10000")
    if not (0.5 <= args.tcp_timeout <= 60.0):
        raise SystemExit("--tcp-timeout must be between 0.5 and 60 seconds")

    try:
        output_dir = _prepare_output_dir(args.output_dir)
    except Gate1Failure as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    report_path = output_dir / "acceptance.json"
    claim_path = output_dir / "acceptance.claim.json"
    raw_path = output_dir / "uart_raw.bin"
    stdout_path = output_dir / "launcher_stdout.log"
    stderr_path = output_dir / "launcher_stderr.log"
    ping_stdout_path = output_dir / "ping_stdout.txt"
    ping_stderr_path = output_dir / "ping_stderr.txt"
    tx_path = output_dir / "tcp_tx.bin"
    rx_path = output_dir / "tcp_rx.bin"
    profile = output_dir / "xsdb_profile"
    run_id = str(uuid.uuid4())
    started_utc = _utc_now()
    stage = "claim"
    process: subprocess.Popen[bytes] | None = None
    input_audit: dict[str, object] | None = None
    startup: dict[str, object] | None = None
    launch: dict[str, object] | None = None
    ping: dict[str, object] | None = None
    tcp: dict[str, object] | None = None
    source_ip: str | None = None
    failure: dict[str, object] | None = None

    claim = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "started_utc": started_utc,
        "mode": "live",
        "port": args.port,
        "workspace": str(args.workspace.resolve()),
        "evidence_directory": str(output_dir),
    }
    try:
        _write_json_exclusive(claim_path, claim)
        stage = "input_audit"
        input_audit = audit_formal_inputs(args.workspace)
        profile.mkdir(exist_ok=False)
        if tuple(profile.iterdir()):
            raise Gate1Failure("isolated XSDB profile is not empty")

        try:
            import serial  # type: ignore[import-not-found]
        except ImportError as error:
            raise Gate1Failure("pyserial is required for live Gate 1") from error

        stage = "serial_and_launch"
        command = _launch_command(args.workspace)
        with raw_path.open("xb") as raw_stream, stdout_path.open(
            "xb"
        ) as stdout_stream, stderr_path.open("xb") as stderr_stream, serial.Serial(
            port=args.port,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        ) as uart:

            def start_launcher() -> subprocess.Popen[bytes]:
                nonlocal process
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    cwd=profile,
                    env=_child_environment(profile),
                    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                )
                return process

            process, startup, launch_started_utc = capture_startup_after_launch(
                uart,
                raw_stream,
                start_launcher,
                timeout=args.startup_timeout,
            )
            try:
                exit_code = process.wait(timeout=args.launcher_exit_timeout)
            except subprocess.TimeoutExpired as error:
                raise Gate1Failure(
                    f"network echo launcher did not exit within {args.launcher_exit_timeout:.1f}s "
                    "after UART READY"
                ) from error
            if exit_code != 0:
                raise Gate1Failure(f"network echo launcher exited with code {exit_code}")

        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
        _validate_ordered_markers(stdout_text, REQUIRED_LAUNCH_MARKERS)
        launch = {
            "command": command,
            "started_utc": launch_started_utc,
            "exit_code": process.returncode,
            "required_markers": list(REQUIRED_LAUNCH_MARKERS),
            "sanitized_environment": list(UNSAFE_CHILD_ENVIRONMENT),
            "working_directory": "xsdb_profile",
            "profile_initially_empty": True,
        }

        stage = "route"
        assert startup is not None
        ip = str(startup["network"]["ip"])  # type: ignore[index]
        source_ip = _select_source_ip(ip, args.source_ip)
        stage = "ping"
        ping = _run_ping(
            ip,
            source_ip,
            args.ping_timeout_ms,
            ping_stdout_path,
            ping_stderr_path,
        )
        stage = "tcp_echo"
        tcp = _run_tcp_echo(
            ip,
            source_ip,
            run_id,
            args.tcp_timeout,
            tx_path,
            rx_path,
        )
        stage = "complete"
    except Exception as error:  # Evidence must survive every in-run failure.
        if process is not None:
            _terminate_process_tree(process)
        failure = {
            "stage": stage,
            "type": type(error).__name__,
            "reason": str(error),
        }
    finally:
        for path in (raw_path, stdout_path, stderr_path):
            if not path.exists():
                path.write_bytes(b"")

    finished_utc = _utc_now()
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "mode": "live",
        "passed": failure is None,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "evidence_directory": str(output_dir),
        "inputs": input_audit,
        "launch": launch,
        "startup": startup,
        "source_ip": source_ip,
        "ping": ping,
        "tcp_echo": tcp,
        "failure": failure,
        "artifacts": _artifact_inventory(output_dir),
    }
    try:
        _write_json_exclusive(report_path, report)
    except Gate1Failure as error:
        print(f"ERROR: could not commit Gate 1 report: {error}", file=sys.stderr)
        return 2

    if failure is not None:
        print(
            f"FAIL Ethernet Gate 1 ({failure['stage']}): {failure['reason']}\n"
            f"Evidence: {report_path}",
            file=sys.stderr,
        )
        return 1
    assert startup is not None
    ip = startup["network"]["ip"]  # type: ignore[index]
    print(
        f"PASS Ethernet Gate 1: PHY {EXPECTED_PHY_ADDRESS} at "
        f"{EXPECTED_LINK_SPEED_MBPS} Mbps full duplex, IP {ip}, "
        f"ping {PING_COUNT}/{PING_COUNT}, exact TCP port {EXPECTED_ECHO_PORT} echo\n"
        f"Evidence: {report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
