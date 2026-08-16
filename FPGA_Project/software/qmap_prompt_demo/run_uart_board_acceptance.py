#!/usr/bin/env python3
"""Run or replay the exact Qwen3 prompt-demo UART acceptance sequence.

Live mode lazily imports pyserial, optionally launches a prompt-demo workbench,
and never sends a command until the preceding command has reached its exact
terminal record.  Replay mode applies the same ordered contract to captured
UART bytes and therefore needs neither pyserial nor a connected board.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import re
import signal
from typing import BinaryIO, Callable, Iterable, Sequence


BAUD_RATE = 115_200
READY_RECORD = "READY vocab=151936 context=256"
STARTUP_RECORDS = (
    "Qwen3 text/token prompt demo",
    "DDR4 status=0x00000005",
    "TOKENIZER tokens=151669 model_vocab=151936 eos=151643 bytes=3629566",
    READY_RECORD,
)
# main_generate.c currently emits no separate sentinel PASS record: a bad
# sentinel exits through ERROR RUNTIME_SENTINEL before TOKENIZER/READY.
PROTOCOL_PREFIXES = (
    "PONG",
    "PROMPT_IDS ",
    "START ",
    "PREFILL ",
    "TOKEN ",
    "BYTES ",
    "DONE ",
    "ERROR ",
)
NEGATIVE_GRACE_FORBIDDEN_PREFIXES = (
    "START ",
    "BUSY",
    "PREFILL ",
    "TOKEN ",
    "BYTES ",
    "DONE ",
)
MAX_READY_TIMEOUT_SECONDS = 3_600.0
MAX_COMMAND_TIMEOUT_SECONDS = 3_600.0


@dataclass(frozen=True)
class AcceptanceCase:
    name: str
    command: str
    expected_records: tuple[str, ...]
    timeout_class: str
    expected_busy_positions: tuple[int, ...] = ()


@dataclass
class CaseResult:
    name: str
    command: str
    expected_records: list[str]
    observed_records: list[str]
    busy_progress: list[dict[str, object]]
    echo_observed: bool
    passed: bool
    duration_seconds: float
    failure: str | None = None


class AcceptanceFailure(RuntimeError):
    """A bounded acceptance operation failed."""


def _model_case(
    name: str,
    command: str,
    body: Sequence[str],
    *,
    run_count: int,
    prompt: bool = False,
) -> AcceptanceCase:
    prefix = ("PROMPT_IDS 5 785 3853 315 89462 374",) if prompt else ()
    return AcceptanceCase(
        name, command, prefix + tuple(body), "model", tuple(range(run_count))
    )


ACCEPTANCE_CASES: tuple[AcceptanceCase, ...] = (
    AcceptanceCase("ping", "PING", ("PONG",), "ping"),
    AcceptanceCase(
        "reject_out_of_range_id",
        "TOKENS 1 1 151936",
        ("ERROR PARSE RANGE offset=11",),
        "ping",
    ),
    _model_case(
        "single_token_two_decode",
        "TOKENS 2 1 374",
        (
            "START prompt=1 max_new=2",
            "TOKEN 0 28458 1227344433",
            "BYTES 0 61616161",
            "TOKEN 1 64 1015661901",
            "BYTES 1 61",
            "DONE 2 MAX_NEW",
        ),
        run_count=2,
    ),
    _model_case(
        "two_token_prefill",
        "TOKENS 1 2 374 28458",
        (
            "START prompt=2 max_new=1",
            "PREFILL 1/2",
            "TOKEN 0 64 1015661901",
            "BYTES 0 61",
            "DONE 1 MAX_NEW",
        ),
        run_count=2,
    ),
    _model_case(
        "text_token_ids",
        "TOKENS 2 5 785 3853 315 89462 374",
        (
            "START prompt=5 max_new=2",
            "PREFILL 1/5",
            "PREFILL 2/5",
            "PREFILL 3/5",
            "PREFILL 4/5",
            "TOKEN 0 264 1296911292",
            "BYTES 0 2061",
            "TOKEN 1 26291 1225544557",
            "BYTES 1 2066617363696e6174696e67",
            "DONE 2 MAX_NEW",
        ),
        run_count=6,
    ),
    _model_case(
        "text_prompt_first",
        "PROMPT 2 The future of FPGA is",
        (
            "START prompt=5 max_new=2",
            "PREFILL 1/5",
            "PREFILL 2/5",
            "PREFILL 3/5",
            "PREFILL 4/5",
            "TOKEN 0 264 1296911292",
            "BYTES 0 2061",
            "TOKEN 1 26291 1225544557",
            "BYTES 1 2066617363696e6174696e67",
            "DONE 2 MAX_NEW",
        ),
        run_count=6,
        prompt=True,
    ),
    _model_case(
        "text_prompt_repeat",
        "PROMPT 2 The future of FPGA is",
        (
            "START prompt=5 max_new=2",
            "PREFILL 1/5",
            "PREFILL 2/5",
            "PREFILL 3/5",
            "PREFILL 4/5",
            "TOKEN 0 264 1296911292",
            "BYTES 0 2061",
            "TOKEN 1 26291 1225544557",
            "BYTES 1 2066617363696e6174696e67",
            "DONE 2 MAX_NEW",
        ),
        run_count=6,
        prompt=True,
    ),
    _model_case(
        "highest_legal_model_id",
        "TOKENS 1 1 151935",
        (
            "START prompt=1 max_new=1",
            "TOKEN 0 28458 1224741478",
            "BYTES 0 61616161",
            "DONE 1 MAX_NEW",
        ),
        run_count=1,
    ),
)


class LineFramer:
    """Frame LF-terminated UART bytes without losing an incomplete prompt."""

    def __init__(self) -> None:
        self._pending = bytearray()

    @staticmethod
    def _decode(raw: bytes) -> str:
        try:
            return raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise AcceptanceFailure(
                "UART contains invalid UTF-8 at byte offset "
                f"{error.start} of framed line: {raw.hex()}"
            ) from error

    def feed(self, data: bytes) -> list[str]:
        self._pending.extend(data)
        lines: list[str] = []
        while True:
            try:
                newline = self._pending.index(0x0A)
            except ValueError:
                break
            raw = bytes(self._pending[:newline])
            del self._pending[: newline + 1]
            if raw.endswith(b"\r"):
                raw = raw[:-1]
            lines.append(self._decode(raw))
        return lines

    def finish(self) -> list[str]:
        if not self._pending:
            return []
        raw = bytes(self._pending)
        self._pending.clear()
        if raw.endswith(b"\r"):
            raw = raw[:-1]
        return [self._decode(raw)]


def normalize_line(line: str) -> str:
    """Remove terminal CR residue and the application's prompt prefix."""

    # The ZynqMP FSBL banner can end with LF followed by a standalone CR before
    # the application starts. A terminal treats that CR as a carriage return;
    # preserve all other bytes and whitespace so protocol matching stays exact.
    normalized = line.lstrip("\r")
    while normalized.startswith("qot> "):
        normalized = normalized[5:]
    return normalized


def protocol_record(line: str) -> str | None:
    normalized = normalize_line(line)
    if normalized == "PONG":
        return normalized
    if any(normalized.startswith(prefix) for prefix in PROTOCOL_PREFIXES[1:]):
        return normalized
    return None


BUSY_PATTERN = re.compile(
    r"^BUSY position=(?P<position>[0-9]+) polls=(?P<polls>[0-9]+) "
    r"status=0x(?P<status>[0-9a-fA-F]{8})$"
)


def parse_busy_record(line: str) -> dict[str, object] | None:
    normalized = normalize_line(line)
    if not normalized.startswith("BUSY"):
        return None
    match = BUSY_PATTERN.fullmatch(normalized)
    if match is None:
        raise AcceptanceFailure(f"malformed BUSY record: {normalized!r}")
    return {
        "position": int(match.group("position")),
        "polls": int(match.group("polls")),
        "status": "0x" + match.group("status").lower(),
    }


def failure_record(line: str) -> str | None:
    normalized = normalize_line(line)
    if normalized == "ERROR" or normalized.startswith("ERROR "):
        return normalized
    if normalized == "FAIL" or normalized.startswith("FAIL "):
        return normalized
    return None


def is_command_echo(line: str, command: str) -> bool:
    return normalize_line(line) == command


def decode_lines(data: bytes) -> list[str]:
    framer = LineFramer()
    return framer.feed(data) + framer.finish()


def _new_report(mode: str) -> dict[str, object]:
    return {
        "schema": "qot-uart-board-acceptance-v1",
        "mode": mode,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "serial": {
            "baud_rate": BAUD_RATE,
            "data_bits": 8,
            "parity": "none",
            "stop_bits": 1,
        },
        "single_flight": True,
        "startup_expected": list(STARTUP_RECORDS),
        "startup_observed": [],
        "ready_expected": READY_RECORD,
        "ready_observed": None,
        "cases": [],
        "passed": False,
        "failure": None,
    }


def _case_result(
    case: AcceptanceCase,
    observed: list[str],
    started: float,
    *,
    echo_observed: bool = True,
    busy_progress: list[dict[str, object]] | None = None,
) -> CaseResult:
    progress = list(busy_progress or [])
    expected = list(case.expected_records)
    passed = observed == expected
    failure = None
    if not passed:
        index = 0
        common = min(len(observed), len(expected))
        while index < common and observed[index] == expected[index]:
            index += 1
        wanted = expected[index] if index < len(expected) else "<end-of-case>"
        actual = observed[index] if index < len(observed) else "<missing>"
        failure = f"record {index}: expected {wanted!r}, observed {actual!r}"
    if passed:
        position_order: list[int] = []
        last_poll: dict[int, int] = {}
        for item in progress:
            position = int(item["position"])
            polls = int(item["polls"])
            if not position_order or position_order[-1] != position:
                position_order.append(position)
            if position in last_poll and polls <= last_poll[position]:
                passed = False
                failure = (
                    f"BUSY polls not increasing at position {position}: "
                    f"{polls} <= {last_poll[position]}"
                )
                break
            last_poll[position] = polls
        if passed and tuple(position_order) != case.expected_busy_positions:
            passed = False
            failure = (
                "BUSY positions mismatch: expected "
                f"{list(case.expected_busy_positions)}, observed {position_order}"
            )
    return CaseResult(
        name=case.name,
        command=case.command,
        expected_records=expected,
        observed_records=observed,
        busy_progress=progress,
        echo_observed=echo_observed,
        passed=passed,
        duration_seconds=round(time.monotonic() - started, 6),
        failure=failure,
    )


def _advance_startup(expected_index: int, line: str) -> tuple[int, str | None]:
    normalized = normalize_line(line)
    fatal = failure_record(normalized)
    if fatal is not None:
        raise AcceptanceFailure(f"application failed before READY: {fatal}")
    expected = STARTUP_RECORDS[expected_index]
    if normalized == expected:
        return expected_index + 1, normalized
    if (
        normalized.startswith("Qwen3 ")
        or normalized.startswith("DDR4 status=")
        or normalized.startswith("TOKENIZER ")
        or normalized.startswith("READY ")
    ):
        raise AcceptanceFailure(
            f"startup record {expected_index}: expected {expected!r}, observed {normalized!r}"
        )
    return expected_index, None


def verify_transcript_bytes(data: bytes) -> dict[str, object]:
    """Verify a raw UART transcript using the live run's exact record order."""

    report = _new_report("transcript")
    line_index = 0
    try:
        lines = decode_lines(data)
        startup_index = 0
        while line_index < len(lines) and startup_index < len(STARTUP_RECORDS):
            startup_index, observed = _advance_startup(
                startup_index, lines[line_index]
            )
            line_index += 1
            if observed is not None:
                report["startup_observed"].append(observed)  # type: ignore[union-attr]
        if startup_index != len(STARTUP_RECORDS):
            raise AcceptanceFailure(
                f"missing startup record {startup_index}: {STARTUP_RECORDS[startup_index]!r}"
            )
        report["ready_observed"] = READY_RECORD

        for case in ACCEPTANCE_CASES:
            started = time.monotonic()
            observed: list[str] = []
            busy_progress: list[dict[str, object]] = []
            echo_observed = False
            while line_index < len(lines):
                line = lines[line_index]
                line_index += 1
                fatal = failure_record(line)
                record = protocol_record(line)
                busy = parse_busy_record(line)
                if fatal is not None or record is not None or busy is not None:
                    raise AcceptanceFailure(
                        f"{case.name}: protocol/failure before command echo: "
                        f"{fatal or record or normalize_line(line)!r}"
                    )
                if is_command_echo(line, case.command):
                    echo_observed = True
                    break
            if not echo_observed:
                raise AcceptanceFailure(
                    f"{case.name}: missing command echo {case.command!r}"
                )
            while line_index < len(lines) and len(observed) < len(case.expected_records):
                line = lines[line_index]
                line_index += 1
                record = protocol_record(line)
                fatal = failure_record(line)
                busy = parse_busy_record(line)
                expected = case.expected_records[len(observed)]
                if fatal is not None and fatal != expected:
                    observed.append(fatal)
                    break
                if record is None:
                    if busy is not None:
                        busy_progress.append(busy)
                    continue
                observed.append(record)
                if record != expected:
                    break
            result = _case_result(
                case,
                observed,
                started,
                echo_observed=echo_observed,
                busy_progress=busy_progress,
            )
            report["cases"].append(asdict(result))  # type: ignore[union-attr]
            if not result.passed:
                raise AcceptanceFailure(f"{case.name}: {result.failure}")

        for line in lines[line_index:]:
            normalized = normalize_line(line)
            fatal = failure_record(line)
            busy = parse_busy_record(line)
            forbidden = next(
                (
                    prefix
                    for prefix in NEGATIVE_GRACE_FORBIDDEN_PREFIXES
                    if normalized.startswith(prefix)
                ),
                None,
            )
            record = protocol_record(line)
            if (
                fatal is not None
                or forbidden is not None
                or record is not None
                or busy is not None
            ):
                raise AcceptanceFailure(
                    "negative-case grace observed forbidden record: "
                    f"{fatal or record or normalized!r}"
                )
        report["passed"] = True
    except AcceptanceFailure as error:
        report["failure"] = str(error)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return report


class SerialLineSource:
    """Bounded synchronous UART reader that preserves every received byte."""

    def __init__(self, serial_port: object, raw_log: BinaryIO) -> None:
        self.serial_port = serial_port
        self.raw_log = raw_log
        self.framer = LineFramer()
        self.pending_lines: deque[str] = deque()

    def read_lines_until(
        self,
        deadline: float,
        abort_check: Callable[[], str | None] | None = None,
    ) -> Iterable[str]:
        while time.monotonic() < deadline:
            if abort_check is not None:
                failure = abort_check()
                if failure is not None:
                    raise AcceptanceFailure(failure)
            if self.pending_lines:
                yield self.pending_lines.popleft()
                continue
            data = self.serial_port.read(4096)  # type: ignore[attr-defined]
            if not data:
                continue
            self.raw_log.write(data)
            self.raw_log.flush()
            self.pending_lines.extend(self.framer.feed(data))


def _wait_for_ready(
    source: SerialLineSource,
    timeout: float,
    launcher_process: object | None = None,
) -> list[str]:
    deadline = time.monotonic() + timeout
    startup_index = 0
    observed_records: list[str] = []
    def launcher_failure() -> str | None:
        if launcher_process is None:
            return None
        poll = getattr(launcher_process, "poll", None)
        if poll is None:
            return None
        return_code = poll()
        if return_code is not None and return_code != 0:
            return f"launcher exited with nonzero return code {return_code} before READY"
        return None

    for line in source.read_lines_until(deadline, launcher_failure):
        startup_index, observed = _advance_startup(startup_index, line)
        if observed is not None:
            observed_records.append(observed)
        if startup_index == len(STARTUP_RECORDS):
            return observed_records
    raise AcceptanceFailure(
        f"startup timeout after {timeout:.1f}s; next expected "
        f"{STARTUP_RECORDS[startup_index]!r}"
    )


def _run_live_case(
    source: SerialLineSource,
    serial_port: object,
    case: AcceptanceCase,
    timeout: float,
) -> CaseResult:
    started = time.monotonic()
    observed: list[str] = []
    busy_progress: list[dict[str, object]] = []
    payload = (case.command + "\r").encode("utf-8")
    serial_port.write(payload)  # type: ignore[attr-defined]
    serial_port.flush()  # type: ignore[attr-defined]
    deadline = time.monotonic() + timeout
    echo_observed = False
    for line in source.read_lines_until(deadline):
        fatal = failure_record(line)
        record = protocol_record(line)
        busy = parse_busy_record(line)
        if not echo_observed:
            if fatal is not None or record is not None or busy is not None:
                result = _case_result(
                    case,
                    [fatal or record or normalize_line(line)],
                    started,
                    echo_observed=False,
                    busy_progress=busy_progress,
                )
                result.passed = False
                result.failure = (
                    "protocol/failure before command echo: "
                    f"{fatal or record or normalize_line(line)!r}"
                )
                return result
            if is_command_echo(line, case.command):
                echo_observed = True
            continue
        expected = case.expected_records[len(observed)]
        if fatal is not None and fatal != expected:
            observed.append(fatal)
            return _case_result(
                case, observed, started, echo_observed=echo_observed
            )
        if record is None:
            if busy is not None:
                busy_progress.append(busy)
            continue
        observed.append(record)
        if record != expected or len(observed) == len(case.expected_records):
            return _case_result(
                case, observed, started, echo_observed=echo_observed
                , busy_progress=busy_progress
            )
    result = _case_result(
        case,
        observed,
        started,
        echo_observed=echo_observed,
        busy_progress=busy_progress,
    )
    if not echo_observed:
        result.failure = f"timeout after {timeout:.1f}s waiting for command echo"
        result.passed = False
        return result
    if result.failure is None:
        result.failure = f"timeout after {timeout:.1f}s"
        result.passed = False
    else:
        result.failure = f"timeout after {timeout:.1f}s; {result.failure}"
    return result


def _check_negative_grace(source: SerialLineSource, duration: float) -> None:
    deadline = time.monotonic() + duration
    for line in source.read_lines_until(deadline):
        normalized = normalize_line(line)
        fatal = failure_record(line)
        busy = parse_busy_record(line)
        forbidden = next(
            (
                prefix
                for prefix in NEGATIVE_GRACE_FORBIDDEN_PREFIXES
                if normalized.startswith(prefix)
            ),
            None,
        )
        record = protocol_record(line)
        if (
            fatal is not None
            or forbidden is not None
            or record is not None
            or busy is not None
        ):
            raise AcceptanceFailure(
                "negative-case grace observed forbidden record: "
                f"{fatal or record or normalized!r}"
            )


def _open_serial(port: str) -> object:
    try:
        import serial  # type: ignore[import-not-found]
    except ImportError as error:
        raise AcceptanceFailure(
            "live mode requires pyserial (install package 'pyserial' in llm_fpga)"
        ) from error
    try:
        return serial.Serial(
            port=port,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
            write_timeout=2.0,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
    except serial.SerialException as error:
        raise AcceptanceFailure(f"cannot open serial port {port!r}: {error}") from error


def _resolve_launcher(workbench: Path) -> Path:
    candidate = workbench / "run_board_smoke.ps1" if workbench.is_dir() else workbench
    candidate = candidate.resolve()
    if candidate.name.lower() != "run_board_smoke.ps1" or not candidate.is_file():
        raise AcceptanceFailure(
            "--launch-workbench must name a workbench directory or run_board_smoke.ps1"
        )
    return candidate


def _launcher_process_group_options() -> dict[str, object]:
    if os.name == "nt":
        return {
            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        }
    return {"start_new_session": True}


def _append_cleanup_error(cleanup: dict[str, object], message: str) -> None:
    errors = cleanup["errors"]
    if not isinstance(errors, list):
        raise RuntimeError("internal launcher cleanup error list is invalid")
    errors.append(message)


def _terminate_launcher_tree(
    process: subprocess.Popen[bytes],
) -> dict[str, object]:
    """Best-effort tree cleanup; a partial cleanup is always reported as failure."""
    cleanup: dict[str, object] = {
        "attempted": True,
        "method": "taskkill-tree" if os.name == "nt" else "process-group-sigkill",
        "tree_signal_succeeded": False,
        "parent_exited": False,
        "fallback_parent_kill": False,
        "succeeded": False,
        "errors": [],
    }
    tree_signal_succeeded = False
    if os.name == "nt":
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        taskkill = system_root / "System32" / "taskkill.exe"
        command = [str(taskkill), "/PID", str(process.pid), "/T", "/F"]
        cleanup["command"] = command
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=15.0,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            cleanup["taskkill_return_code"] = completed.returncode
            output = (completed.stdout or b"").decode("utf-8", errors="replace")
            if output:
                cleanup["taskkill_output"] = output[-4096:]
            tree_signal_succeeded = completed.returncode == 0
            if not tree_signal_succeeded:
                _append_cleanup_error(
                    cleanup,
                    f"taskkill returned {completed.returncode}",
                )
        except (OSError, subprocess.SubprocessError) as error:
            cleanup["taskkill_return_code"] = None
            _append_cleanup_error(cleanup, f"taskkill failed: {error}")
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            tree_signal_succeeded = True
        except OSError as error:
            _append_cleanup_error(cleanup, f"process-group kill failed: {error}")

    parent_exited = False
    try:
        process.wait(timeout=5.0)
        parent_exited = True
    except subprocess.TimeoutExpired:
        _append_cleanup_error(cleanup, "launcher parent remained alive after tree kill")

    cleanup["parent_exited"] = parent_exited
    cleanup["parent_return_code"] = process.returncode
    if (
        os.name == "nt"
        and not tree_signal_succeeded
        and parent_exited
        and process.returncode == 0
    ):
        # The synchronous wrapper can finish normally in the small race between
        # the initial timeout and taskkill. Its child has completed in that case.
        cleanup["normal_exit_race"] = True
        tree_signal_succeeded = True
    cleanup["tree_signal_succeeded"] = tree_signal_succeeded
    cleanup["succeeded"] = tree_signal_succeeded and parent_exited

    if not cleanup["succeeded"]:
        try:
            if process.poll() is None:
                process.kill()
                cleanup["fallback_parent_kill"] = True
            process.wait(timeout=5.0)
            cleanup["parent_return_code"] = process.returncode
        except (OSError, subprocess.SubprocessError) as error:
            _append_cleanup_error(cleanup, f"fallback parent kill failed: {error}")
    return cleanup


def run_live(
    port: str,
    raw_log: BinaryIO,
    *,
    ready_timeout: float,
    ping_timeout: float,
    model_timeout: float,
    negative_grace: float,
    launcher: Path | None,
    launcher_log: BinaryIO | None,
) -> dict[str, object]:
    report = _new_report("live")
    report["serial"]["port"] = port  # type: ignore[index]
    process: subprocess.Popen[bytes] | None = None
    serial_port: object | None = None
    try:
        serial_port = _open_serial(port)
        source = SerialLineSource(serial_port, raw_log)
        if launcher is not None:
            if launcher_log is None:
                raise AcceptanceFailure("internal error: launcher log is not open")
            process = subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(launcher),
                    "-Mode",
                    "generate",
                ],
                cwd=str(launcher.parent),
                stdin=subprocess.DEVNULL,
                stdout=launcher_log,
                stderr=subprocess.STDOUT,
                **_launcher_process_group_options(),
            )
            report["launcher"] = {"script": str(launcher), "pid": process.pid}
        startup_observed = _wait_for_ready(source, ready_timeout, process)
        report["startup_observed"] = startup_observed
        report["ready_observed"] = startup_observed[-1]

        for case in ACCEPTANCE_CASES:
            timeout = ping_timeout if case.timeout_class == "ping" else model_timeout
            result = _run_live_case(source, serial_port, case, timeout)
            report["cases"].append(asdict(result))  # type: ignore[union-attr]
            if not result.passed:
                raise AcceptanceFailure(f"{case.name}: {result.failure}")
            if case.name == "reject_out_of_range_id":
                _check_negative_grace(source, negative_grace)
        _check_negative_grace(source, negative_grace)
        report["passed"] = True
    except AcceptanceFailure as error:
        report["failure"] = str(error)
    finally:
        if serial_port is not None:
            serial_port.close()  # type: ignore[attr-defined]
        if process is not None:
            launcher_timed_out = False
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                launcher_timed_out = True
                cleanup = _terminate_launcher_tree(process)
                report["launcher"]["cleanup"] = cleanup  # type: ignore[index]
                report["passed"] = False
                timeout_failure = "launcher did not exit within 10.0s"
                if not cleanup["succeeded"]:
                    cleanup_errors = "; ".join(cleanup["errors"])  # type: ignore[arg-type]
                    timeout_failure += (
                        "; process-tree cleanup failed"
                        + (f": {cleanup_errors}" if cleanup_errors else "")
                    )
                if report["failure"] is None:
                    report["failure"] = timeout_failure
                elif not cleanup["succeeded"]:
                    report["failure"] = f"{report['failure']}; {timeout_failure}"
            report["launcher"]["return_code"] = process.returncode  # type: ignore[index]
            if process.returncode != 0 and not launcher_timed_out:
                report["passed"] = False
                if report["failure"] is None:
                    report["failure"] = (
                        f"launcher exited with nonzero return code {process.returncode}"
                    )
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def _bounded_timeout(parser: argparse.ArgumentParser, name: str, value: float, maximum: float) -> float:
    if not (0.0 < value <= maximum):
        parser.error(f"{name} must be > 0 and <= {maximum:g} seconds")
    return value


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(tempfile.gettempdir()) / "qot_uart_acceptance" / stamp


def _prepare_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    for name in ("uart_raw.bin", "acceptance.json", "launcher.log"):
        if (resolved / name).exists():
            raise AcceptanceFailure(f"refusing to overwrite {resolved / name}")
    return resolved


def _write_report(path: Path, report: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--port", help="live serial port, for example COM230")
    mode.add_argument("--verify-transcript", type=Path, help="replay raw UART bytes")
    parser.add_argument(
        "--launch-workbench",
        type=Path,
        help="optionally launch run_board_smoke.ps1 -Mode generate after opening UART",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=MAX_READY_TIMEOUT_SECONDS,
        help="startup budget including cold JTAG runtime load (default/max: 3600s)",
    )
    parser.add_argument("--ping-timeout", type=float, default=10.0)
    parser.add_argument("--model-timeout", type=float, default=1_800.0)
    parser.add_argument(
        "--negative-grace",
        type=float,
        default=1.0,
        help="bounded quiet period after the out-of-range ERROR (max 10s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verify_transcript is not None and args.launch_workbench is not None:
        parser.error("--launch-workbench is valid only with --port")
    ready_timeout = _bounded_timeout(
        parser, "--ready-timeout", args.ready_timeout, MAX_READY_TIMEOUT_SECONDS
    )
    ping_timeout = _bounded_timeout(
        parser, "--ping-timeout", args.ping_timeout, MAX_COMMAND_TIMEOUT_SECONDS
    )
    model_timeout = _bounded_timeout(
        parser, "--model-timeout", args.model_timeout, MAX_COMMAND_TIMEOUT_SECONDS
    )
    negative_grace = _bounded_timeout(
        parser, "--negative-grace", args.negative_grace, 10.0
    )

    try:
        output_dir = _prepare_output_dir(args.output_dir or _default_output_dir())
        raw_path = output_dir / "uart_raw.bin"
        report_path = output_dir / "acceptance.json"
        if args.verify_transcript is not None:
            transcript_path = args.verify_transcript.resolve()
            data = transcript_path.read_bytes()
            with raw_path.open("xb") as stream:
                stream.write(data)
            report = verify_transcript_bytes(data)
            report["input_transcript"] = str(transcript_path)
        else:
            launcher = (
                _resolve_launcher(args.launch_workbench)
                if args.launch_workbench is not None
                else None
            )
            launcher_path = output_dir / "launcher.log"
            launcher_stream = launcher_path.open("xb") if launcher is not None else None
            try:
                with raw_path.open("xb") as raw_stream:
                    report = run_live(
                        args.port,
                        raw_stream,
                        ready_timeout=ready_timeout,
                        ping_timeout=ping_timeout,
                        model_timeout=model_timeout,
                        negative_grace=negative_grace,
                        launcher=launcher,
                        launcher_log=launcher_stream,
                    )
            finally:
                if launcher_stream is not None:
                    launcher_stream.close()
        report["artifacts"] = {
            "uart_raw": str(raw_path),
            "uart_raw_bytes": raw_path.stat().st_size,
            "uart_raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "report": str(report_path),
        }
        _write_report(report_path, report)
    except (AcceptanceFailure, OSError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    if report["passed"]:
        print(f"PASS: {len(report['cases'])} single-flight UART acceptance cases")
        print(f"report: {report_path}")
        print(f"raw UART: {raw_path}")
        return 0
    print(f"FAIL: {report['failure']}", file=sys.stderr)
    print(f"report: {report_path}", file=sys.stderr)
    print(f"raw UART: {raw_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
