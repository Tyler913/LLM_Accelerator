#!/usr/bin/env python3
"""Capture one portable QWEB startup from UART and preserve its raw evidence."""

from __future__ import annotations

import argparse
import codecs
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid


READY_RE = re.compile(rb"QWEB READY (http://[0-9]+(?:\.[0-9]+){3}:[0-9]+/)")
MAX_CAPTURE_BYTES = 4 * 1024 * 1024
LAUNCH_REPORT_WAIT_SECONDS = 60.0
HEARTBEAT_INTERVAL_SECONDS = 0.5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict, *, exclusive: bool = True) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if exclusive and (path.exists() or temporary.exists()):
        raise RuntimeError(f"refusing to overwrite evidence: {path}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
    temporary.replace(path)


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp has no UTC offset")
    return parsed.astimezone(timezone.utc)


def wait_for_launch_binding(
    output_dir: Path,
    *,
    capture_id: str,
    capture_claim_sha256: str,
    capture_started: datetime,
    ready_utc: datetime,
) -> dict:
    launch_path = output_dir / "launch.json"
    deadline = time.monotonic() + LAUNCH_REPORT_WAIT_SECONDS
    while not launch_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not launch_path.is_file():
        raise RuntimeError(
            f"launch.json did not appear within {LAUNCH_REPORT_WAIT_SECONDS:.0f}s"
        )
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if launch.get("schema_version") != 1 or launch.get("tool") != "run_demo.ps1":
        raise RuntimeError("launch.json has an unsupported schema or tool")
    if launch.get("passed") is not True:
        raise RuntimeError(f"bound launch failed: {launch.get('failure')}")
    if launch.get("capture_id") != capture_id:
        raise RuntimeError("launch.json belongs to another UART capture")
    if launch.get("capture_claim_sha256") != capture_claim_sha256:
        raise RuntimeError("launch.json capture-claim hash mismatch")
    if Path(str(launch.get("evidence_directory", ""))).resolve() != output_dir:
        raise RuntimeError("launch.json names another evidence directory")
    run_id = str(launch.get("run_id", ""))
    uuid.UUID(run_id)
    launch_started = parse_utc(str(launch["started_utc"]))
    launch_completed = parse_utc(str(launch["completed_utc"]))
    if launch_started < capture_started:
        raise RuntimeError("launch predates this UART capture")
    if ready_utc < launch_started:
        raise RuntimeError("QWEB READY predates the bound physical launch")
    if launch_completed < launch_started:
        raise RuntimeError("launch completion predates launch start")
    if launch.get("runtime_segments") != 61 or launch.get("runtime_bytes") != 394_547_200:
        raise RuntimeError("launch.json does not bind the complete Q4 runtime")
    return {
        "run_id": run_id,
        "launch_started_utc": launch["started_utc"],
        "launch_completed_utc": launch["completed_utc"],
        "launch_report_sha256": hashlib.sha256(launch_path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="UART port, for example COM230")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--linger", type=float, default=2.0)
    args = parser.parse_args()
    if not (1.0 <= args.timeout <= 3600.0):
        parser.error("--timeout must be between 1 and 3600 seconds")
    if not (0.0 <= args.linger <= 10.0):
        parser.error("--linger must be between 0 and 10 seconds")

    try:
        import serial
    except ImportError as error:
        print("ERROR: pyserial is missing; install init/requirements.txt", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"ERROR: evidence directory already exists: {output_dir}", file=sys.stderr)
        return 2

    raw_path = output_dir / "uart_raw.bin"
    capture_claim_path = output_dir / "capture.claim.json"
    heartbeat_path = output_dir / "capture.heartbeat.json"
    text_path = output_dir / "uart.txt"
    report_path = output_dir / "startup.json"
    capture_id = str(uuid.uuid4())
    capture_started = datetime.now(timezone.utc)
    started_utc = capture_started.isoformat()
    started = time.monotonic()
    ready_at: float | None = None
    captured = bytearray()
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    interrupted = False
    capture_error: str | None = None
    ready_utc: datetime | None = None
    capture_claim_sha256: str | None = None
    heartbeat_sequence = 0

    print(f"Capturing {args.port} at 115200 8N1")
    print(f"Evidence directory: {output_dir}")
    try:
        # Open the UART before publishing uart_raw.bin. The launcher uses the
        # file's presence as the hand-off that capture really owns the port.
        with serial.Serial(
            args.port,
            baudrate=115_200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
        ) as port:
            capture_claim = {
                "schema_version": 1,
                "tool": "capture_qweb_uart.py",
                "state": "claimed",
                "capture_id": capture_id,
                "pid": os.getpid(),
                "started_utc": started_utc,
                "port": args.port,
                "baud_rate": 115200,
                "evidence_directory": str(output_dir),
            }
            write_json(capture_claim_path, capture_claim)
            capture_claim_sha256 = hashlib.sha256(
                capture_claim_path.read_bytes()
            ).hexdigest()
            heartbeat_sequence += 1
            write_json(
                heartbeat_path,
                {
                    "schema_version": 1,
                    "tool": "capture_qweb_uart.py",
                    "state": "capturing",
                    "capture_id": capture_id,
                    "pid": os.getpid(),
                    "sequence": heartbeat_sequence,
                    "heartbeat_utc": utc_now(),
                    "evidence_directory": str(output_dir),
                },
                exclusive=False,
            )
            next_heartbeat = time.monotonic() + HEARTBEAT_INTERVAL_SECONDS
            with raw_path.open("xb", buffering=0) as raw:
                while True:
                    now = time.monotonic()
                    if now >= next_heartbeat:
                        heartbeat_sequence += 1
                        write_json(
                            heartbeat_path,
                            {
                                "schema_version": 1,
                                "tool": "capture_qweb_uart.py",
                                "state": "capturing",
                                "capture_id": capture_id,
                                "pid": os.getpid(),
                                "sequence": heartbeat_sequence,
                                "heartbeat_utc": utc_now(),
                                "evidence_directory": str(output_dir),
                            },
                            exclusive=False,
                        )
                        next_heartbeat = now + HEARTBEAT_INTERVAL_SECONDS
                    if now - started >= args.timeout:
                        break
                    if ready_at is not None and now - ready_at >= args.linger:
                        break
                    chunk = port.read(max(1, min(port.in_waiting or 1, 65536)))
                    if not chunk:
                        continue
                    if len(captured) + len(chunk) > MAX_CAPTURE_BYTES:
                        raise RuntimeError(
                            f"UART capture exceeded {MAX_CAPTURE_BYTES} bytes"
                        )
                    raw.write(chunk)
                    captured.extend(chunk)
                    rendered = decoder.decode(chunk)
                    if rendered:
                        sys.stdout.write(rendered)
                        sys.stdout.flush()
                    if ready_at is None and READY_RE.search(captured):
                        ready_at = time.monotonic()
                        ready_utc = datetime.now(timezone.utc)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as error:
        capture_error = str(error)
        print(f"ERROR: UART capture failed: {error}", file=sys.stderr)

    if capture_claim_sha256 is not None:
        try:
            heartbeat_sequence += 1
            write_json(
                heartbeat_path,
                {
                    "schema_version": 1,
                    "tool": "capture_qweb_uart.py",
                    "state": "capture_closed",
                    "capture_id": capture_id,
                    "pid": os.getpid(),
                    "sequence": heartbeat_sequence,
                    "heartbeat_utc": utc_now(),
                    "evidence_directory": str(output_dir),
                },
                exclusive=False,
            )
        except Exception as error:
            if capture_error is None:
                capture_error = f"could not close capture heartbeat: {error}"
            print(f"ERROR: heartbeat close failed: {error}", file=sys.stderr)

    decoded = bytes(captured).decode("utf-8", errors="replace")
    text_path.write_text(decoded, encoding="utf-8")
    match = READY_RE.search(captured)
    ready_url = match.group(1).decode("ascii") if match else None
    launch_binding = None
    if (
        ready_url is not None
        and ready_utc is not None
        and capture_claim_sha256 is not None
        and capture_error is None
        and not interrupted
    ):
        try:
            launch_binding = wait_for_launch_binding(
                output_dir,
                capture_id=capture_id,
                capture_claim_sha256=capture_claim_sha256,
                capture_started=capture_started,
                ready_utc=ready_utc,
            )
        except Exception as error:
            capture_error = str(error)
            print(f"ERROR: launch binding failed: {error}", file=sys.stderr)
    passed = (
        ready_url is not None
        and ready_utc is not None
        and launch_binding is not None
        and capture_error is None
        and not interrupted
    )
    payload = {
        "schema_version": 1,
        "passed": passed,
        "capture_id": capture_id,
        "port": args.port,
        "baud_rate": 115200,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "captured_bytes": len(captured),
        "uart_sha256": hashlib.sha256(captured).hexdigest(),
        "ready_url": ready_url,
        "ready_utc": ready_utc.isoformat() if ready_utc is not None else None,
        "interrupted": interrupted,
        "failure": capture_error,
        "launch_binding": launch_binding,
    }
    write_json(report_path, payload)
    if not passed:
        reason = capture_error or "QWEB READY was not observed"
        print(f"ERROR: {reason}; evidence: {report_path}", file=sys.stderr)
        return 1
    print(f"\nPASS QWEB UART startup: {ready_url}")
    print(f"Evidence: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
