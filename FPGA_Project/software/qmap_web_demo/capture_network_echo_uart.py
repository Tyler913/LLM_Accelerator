#!/usr/bin/env python3
"""Capture a bounded network-echo UART startup without sending serial data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


READY_RECORD = "TCP echo server started @ port 7"
FAILURE_PREFIXES = (
    "Error adding N/W interface",
    "Error creating PCB",
    "Unable to bind to port",
    "Out of memory while tcp_listen",
    "Auto negotiation error",
    "Phy setup error",
    "Phy setup failure",
    "Ethernet Link down",
)


def _lines(pending: bytearray) -> list[str]:
    result: list[str] = []
    while True:
        try:
            index = pending.index(0x0A)
        except ValueError:
            break
        raw = bytes(pending[:index])
        del pending[: index + 1]
        if raw.endswith(b"\r"):
            raw = raw[:-1]
        result.append(raw.decode("utf-8", errors="replace").lstrip("\r"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--linger", type=float, default=2.0)
    args = parser.parse_args()
    if not (0.0 < args.timeout <= 600.0):
        parser.error("--timeout must be > 0 and <= 600 seconds")
    if not (0.0 <= args.linger <= 10.0):
        parser.error("--linger must be between 0 and 10 seconds")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"ERROR: refusing to overwrite {output}", file=sys.stderr)
        return 2
    try:
        import serial  # type: ignore[import-not-found]
    except ImportError:
        print("ERROR: pyserial is required", file=sys.stderr)
        return 2

    pending = bytearray()
    ready_at: float | None = None
    deadline = time.monotonic() + args.timeout
    failure: str | None = None
    try:
        with output.open("xb") as raw_stream, serial.Serial(
            port=args.port,
            baudrate=115_200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        ) as uart:
            uart.reset_input_buffer()
            while time.monotonic() < deadline:
                data = uart.read(4096)
                if data:
                    raw_stream.write(data)
                    raw_stream.flush()
                    pending.extend(data)
                    for line in _lines(pending):
                        print(line, flush=True)
                        normalized = line.strip()
                        if normalized == READY_RECORD and ready_at is None:
                            ready_at = time.monotonic()
                        if any(normalized.startswith(prefix)
                               for prefix in FAILURE_PREFIXES):
                            failure = normalized
                            break
                if failure is not None:
                    break
                if ready_at is not None and time.monotonic() - ready_at >= args.linger:
                    break
    except (OSError, serial.SerialException) as error:
        print(f"ERROR: UART capture failed: {error}", file=sys.stderr)
        return 2

    if failure is not None:
        print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    if ready_at is None:
        print(
            f"FAIL: did not observe {READY_RECORD!r} within {args.timeout:.1f}s",
            file=sys.stderr,
        )
        return 1
    print(f"PASS captured network echo READY: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
