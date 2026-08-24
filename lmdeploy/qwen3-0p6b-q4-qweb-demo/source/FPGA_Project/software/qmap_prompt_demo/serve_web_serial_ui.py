from __future__ import annotations

import argparse
import functools
import http.server
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve the local Qwen3 FPGA Web Serial demo UI"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    web_root = Path(__file__).resolve().parent / "web_serial_ui"
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(web_root)
    )
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Qwen3 FPGA Web Serial UI: http://{args.host}:{args.port}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
