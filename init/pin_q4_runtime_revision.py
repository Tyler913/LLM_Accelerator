#!/usr/bin/env python3
"""Pin the Q4 asset manifest to an exact Hugging Face commit revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


DEFAULT_MANIFEST = Path(__file__).with_name("q4_runtime_assets.json")
REVISION_RE = re.compile(r"[0-9a-fA-F]{40}\Z")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", help="Exact commit SHA returned by Hugging Face")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Q4 asset manifest to update.",
    )
    args = parser.parse_args()
    revision = args.revision.strip().lower()
    if not REVISION_RE.fullmatch(revision):
        parser.error("revision must be the 40-character hexadecimal Hub commit SHA")

    path = args.manifest.resolve()
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    previous = str(manifest["artifact"].get("revision", ""))
    manifest["artifact"]["revision"] = revision

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"Pinned {path}")
    print(f"Previous revision: {previous}")
    print(f"New revision:      {revision}")


if __name__ == "__main__":
    main()
