#!/usr/bin/env python3
"""Download the exact Q4 runtime segment set named by the GitHub manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_MANIFEST = Path(__file__).with_name("q4_runtime_assets.json")
SEGMENT_NAME_RE = re.compile(r"qwen3_runtime_[0-9]{2}\.bin\Z")
PINNED_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}\Z")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_contract(asset_manifest_path: Path) -> tuple[dict, dict, list[str]]:
    asset_manifest = load_json(asset_manifest_path)
    artifact = asset_manifest["artifact"]
    segment_manifest_path = resolve_repo_path(artifact["segment_manifest"])
    segment_manifest = load_json(segment_manifest_path)
    segments = segment_manifest.get("segments", [])

    expected_count = int(artifact["expected_segment_count"])
    expected_bytes = int(artifact["expected_total_bytes"])
    expected_names = [f"qwen3_runtime_{index:02d}.bin" for index in range(expected_count)]
    actual_names = [str(segment.get("file", "")) for segment in segments]

    if len(segments) != expected_count:
        raise ValueError(
            f"segment manifest has {len(segments)} entries; expected {expected_count}"
        )
    if actual_names != expected_names:
        raise ValueError("segment manifest names or ordering do not match _00 through _60")
    if any(not SEGMENT_NAME_RE.fullmatch(name) for name in actual_names):
        raise ValueError("segment manifest contains an unsafe or malformed file name")
    manifest_count = int(segment_manifest.get("segment_count", -1))
    manifest_bytes = int(segment_manifest.get("total_segment_bytes", -1))
    listed_bytes = sum(int(segment["nbytes"]) for segment in segments)
    if manifest_count != expected_count:
        raise ValueError(
            f"segment_count is {manifest_count}; expected {expected_count}"
        )
    if manifest_bytes != expected_bytes or listed_bytes != expected_bytes:
        raise ValueError(
            "segment byte contract mismatch: "
            f"header={manifest_bytes}, entries={listed_bytes}, expected={expected_bytes}"
        )

    return artifact, segment_manifest, expected_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ASSET_MANIFEST,
        help="Q4 asset manifest (default: init/q4_runtime_assets.json).",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Override the output directory; relative paths resolve from the repo root.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Override the Hugging Face revision recorded in the asset manifest.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only an already-populated local Hugging Face cache.",
    )
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="Permit a floating branch revision during the one-time initial upload only.",
    )
    args = parser.parse_args()

    asset_manifest_path = args.manifest.resolve()
    artifact, _segment_manifest, names = load_contract(asset_manifest_path)
    local_dir = args.local_dir or Path(artifact["local_dir"])
    local_dir = resolve_repo_path(local_dir).resolve()
    revision = args.revision or artifact["revision"]
    revision_is_pinned = PINNED_REVISION_RE.fullmatch(str(revision)) is not None
    initial_main_is_allowed = args.allow_unpinned and str(revision) == "main"
    if not revision_is_pinned and not initial_main_is_allowed:
        parser.error(
            "Q4 runtime revision must be an exact commit SHA; the only exception "
            "is literal main together with --allow-unpinned during the controlled "
            "first publication"
        )

    print(f"Repository: {artifact['repo_id']}")
    print(f"Repo type:  {artifact.get('repo_type', 'model')}")
    print(f"Revision:   {revision}")
    print(f"Local dir:  {local_dir}")
    print(f"Segments:   {len(names)}")

    snapshot_download(
        repo_id=artifact["repo_id"],
        repo_type=artifact.get("repo_type", "model"),
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=names,
        local_files_only=args.local_files_only,
    )
    print("Q4 runtime download complete. Run init/verify_q4_runtime.py next.")


if __name__ == "__main__":
    main()
