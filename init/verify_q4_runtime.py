#!/usr/bin/env python3
"""Verify the complete Q4 runtime set against its canonical GitHub manifest."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download

from download_q4_runtime import (
    DEFAULT_ASSET_MANIFEST,
    PINNED_REVISION_RE,
    load_contract,
    resolve_repo_path,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        help="Override the runtime directory; relative paths resolve from the repo root.",
    )
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="Permit the temporary main revision before the first Hub upload.",
    )
    parser.add_argument(
        "--from-hub",
        action="store_true",
        help="Download the pinned Hub revision into a temporary directory and verify it.",
    )
    args = parser.parse_args()

    asset_manifest_path = args.manifest.resolve()
    artifact, segment_manifest, expected_names = load_contract(asset_manifest_path)
    revision = str(artifact.get("revision", ""))
    revision_is_pinned = PINNED_REVISION_RE.fullmatch(revision) is not None
    initial_main_is_allowed = args.allow_unpinned and revision == "main"
    if not revision_is_pinned and not initial_main_is_allowed:
        parser.error(
            "Q4 runtime revision must be an exact commit SHA; the only exception "
            "is literal main together with --allow-unpinned while preparing the "
            "first Hub upload"
        )
    if args.from_hub and args.local_dir is not None:
        parser.error("--from-hub and --local-dir cannot be used together")
    if args.from_hub and args.allow_unpinned:
        parser.error("--from-hub requires an exact pinned revision")

    canonical_manifest = resolve_repo_path(artifact["segment_manifest"]).resolve()
    errors: list[str] = []

    with ExitStack() as stack:
        if args.from_hub:
            temporary = Path(
                stack.enter_context(TemporaryDirectory(prefix="q4-hub-verify-"))
            )
            runtime_dir = temporary / "model"
            print(
                f"Downloading pinned Hub revision {artifact['revision']} "
                f"from {artifact['repo_id']}"
            )
            snapshot_download(
                repo_id=artifact["repo_id"],
                repo_type=artifact.get("repo_type", "model"),
                revision=artifact["revision"],
                local_dir=str(runtime_dir),
                allow_patterns=expected_names,
            )
        else:
            runtime_dir = resolve_repo_path(
                args.local_dir or artifact["local_dir"]
            ).resolve()

        if not runtime_dir.is_dir():
            raise SystemExit(f"Q4 runtime directory is missing: {runtime_dir}")
        manifest_hash = sha256_file(canonical_manifest)
        if manifest_hash != artifact["segment_manifest_sha256"]:
            errors.append(
                "canonical segment manifest SHA-256 mismatch: "
                f"{manifest_hash} != {artifact['segment_manifest_sha256']}"
            )

        actual_names = sorted(
            path.name for path in runtime_dir.glob("qwen3_runtime_*.bin")
        )
        if actual_names != expected_names:
            missing = sorted(set(expected_names) - set(actual_names))
            extra = sorted(set(actual_names) - set(expected_names))
            if missing:
                errors.append("missing segments: " + ", ".join(missing))
            if extra:
                errors.append("unexpected segments: " + ", ".join(extra))

        aperture = artifact["pl_ddr_aperture"]
        aperture_start = int(aperture["start"], 0)
        aperture_end = int(aperture["end_exclusive"], 0)
        alignment = int(aperture["alignment_bytes"])
        previous_end: int | None = None
        checked_bytes = 0

        for expected_index, segment in enumerate(segment_manifest["segments"]):
            name = segment["file"]
            path = runtime_dir / name
            address = int(segment["address"])
            nbytes = int(segment["nbytes"])
            end = address + nbytes
            if int(segment["index"]) != expected_index:
                errors.append(f"{name}: index is not {expected_index}")
            if address % alignment != 0 or nbytes % alignment != 0:
                errors.append(f"{name}: address/size is not {alignment}-byte aligned")
            if address < aperture_start or end > aperture_end:
                errors.append(f"{name}: range is outside the declared PL-DDR aperture")
            if previous_end is not None and address < previous_end:
                errors.append(f"{name}: range overlaps the preceding segment")
            previous_end = end

            if not path.is_file():
                continue
            size = path.stat().st_size
            checked_bytes += size
            if size != nbytes:
                errors.append(f"{name}: {size} bytes != expected {nbytes}")
                continue
            digest = sha256_file(path)
            if digest != segment["sha256"]:
                errors.append(f"{name}: SHA-256 mismatch ({digest})")
            else:
                print(f"OK {expected_index:02d} {name} {size} bytes")

        expected_total = int(artifact["expected_total_bytes"])
        if checked_bytes != expected_total:
            errors.append(
                f"runtime byte total is {checked_bytes}; expected {expected_total}"
            )

        if errors:
            print("Q4 runtime verification failed:")
            for error in errors:
                print(f"  - {error}")
            raise SystemExit(1)

        source = "pinned Hugging Face revision" if args.from_hub else str(runtime_dir)
        print(
            "PASS Q4 runtime: "
            f"{len(expected_names)} segments / {checked_bytes} bytes / "
            f"all SHA-256 values match ({source})"
        )


if __name__ == "__main__":
    main()
