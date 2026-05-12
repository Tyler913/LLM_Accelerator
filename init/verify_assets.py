from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("model_assets.json")


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_file(base_dir: Path, entry: dict, strict_sizes: bool) -> list[str]:
    errors = []
    path = base_dir / entry["path"]
    required = entry.get("required", False)

    if not path.exists():
        if required:
            errors.append(f"missing required file: {path}")
        return errors

    size = path.stat().st_size
    min_bytes = entry.get("min_bytes")
    if min_bytes is not None and size < min_bytes:
        errors.append(f"{path} is too small: {size} bytes < {min_bytes} bytes")

    reference_bytes = entry.get("reference_bytes")
    if strict_sizes and reference_bytes is not None and size != reference_bytes:
        errors.append(
            f"{path} size differs from reference: {size} bytes != {reference_bytes} bytes"
        )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify local model assets listed in init/model_assets.json."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the model asset manifest.",
    )
    parser.add_argument(
        "--strict-sizes",
        action="store_true",
        help="Require exact reference byte sizes from the manifest.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    local_dir = REPO_ROOT / manifest["model"]["local_dir"]

    errors = []
    print(f"Checking assets under: {local_dir}")

    for entry in manifest["files"]:
        path = local_dir / entry["path"]
        file_errors = check_file(local_dir, entry, args.strict_sizes)
        if file_errors:
            errors.extend(file_errors)
            print(f"FAIL {path}")
        elif path.exists():
            print(f"OK   {path} ({path.stat().st_size} bytes)")
        else:
            print(f"SKIP {path} (optional)")

    if errors:
        print()
        print("Asset verification failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("Asset verification passed.")


if __name__ == "__main__":
    main()
