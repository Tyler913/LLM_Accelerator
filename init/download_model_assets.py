from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("model_assets.json")


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def manifest_patterns(manifest: dict, skip_large: bool) -> list[str]:
    patterns = []
    for entry in manifest["files"]:
        if skip_large and entry.get("large", False):
            continue
        patterns.append(entry["path"])
    return patterns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download local model assets listed in init/model_assets.json."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the model asset manifest.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Override the local output directory from the manifest.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Override the Hugging Face revision from the manifest.",
    )
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="Download only small config/tokenizer files, not model weights.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    model_info = manifest["model"]
    repo_id = model_info["repo_id"]
    repo_type = model_info.get("repo_type", "model")
    revision = args.revision or model_info.get("revision", "main")
    local_dir = args.local_dir or REPO_ROOT / model_info["local_dir"]
    local_dir = local_dir if local_dir.is_absolute() else REPO_ROOT / local_dir
    patterns = manifest_patterns(manifest, skip_large=args.skip_large)

    print(f"Repository: {repo_id}")
    print(f"Repo type:  {repo_type}")
    print(f"Revision:   {revision}")
    print(f"Local dir:  {local_dir}")
    print("Files:")
    for pattern in patterns:
        print(f"  - {pattern}")

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=patterns,
    )

    print("Download complete.")


if __name__ == "__main__":
    main()
