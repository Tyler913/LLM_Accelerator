from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify every file in a packaged Qwen3 FPGA board release"
    )
    parser.add_argument(
        "package",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    package = args.package.resolve()
    manifest_path = package / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_paths = set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        expected_paths.add(relative.as_posix())
        path = package / relative
        if not path.is_file():
            raise FileNotFoundError(f"Packaged file is missing: {relative}")
        if path.stat().st_size != item["nbytes"]:
            raise RuntimeError(f"Packaged file size mismatch: {relative}")
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Packaged file SHA256 mismatch: {relative}")

    actual_paths = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name != "package_manifest.json"
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise RuntimeError(
            f"Package inventory mismatch; missing={missing}, extra={extra}"
        )

    audit = json.loads(
        (package / "board_readiness_audit.json").read_text(encoding="utf-8")
    )
    required_state = "BOARD_TEST_READY_NOT_YET_HARDWARE_VALIDATED"
    if audit.get("release_state") != required_state:
        raise RuntimeError(
            f"Unexpected board release state: {audit.get('release_state')}"
        )

    readme_path = package / "BOARD_TEST_README.md"
    readme = readme_path.read_text(encoding="utf-8")
    invalid_controls = [
        (index, ord(character))
        for index, character in enumerate(readme)
        if ord(character) < 32 and character not in "\t\n\r"
    ]
    if invalid_controls:
        raise RuntimeError(
            "BOARD_TEST_README.md contains disallowed control characters: "
            f"{invalid_controls[:8]}"
        )

    required_commands = (
        r"conda run -n llm_fpga python .\verify_board_release.py",
        r".\run_board_smoke.ps1 -Mode control",
        r".\run_board_smoke.ps1 -Mode model",
    )
    for command in required_commands:
        if command not in readme:
            raise RuntimeError(
                f"BOARD_TEST_README.md is missing command: {command}"
            )

    print(
        "PASS board release package: "
        f"{manifest['file_count']} files, {manifest['total_file_bytes']} bytes"
    )
    print(f"release_state={required_state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
