"""Fetch and verify the pinned Unicode-table source crates into ignored Temp."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[4]
LOCK_PATH = Path(__file__).resolve().with_name("unicode_sources.lock.json")
DEFAULT_OUTPUT = REPO_ROOT / "Temp" / "qmap_prompt_demo_unicode_sources"


class FetchError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            target = (destination / member.name).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise FetchError(f"Unsafe archive member {member.name!r}")
            if member.issym() or member.islnk():
                raise FetchError(f"Links are not accepted in source archive: {member.name!r}")
        stream.extractall(destination, filter="data")


def fetch_package(output_dir: Path, name: str, record: dict[str, object]) -> Path:
    version = str(record["version"])
    package_dir = output_dir / f"{name}-{version}"
    archive = output_dir / f"{name}-{version}.crate"
    expected_archive_hash = str(record["archive_sha256"]).upper()

    if not archive.is_file() or sha256_file(archive) != expected_archive_hash:
        temporary = archive.with_suffix(archive.suffix + ".tmp")
        with urllib.request.urlopen(str(record["url"]), timeout=60) as response:
            temporary.write_bytes(response.read())
        if sha256_file(temporary) != expected_archive_hash:
            temporary.unlink(missing_ok=True)
            raise FetchError(f"Archive SHA-256 mismatch for {name} {version}")
        temporary.replace(archive)

    if package_dir.exists():
        shutil.rmtree(package_dir)
    extraction_root = output_dir / f".{name}-{version}.extracting"
    if extraction_root.exists():
        shutil.rmtree(extraction_root)
    extraction_root.mkdir(parents=True)
    safe_extract(archive, extraction_root)
    extracted = extraction_root / f"{name}-{version}"
    if not extracted.is_dir():
        raise FetchError(f"Archive {archive} lacks expected root directory")
    extracted.replace(package_dir)
    extraction_root.rmdir()

    for relative_path, expected_hash in dict(record["files"]).items():
        path = package_dir / relative_path
        if not path.is_file() or sha256_file(path) != str(expected_hash).upper():
            raise FetchError(f"Pinned source file mismatch: {name}/{relative_path}")
    return package_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        package_paths = {
            name: fetch_package(output_dir, name, record)
            for name, record in sorted(lock["packages"].items())
        }
    except (FetchError, OSError, KeyError, ValueError, tarfile.TarError) as error:
        print(f"FAIL Unicode source fetch: {error}", file=sys.stderr)
        return 1

    print("PASS Unicode source fetch")
    for name, path in package_paths.items():
        print(f"package={name} path={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
